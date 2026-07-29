#!/usr/bin/env node
/* Single source of truth for the version: the VERSION file (SemVer, e.g.
 * 0.0.1-alpha.1). Syncs it into every package manifest so a release bump is one
 * command:
 *
 *   node scripts/sync-version.mjs                # sync the manifests to VERSION
 *   node scripts/sync-version.mjs 0.0.1-alpha.2  # set VERSION first, then sync
 *
 * Cargo.toml and package.json take the SemVer string directly; pyproject.toml
 * takes the PEP 440 form PyPI requires (0.0.1-alpha.1 -> 0.0.1a1). The C header
 * is generated from VERSION at build time, so it needs no edit here.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const versionFile = path.join(root, "VERSION");

let version = process.argv[2];
if (version) fs.writeFileSync(versionFile, version + "\n");
else version = fs.readFileSync(versionFile, "utf8").trim();

if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version)) {
  console.error(`not a SemVer version: ${version}`);
  process.exit(1);
}

/* SemVer prerelease -> PEP 440: 0.0.1-alpha.1 -> 0.0.1a1, 1.2.3-rc.2 -> 1.2.3rc2 */
function pep440(v) {
  const m = v.match(/^(\d+\.\d+\.\d+)(?:-(alpha|beta|rc)\.?(\d+))?$/);
  if (!m) {
    console.error(`cannot map ${v} to a PEP 440 version`);
    process.exit(1);
  }
  const [, base, kind, n] = m;
  return kind ? `${base}${{ alpha: "a", beta: "b", rc: "rc" }[kind]}${n}` : base;
}

function rewriteLine(file, re, replacement, shown) {
  const p = path.join(root, file);
  const before = fs.readFileSync(p, "utf8");
  if (!re.test(before)) {
    console.error(`no version line matched in ${file}`);
    process.exit(1);
  }
  fs.writeFileSync(p, before.replace(re, replacement));
  console.log(`  ${file} -> ${shown}`);
}

rewriteLine("bindings/rust/Cargo.toml", /^version = "[^"]*"/m,
            `version = "${version}"`, version);

const pv = pep440(version);
rewriteLine("bindings/python/pyproject.toml", /^version = "[^"]*"/m,
            `version = "${pv}"`, pv);
rewriteLine("bindings/python/sqlite_predict/__init__.py",
            /^__version__ = "[^"]*"/m, `__version__ = "${pv}"`, pv);

const pkgPath = path.join(root, "bindings", "node", "package.json");
const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8"));
pkg.version = version;
const deps = Object.keys(pkg.optionalDependencies || {});
for (const d of deps) pkg.optionalDependencies[d] = version;
fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + "\n");
console.log(`  bindings/node/package.json -> ${version} (+ ${deps.length} optionalDependencies)`);

/* Promote CHANGELOG's [Unreleased] into a dated section for this version.
 * Idempotent: if the section already exists, leave the file alone. The
 * changelog itself is written with the change; this owns only the
 * release-time promotion, like every other version stamp here. */
const clPath = path.join(root, "CHANGELOG.md");
let cl = fs.readFileSync(clPath, "utf8");
if (cl.includes(`## [${version}]`)) {
  console.log(`  CHANGELOG.md already has a [${version}] section`);
} else {
  if (!/## \[Unreleased\]\n/.test(cl)) {
    console.error("CHANGELOG.md has no [Unreleased] section to promote");
    process.exit(1);
  }
  const unreleasedBody = cl
    .split("## [Unreleased]\n")[1]
    .split(/\n## \[/)[0]
    .trim();
  if (!unreleasedBody) {
    console.error(`CHANGELOG.md [Unreleased] is empty; write the release's
entries before running sync-version`);
    process.exit(1);
  }
  const date = new Date().toISOString().slice(0, 10);
  cl = cl.replace("## [Unreleased]\n",
                  `## [Unreleased]\n\n## [${version}] - ${date}\n`);
  /* retarget the compare links: Unreleased diffs from the new tag, and the
   * new tag diffs from the previous one (the old Unreleased base). */
  const linkRe = /^\[Unreleased\]: (.*)\/compare\/(v[^.]+[^ ]*)\.\.\.HEAD$/m;
  const lm = cl.match(linkRe);
  if (lm) {
    const [, repo, prevTag] = lm;
    cl = cl.replace(linkRe,
      `[Unreleased]: ${repo}/compare/v${version}...HEAD\n` +
      `[${version}]: ${repo}/compare/${prevTag}...v${version}`);
  }
  fs.writeFileSync(clPath, cl);
  console.log(`  CHANGELOG.md -> [${version}] - ${date}`);
}

console.log(`VERSION = ${version}  (PyPI: ${pv})`);
