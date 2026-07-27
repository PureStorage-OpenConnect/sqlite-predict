#!/usr/bin/env node
/* Build the per-platform npm binary packages and keep the main package version
 * in sync, for the npm publish workflow. Two modes:
 *
 *   platform <pkg> <os> <cpu> <ext> <version>
 *       writes npm/<pkg>/ with package.json (os/cpu-gated), an index.js that
 *       exports loadablePath, and the built dist/predict0.<ext> binary.
 *
 *   main <version>
 *       rewrites bindings/node/package.json version + every optionalDependency
 *       to <version>, so the main package pins its platform packages exactly.
 *
 * The main package (bindings/node/index.cjs) resolves a platform package via
 * require(pkg).loadablePath, so each platform package only needs to export that.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const nodeDir = path.join(root, "bindings", "node");
const mainPkgPath = path.join(nodeDir, "package.json");
const mainPkg = JSON.parse(fs.readFileSync(mainPkgPath, "utf8"));

const [mode, ...args] = process.argv.slice(2);

if (mode === "platform") {
  const [pkg, os, cpu, ext, version] = args;
  if (!pkg || !os || !cpu || !ext || !version) {
    console.error("usage: npm-pkg.mjs platform <pkg> <os> <cpu> <ext> <version>");
    process.exit(1);
  }
  const bin = `predict0.${ext}`;
  const src = path.join(root, "dist", bin);
  if (!fs.existsSync(src)) {
    console.error(`missing built binary: ${src} (run 'make loadable' first)`);
    process.exit(1);
  }
  const outDir = path.join(root, "npm", pkg);
  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(outDir, { recursive: true });
  fs.copyFileSync(src, path.join(outDir, bin));
  fs.writeFileSync(
    path.join(outDir, "index.js"),
    'module.exports = { loadablePath: require("node:path").join(__dirname, "predict0") };\n',
  );
  const pj = {
    name: pkg,
    version,
    description: `Prebuilt sqlite-predict loadable extension for ${os}-${cpu}.`,
    license: mainPkg.license,
    repository: mainPkg.repository,
    os: [os],
    cpu: [cpu],
    main: "index.js",
    files: ["index.js", bin],
  };
  fs.writeFileSync(path.join(outDir, "package.json"), JSON.stringify(pj, null, 2) + "\n");
  console.log(`wrote npm/${pkg} (${pkg}@${version}, ${bin})`);
} else if (mode === "main") {
  const [version] = args;
  if (!version) {
    console.error("usage: npm-pkg.mjs main <version>");
    process.exit(1);
  }
  mainPkg.version = version;
  const deps = Object.keys(mainPkg.optionalDependencies || {});
  for (const dep of deps) mainPkg.optionalDependencies[dep] = version;
  fs.writeFileSync(mainPkgPath, JSON.stringify(mainPkg, null, 2) + "\n");
  console.log(`set ${mainPkg.name}@${version} + ${deps.length} optionalDependencies`);
} else {
  console.error("usage: npm-pkg.mjs <platform|main> ...");
  process.exit(1);
}
