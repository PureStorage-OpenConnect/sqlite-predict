# Releasing sqlite-predict

Releases are tag-driven. Pushing a `v*` tag fires four workflows that
build and publish everything; the manual part is syncing the version,
committing, and tagging.

## Cut a release

1. Sync every manifest to the new version:

   ```sh
   make sync-version V=0.0.2
   ```

   This writes `VERSION` and syncs it into `bindings/rust/Cargo.toml`,
   `bindings/python/pyproject.toml` (PEP 440 form, e.g. `0.0.1-alpha.5`
   becomes `0.0.1a5`), `bindings/python/sqlite_predict/__init__.py`
   (`__version__`), and `bindings/node/package.json` (including the
   per-platform `optionalDependencies` pins). The C header derives its
   version from `VERSION` at build time and needs no edit.

2. Commit the version bump.

3. Tag and push:

   ```sh
   git tag v0.0.2
   git push && git push --tags
   ```

That is the whole procedure. Registry versions (PyPI, crates.io, npm)
are immutable: if a publish job fails partway, fix the cause and cut a
fresh version rather than re-tagging.

## What the tag triggers

All four workflows in `.github/workflows/` fire on every `v*` tag push:

| Workflow | Publishes | Credentials |
| --- | --- | --- |
| `release.yml` | GitHub Release: prebuilt loadables for Linux, macOS, and Windows, the single-file amalgamation (`sqlite-predict.c`), and `SHA256SUMS`, with generated release notes | built-in `GITHUB_TOKEN` |
| `wheels.yml` | Python wheels (manylinux and macOS, compiler-free `pip install sqlite-predict`) plus the sdist, to PyPI | trusted publishing (OIDC), no stored token |
| `npm.yml` | the per-platform binary packages (`sqlite-predict-darwin-arm64`, `sqlite-predict-darwin-x64`, `sqlite-predict-linux-x64`, `sqlite-predict-linux-arm64`, `sqlite-predict-windows-x64`), then the main `sqlite-predict` package with `optionalDependencies` pinned to the same version | `NPM_TOKEN` repo secret |
| `crate.yml` | the `sqlite-predict` crate to crates.io (the crate's `build.rs` compiles the bundled amalgamation on the user's machine) | `CARGO_REGISTRY_TOKEN` repo secret |

`npm.yml` and `crate.yml` also support `workflow_dispatch` with an
explicit version input, for re-running a single registry publish
without a new tag.

## One-time setup (already done, recorded for repo moves)

- **PyPI trusted publisher** on the `sqlite-predict` project: owner
  `PureStorage-OpenConnect`, repository `sqlite-predict`, workflow
  `wheels.yml`, environment `pypi`. A matching `pypi` GitHub
  environment must exist (Settings, then Environments). If the repo
  moves, update the trusted publisher's owner.
- **`NPM_TOKEN`**: an npm automation token for the org account, stored
  as a repo secret.
- **`CARGO_REGISTRY_TOKEN`**: a crates.io API token for the org
  account, stored as a repo secret.

All registry accounts are org-owned (everpure / Pure Storage), never a
personal login, so publishing survives people leaving.

## Names claimed as of alpha.5

The generic `sqlite-predict` name is claimed and functional on all
three flat-namespace registries: PyPI, crates.io, and npm (the main
package plus the five per-platform binary packages listed above).
GitHub Releases need no claim; the repo owns them.

The everpure-branded mirror names (`everpure-sqlite-predict` on PyPI
and crates.io, `@everpure/sqlite-predict` on npm) are unclaimed and low
urgency; the npm scope already belongs to the org.
