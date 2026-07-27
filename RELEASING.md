# Releasing & claiming sqlite-predict

Two separate jobs: **claim the names** (do this before any public mention, so
nobody squats them) and **cut a release** (the functional launch). The generic
`sqlite-predict` name is the one worth claiming everywhere; the everpure-scoped /
prefixed packages are brand mirrors and lower urgency.

Do everything under **org-owned accounts** (everpure / Pure Storage), never a
personal login, so publishing survives people leaving.

---

## Part 1 — Claim the names now

Flat-namespace registries are first-come-first-served, so these three are the
real squat risk. Each artifact below is already built and verified in this repo;
you only need the account. Bump the version and re-run for later releases.

### PyPI — `sqlite-predict`

1. Create a PyPI account, then an API token (Account settings -> API tokens).
2. Build (or reuse the built) sdist and upload:

   ```sh
   make python-src
   python -m build --sdist --outdir bindings/python/dist bindings/python
   python -m twine upload -u __token__ -p pypi-XXXX \
     bindings/python/dist/sqlite_predict-*.tar.gz
   ```

   The sdist claims the name and is installable (it compiles on the user's
   machine). Compiler-free **wheels** come from the tag pipeline in Part 2.

### crates.io — `sqlite-predict`

1. Log in to crates.io with the org GitHub account; create an API token.
2. Publish (the crate compiles the amalgamation itself; fully functional on
   install):

   ```sh
   cargo login <token>
   make rust-src
   cargo publish --manifest-path bindings/rust/Cargo.toml --allow-dirty
   ```

   `--allow-dirty` is needed because `csrc/` is generated (git-ignored); it is
   still included in the published crate via `Cargo.toml`'s `include`.

### npm — `sqlite-predict` (unscoped)

1. `npm login` as the org account.
2. Publish the main package to claim the name:

   ```sh
   cd bindings/node && npm publish --access public
   ```

   NOTE: this **claims the name but is not yet functional.** The package resolves
   its binary from per-platform packages (`sqlite-predict-<platform>`) that must
   be built and published by CI (not wired yet — see "Open follow-ups"). Claim
   now; make it functional before you announce npm.

### GitHub Releases — no account

Nothing to claim; the repo owns it. See Part 2.

---

## Part 2 — Cut a release

Tagging drives everything else automatically (built-in `GITHUB_TOKEN`, no
external secrets for the binaries):

```sh
git tag v0.0.1-alpha.1
git push origin v0.0.1-alpha.1
```

- `release.yml` builds Linux/macOS/Windows loadables + the amalgamation
  (`sqlite-predict.c`) + `SHA256SUMS` and publishes a GitHub Release.
- `wheels.yml` builds the Python wheels + sdist and, once PyPI trusted
  publishing is configured (below), uploads them so `pip install sqlite-predict`
  is compiler-free.

### PyPI trusted publishing (one-time, recommended over tokens)

On PyPI -> the `sqlite-predict` project -> Publishing -> add a trusted publisher:

| field | value |
| --- | --- |
| Owner | `PureStorage-OpenConnect` (or the everpure org, if the repo moves) |
| Repository | `sqlite-predict` |
| Workflow | `wheels.yml` |
| Environment | `pypi` |

Then create a `pypi` GitHub environment (Settings -> Environments). After that,
a `v*` tag publishes wheels with no token stored in GitHub.

---

## Part 3 — Brand mirrors (everpure)

Lower urgency: `@everpure/*` on npm is already yours by scope, and the prefixed
PyPI/crates names aren't the discoverable target. When ready, publish the same
content under the branded names (`everpure-sqlite-predict` on PyPI/crates,
`@everpure/sqlite-predict` on npm) as thin mirror packages. Ask and this can be
wired into the release workflows.

---

## Open follow-ups (not blocking the name claims)

- **npm per-platform binary packages** + a workflow to build/publish them, so
  `npm install sqlite-predict` ships a working binary (today only the name is
  claimable). Same pattern as the wheels.
- **Windows/musl/arm wheel coverage** in `wheels.yml` (currently CPython
  manylinux + macOS).
- The brand mirror packages in Part 3.
