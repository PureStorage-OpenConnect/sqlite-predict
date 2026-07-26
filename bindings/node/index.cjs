const fs = require("node:fs");
const path = require("node:path");

// per-platform binary packages, installed as optionalDependencies so npm pulls
// only the one matching the host (the esbuild / sqlite-vec pattern)
const PLATFORM_PACKAGES = {
  "darwin-arm64": "sqlite-predict-darwin-arm64",
  "darwin-x64": "sqlite-predict-darwin-x64",
  "linux-x64": "sqlite-predict-linux-x64",
  "linux-arm64": "sqlite-predict-linux-arm64",
  "win32-x64": "sqlite-predict-windows-x64",
};

/**
 * Absolute path to the loadable extension for this platform, without a file
 * suffix (the form SQLite's load_extension prefers).
 */
function getLoadablePath() {
  // dev / bundled fallback: a binary sitting next to this file
  for (const ext of ["dylib", "so", "dll"]) {
    const local = path.join(__dirname, `predict0.${ext}`);
    if (fs.existsSync(local)) return path.join(__dirname, "predict0");
  }
  const key = `${process.platform}-${process.arch}`;
  const pkg = PLATFORM_PACKAGES[key];
  if (!pkg) {
    throw new Error(
      `sqlite-predict: no prebuilt binary for ${key}; build from source`
    );
  }
  try {
    return require(pkg).loadablePath;
  } catch (_) {
    throw new Error(
      `sqlite-predict: the platform package ${pkg} is not installed`
    );
  }
}

/**
 * Load sqlite-predict into a connection with a loadExtension method
 * (better-sqlite3, or node:sqlite's DatabaseSync with allowExtension: true).
 */
function load(db) {
  db.loadExtension(getLoadablePath());
}

module.exports = { getLoadablePath, load };
