/** Absolute path to the bundled loadable extension for this platform. */
export function getLoadablePath(): string;

/**
 * Load sqlite-predict into a connection exposing `loadExtension`
 * (better-sqlite3, or node:sqlite's `DatabaseSync` with `allowExtension: true`).
 */
export function load(db: { loadExtension(path: string): void }): void;
