# sqlite-predict

Prediction as a SQL primitive for SQLite: `forecast()`, `detect_anomalies()`,
`predict()`, with replayable receipts. Bundles the zero-dependency loadable
extension for your platform.

```js
const Database = require("better-sqlite3");
const sqlitePredict = require("sqlite-predict");

const db = new Database(":memory:");
sqlitePredict.load(db);

for (const row of db
  .prepare("SELECT * FROM forecast('SELECT ts, value FROM readings', 24)")
  .all()) {
  console.log(row);
}
```

Also works with Node's built-in `node:sqlite`:

```js
const { DatabaseSync } = require("node:sqlite");
const sqlitePredict = require("sqlite-predict");
const db = new DatabaseSync(":memory:", { allowExtension: true });
db.loadExtension(sqlitePredict.getLoadablePath());
```

See the [project README](https://github.com/PureStorage-OpenConnect/sqlite-predict)
for the full SQL surface, models, and benchmarks.
