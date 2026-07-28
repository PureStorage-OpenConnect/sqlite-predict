# sqlite-predict

Prediction as a SQL primitive for SQLite: `forecast()`, `detect_anomalies()`,
`predict()`. Bundles the zero-dependency loadable extension for your platform.

```js
const Database = require("better-sqlite3");
const sqlitePredict = require("sqlite-predict");

const db = new Database(":memory:");
sqlitePredict.load(db);

// forecast() is an aggregate: your query supplies the rows, and each
// group returns one JSON document
const { doc } = db
  .prepare("SELECT forecast(ts, value, 24) AS doc FROM readings")
  .get();
console.log(JSON.parse(doc));
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
