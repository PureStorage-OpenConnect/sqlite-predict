---
title: JavaScript / Node
description: Use sqlite-predict from Node with node:sqlite or better-sqlite3.
---

Install from npm. A prebuilt binary for your platform is pulled in as an
optional dependency, so there's no build step:

```sh
npm install sqlite-predict
```

It loads into either the built-in `node:sqlite` or better-sqlite3, anything with
a `loadExtension` method:

```js
import * as sqlitePredict from "sqlite-predict";
import { DatabaseSync } from "node:sqlite";

const db = new DatabaseSync(":memory:", { allowExtension: true });
sqlitePredict.load(db);

// did it load?
console.log(db.prepare("select predict_version()").get());

db.exec("create table readings(ts text, value real)");
const ins = db.prepare("insert into readings values (?, ?)");
for (let h = 0; h < 24; h++) {
  ins.run(`2024-01-01T${String(h).padStart(2, "0")}:00:00`, 50 + h);
}

// forecast() is an aggregate like sum(): your statement supplies the
// rows, and each group returns one JSON document
const { doc } = db
  .prepare("select forecast(ts, value, 6) as doc from readings")
  .get();
console.log(JSON.parse(doc).rows[0]);
// { step, forecast_timestamp, forecast, lower_bound, upper_bound }
```

Because it is an aggregate, `WHERE`, joins, and bound parameters compose, and
`GROUP BY city` returns one document per city. This is the form Drizzle and
other query builders build naturally.

With `node:sqlite` you need `--experimental-sqlite` on Node 22, or a recent Node
where it is stable. `sqlitePredict.getLoadablePath()` returns the binary path if
you load it yourself.

Next: [Operations](../../guides/operations/), or
[Using with ORMs](../../guides/orms/) for the Drizzle pattern.
