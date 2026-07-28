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
`GROUP BY city` returns one document per city.

## Drizzle

The same call through the query builder, with a bound parameter and no SQL
strings (this is the code the CI smoke test runs):

```ts
import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import { sql, eq } from "drizzle-orm";
import { sqliteTable, text, real } from "drizzle-orm/sqlite-core";
import * as sp from "sqlite-predict";

const sqlite = new Database("app.db");
sqlite.loadExtension(sp.getLoadablePath());
const db = drizzle(sqlite);

const readings = sqliteTable("readings", {
  city: text("city"), ts: text("ts"), value: real("value"),
});

const out = db
  .select({
    city: readings.city,
    doc: sql`forecast(${readings.ts}, ${readings.value}, 24)`,
  })
  .from(readings)
  .where(eq(readings.city, "SF"))   // bound parameter, no string SQL
  .groupBy(readings.city)
  .all();

const { rows } = JSON.parse(out[0].doc);
```

If you use drizzle-kit, exclude sqlite-predict's model registry from schema
diffs (it lives in your database, and diff tools will otherwise try to drop
it):

```ts
// drizzle.config.ts
export default { /* … */ tablesFilter: ["!_predict_*"] };
```

**Prisma**: the default engine never exposes the raw connection, so extensions
can't load. Use a [driver adapter](https://www.prisma.io/docs/orm/overview/databases/database-drivers)
(better-sqlite3 or libSQL) and load on the underlying connection as above; and
since `_predict_*` tables are not in your Prisma schema, prefer `migrate diff`
reviews over auto-apply so drift detection doesn't drop them.

With `node:sqlite` you need `--experimental-sqlite` on Node 22, or a recent Node
where it is stable. `sqlitePredict.getLoadablePath()` returns the binary path if
you load it yourself.

Next: [Operations](../../guides/operations/).
