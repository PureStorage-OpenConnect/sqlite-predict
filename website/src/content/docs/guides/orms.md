---
title: Using with ORMs
description: Compose forecasts through Drizzle, SQLAlchemy, or Diesel, and keep migration tools away from the model registry.
---

`forecast` and `detect_anomalies` are aggregate functions: the statement
supplies the rows, so filtering, joins, parameters, and `GROUP BY`
series-splitting are all ordinary query-builder territory.

```sql
SELECT city, forecast(ts, value, 24) FROM readings GROUP BY city;
```

Each group returns one JSON document; parse it in your app (one line) or expand
it back to typed rows in SQL with
[`forecast_rows()`](../../reference/functions/#forecast_rowsdoc--anomaly_rowsdoc).
The snippets below are the same code the CI smoke tests run.

## Drizzle (Node)

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

## SQLAlchemy (Python)

```python
import json, sqlalchemy as sa
import sqlite_predict

eng = sa.create_engine("sqlite:///app.db")

@sa.event.listens_for(eng, "connect")
def load_ext(conn, _):
    conn.enable_load_extension(True)
    sqlite_predict.load(conn)
    conn.enable_load_extension(False)

readings = sa.table("readings", sa.column("city"),
                    sa.column("ts"), sa.column("value"))

stmt = (
    sa.select(
        readings.c.city,
        sa.func.forecast(readings.c.ts, readings.c.value, 24).label("doc"))
    .where(readings.c.city == sa.bindparam("city"))
    .group_by(readings.c.city))

with eng.connect() as conn:
    for city, doc in conn.execute(stmt, {"city": "SF"}):
        forecast = json.loads(doc)
```

## Diesel / rusqlite (Rust)

The crate registers via `sqlite3_auto_extension`, so every connection the ORM
opens afterward has the functions with no per-connection hook:

```rust
unsafe { sqlite_predict::register()?; }   // once, before connecting

// then through any rusqlite-based stack:
let doc: String = conn.query_row(
    "SELECT forecast(ts, value, 24) FROM readings WHERE city = ?1",
    ["SF"], |r| r.get(0))?;
```

## Prisma

Prisma's default engine never exposes the raw connection, so extensions can't
load. Use a [driver adapter](https://www.prisma.io/docs/orm/overview/databases/database-drivers)
(better-sqlite3 or libSQL) and load on the underlying connection as in the
Drizzle example.

## Keep migration tools away from `_predict_*`

sqlite-predict stores its model registry (bundled models and distilled
students) in `_predict_models`, inside your database. Schema-diffing tools see
a table they don't own and will happily generate `DROP TABLE` for it. Exclude
the prefix:

**drizzle-kit** (`drizzle.config.ts`):

```ts
export default { /* … */ tablesFilter: ["!_predict_*"] };
```

**Alembic** (`env.py`):

```python
def include_name(name, type_, parent_names):
    return not (type_ == "table" and name.startswith("_predict_"))

context.configure(include_name=include_name, ...)
```

**Prisma**: `_predict_*` tables are not in your schema, so `prisma migrate`
drift detection flags them; keep predict tables in a separate attached
database, or use `migrate diff` reviews rather than auto-apply.

Reads stay reads: the aggregate writes nothing, so replicas and read-only
databases just work.
