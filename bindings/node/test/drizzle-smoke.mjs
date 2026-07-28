// Drizzle ORM smoke test: compose the aggregate form of forecast()
// through the query builder (the RFC 0005 §4.2.8 ORM path). Run by CI's
// bindings job; the same pattern is documented in the JavaScript
// getting-started guide.
import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import { sql, eq } from "drizzle-orm";
import { sqliteTable, text, real } from "drizzle-orm/sqlite-core";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const sp = require("../index.cjs");

const sqlite = new Database(":memory:");
sqlite.loadExtension(sp.getLoadablePath());
const db = drizzle(sqlite);

const readings = sqliteTable("readings", {
  city: text("city"),
  ts: text("ts"),
  value: real("value"),
});

db.run(sql`CREATE TABLE readings (city TEXT, ts TEXT, value REAL)`);
const rows = [];
for (const city of ["SF", "LA"]) {
  for (let i = 0; i < 96; i++) {
    const day = String(1 + ((i / 24) | 0)).padStart(2, "0");
    const hour = String(i % 24).padStart(2, "0");
    rows.push({
      city,
      ts: `2024-01-${day}T${hour}:00:00Z`,
      value: 50 + 10 * Math.sin(i / 4),
    });
  }
}
db.insert(readings).values(rows).run();

// the aggregate form: WHERE, GROUP BY, and parameter binding are all
// ordinary query-builder territory; no SQL-in-a-string anywhere
const out = db
  .select({
    city: readings.city,
    doc: sql`forecast(${readings.ts}, ${readings.value}, 4)`,
  })
  .from(readings)
  .where(eq(readings.city, "SF"))
  .groupBy(readings.city)
  .all();

if (out.length !== 1) throw new Error(`expected one group, got ${out.length}`);
const doc = JSON.parse(out[0].doc);
if (doc.status !== "ok" || doc.rows.length !== 4)
  throw new Error(`unexpected document: ${out[0].doc}`);
if (typeof doc.rows[0].forecast !== "number")
  throw new Error("forecast is not numeric");
console.log("drizzle smoke OK:", doc.rows[0]);
