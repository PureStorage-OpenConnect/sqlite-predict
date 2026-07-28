# sqlite-predict

Prediction as a SQL primitive for SQLite: `forecast()`, `detect_anomalies()`,
`predict()`. The zero-dependency C core is compiled into the crate; pair it
with `rusqlite`.

```rust
unsafe { sqlite_predict::register().unwrap() };
let conn = rusqlite::Connection::open_in_memory()?;
// forecast() is an aggregate: your query supplies the rows, and each
// group returns one JSON document
let doc: String = conn.query_row(
    "SELECT forecast(ts, value, 24) FROM readings", [], |r| r.get(0))?;
```

See the [project README](https://github.com/PureStorage-OpenConnect/sqlite-predict).
