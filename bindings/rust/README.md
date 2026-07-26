# sqlite-predict

Prediction as a SQL primitive for SQLite: `forecast()`, `detect_anomalies()`,
`predict()`, with replayable receipts. The zero-dependency C core is compiled
into the crate; pair it with `rusqlite`.

```rust
unsafe { sqlite_predict::register().unwrap() };
let conn = rusqlite::Connection::open_in_memory()?;
let mut stmt = conn.prepare("SELECT * FROM forecast('SELECT ts, value FROM readings', 24)")?;
```

See the [project README](https://github.com/PureStorage-OpenConnect/sqlite-predict).
