// End-to-end: register the extension, then forecast through a rusqlite connection.
#[test]
fn registers_and_forecasts() {
    unsafe { sqlite_predict::register().expect("register") };
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE r(ts TEXT, value REAL);").unwrap();
    for i in 0..96i64 {
        conn.execute(
            "INSERT INTO r VALUES (?1, ?2)",
            rusqlite::params![
                format!("2024-01-01T{:02}:00:00", i % 24),
                50.0 + 10.0 * (i as f64).sin()
            ],
        )
        .unwrap();
    }
    let n: i64 = conn
        .query_row(
            "SELECT count(*) FROM forecast('SELECT ts, value FROM r ORDER BY ts', 6)",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(n, 6, "forecast should return 6 rows");
}
