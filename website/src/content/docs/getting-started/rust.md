---
title: Rust
description: Use sqlite-predict from Rust with rusqlite.
---

The crate compiles the bundled amalgamation itself (via `build.rs` and the `cc`
crate), so there are no system dependencies:

```sh
cargo add sqlite-predict
cargo add rusqlite --features bundled
```

Register it as an auto-extension once at startup; every connection you open
afterward has the functions:

```rust
fn main() -> rusqlite::Result<()> {
    // Registers sqlite-predict process-wide. Call once, before opening
    // connections. `register` is unsafe: it registers a C entry point globally.
    unsafe { sqlite_predict::register().expect("register sqlite-predict") };

    let db = rusqlite::Connection::open_in_memory()?;
    let version: String = db.query_row("select predict_version()", [], |r| r.get(0))?;
    println!("{version}");

    db.execute_batch(
        "create table readings(ts text, value real);
         insert into readings
           select printf('2024-01-01T%02d:00:00', n), 50 + n
           from (with recursive c(n) as (select 0 union all select n+1 from c where n < 23)
                 select n from c);",
    )?;

    let mut stmt =
        db.prepare("select step, forecast from forecast('select ts, value from readings', 6)")?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?)))?;
    for row in rows {
        println!("{:?}", row?);
    }
    Ok(())
}
```

Next: [Operations](/guides/operations/).
