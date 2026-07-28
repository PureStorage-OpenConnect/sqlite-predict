---
title: CLI / C
description: Load sqlite-predict in the SQLite CLI, or vendor the single C file.
---

## SQLite CLI

Download a prebuilt loadable for your platform from the
[GitHub releases](https://github.com/PureStorage-OpenConnect/sqlite-predict/releases)
and `.load` it:

```sh
sqlite3 mydata.db
```

```sql
.load ./predict0

SELECT predict_version();

-- forecast() is an aggregate like sum(): your query supplies the rows,
-- and each group returns one JSON document
SELECT forecast(ts, value, 24) FROM readings;
```

## Vendor the single C file

Every release also ships a single-file amalgamation (`sqlite-predict.c`) you can
drop into your own program, or compile straight to a loadable:

```sh
curl -LO https://github.com/PureStorage-OpenConnect/sqlite-predict/releases/latest/download/sqlite-predict.c
cc -fPIC -shared -O3 sqlite-predict.c -o predict0.so   # .dylib on macOS, .dll on Windows
```

To statically link into your app, compile with `-DSQLITE_CORE
-DSQLITE_PREDICT_STATIC` and call `sqlite3_predict_init(db, 0, 0)` after opening
each connection.

Next: [Operations](../../guides/operations/).
