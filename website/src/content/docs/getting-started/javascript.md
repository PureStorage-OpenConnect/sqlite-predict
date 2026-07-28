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

const rows = db
  .prepare("select step, forecast from forecast('select ts, value from readings', 6)")
  .all();
console.log(rows);
```

With `node:sqlite` you need `--experimental-sqlite` on Node 22, or a recent Node
where it is stable. `sqlitePredict.getLoadablePath()` returns the binary path if
you load it yourself.

Next: [Operations](/guides/operations/).
