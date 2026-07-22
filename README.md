# sqlite-predict

A work-in-progress SQLite extension for zero-shot prediction —
`forecast()`, `detect_anomalies()`, and friends as table-valued functions,
with every result bound to a replayable receipt.

> [!IMPORTANT]
> Pre-alpha and unreleased. Nothing here is stable. The spec lives in
> RFC 0005 (private, for now).

## Build and test

```sh
make loadable
cd tests && uv run pytest
```

Requires a C99 compiler; `make` fetches the SQLite amalgamation headers
into `vendor/` on first build.
