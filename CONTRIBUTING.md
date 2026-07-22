# Contributing to sqlite-predict

Thanks for your interest. This project is pre-alpha; the API and formats
are unstable, so it's a good time to influence them.

## Ground rules

- **Correctness is verified, not asserted.** Every change must keep the
  test suite green and the sanitizers clean. New behavior needs tests,
  including adversarial ones that try to break it. The suite has caught
  real memory bugs this way.
- **Fail loudly.** No silent fallbacks. If an input is invalid, raise a
  `PREDICT_ERR_*` error; don't guess or degrade silently.
- **The receipt is a contract.** If you change what an operation returns,
  the result-hash layout changes and `predict_replay` round-trips will
  fail. That is the intended tripwire; update it deliberately.

## Developer Certificate of Origin (DCO)

Contributions are accepted under the [DCO](https://developercertificate.org/).
Sign off each commit to certify you wrote it or have the right to submit it:

```sh
git commit -s -m "your message"
```

This appends a `Signed-off-by: Your Name <you@example.com>` trailer.

## Workflow

1. Fork and branch from `main`.
2. Make your change with tests.
3. Run the checks locally:
   ```sh
   make test          # pytest suite
   make test-asan     # AddressSanitizer + UBSan
   make test-valgrind # valgrind (needs Docker)
   ```
4. Open a pull request. CI runs the same checks on Linux and macOS.

## Code style

- C99, no compiler warnings (`-Wall -Wextra`), formatted with
  `clang-format` (`make format`).
- Follow the existing conventions: `predict_*` for SQL-visible functions,
  `predict0_*` for internals; contract constants live in
  `predict-internal.h`.
- Comments explain *why*, not *what*.

By contributing, you agree that your contributions are dual-licensed under
MIT OR Apache-2.0, matching the project license.
