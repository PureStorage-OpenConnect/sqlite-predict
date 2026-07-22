# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately. **Do not open a public
issue for security reports.**

- Use GitHub's [private vulnerability reporting][ghsa] on this repository
  (the Security tab, then "Report a vulnerability"), or
- Email the Pure Storage security team at **psirt@purestorage.com**.

Please include a description, reproduction steps or a proof of concept, and
the affected version or commit. We will acknowledge receipt and keep you
updated on remediation. Please give us a reasonable window to release a fix
before public disclosure.

## Trust model

`sqlite-predict` executes inside the host database process with that
process's privileges. Three inputs cross a trust boundary and receive
particular attention:

- **Model blobs.** Deserializing model weights is code-adjacent (ONNX and
  GGUF parsers have had memory-safety CVEs). The extension verifies a
  content hash before loading any model, and the model registry table is
  writable by any SQL caller, so catalog and user-registered models are
  treated identically.
- **SQL arguments and inner queries.** Callers may be agents executing
  partially untrusted plans. Inner queries are enforced read-only and
  single-statement; stored receipt SQL is re-executed read-only on replay.
- **Resource exhaustion.** Inference runs in-process; input row counts,
  forecast horizons, and per-call work are bounded, and interrupts are
  honored between batches.

Error messages do not echo input row values. Receipts are as sensitive as
the queries they record.

[ghsa]: https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability
