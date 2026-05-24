# Security policy

## Status

EchoForge is pre-1.0 research code. There are no security-supported releases
yet. The expectations below apply to `main` only.

## Scope

EchoForge is a strict-open, public-proxy synthetic-sensing foundry. Hard
negatives (birds, drones, weather, RFI, multipath) are included for
robustness and false-alarm research, **not** evasion engineering. The
project does not produce, depend on, or facilitate offensive targeting,
detection-evasion, or operational engagement tooling.

If a contribution or proposed feature appears to cross that line, raise
it as a security concern using the channel below.

## Reporting a vulnerability

Email the maintainer at `jepson@veox.ai` with subject `EchoForge security:
<short summary>`. Please include:

- Affected version / commit SHA.
- Reproducer or proof-of-concept (minimal, please).
- The impact you observed and the impact you expect.
- Whether the issue is already public.

We aim to acknowledge within 5 business days and to coordinate
disclosure under a 90-day window from acknowledgement, extended if a
fix or mitigation requires longer.

Do not open public GitHub issues for vulnerabilities. Once a fix lands,
a credited advisory will be published in `SECURITY-ADVISORIES.md`.

## Supply-chain posture

- Licenses are gated by `deny.toml`; the `licenses` lane in CI enforces
  the strict-open allow-list (Apache-2.0/MIT/BSD/ISC/Unicode/Zlib/MPL-2.0/0BSD/CC0-1.0).
- SBOMs (CycloneDX) for Cargo + npm + Python are emitted by
  `tools/sbom_emit.sh` and rolled up into `sbom/index.md`.
- Third-party Rust dependencies are pinned via `Cargo.lock`.
- Python optional extras are pinned via `pyproject.toml`.
- Vendor-name banlist (`agent/banned-terms.toml`) is enforced by the
  `vendor-scrub` CI job, blocking from day one.
