# EchoForge

EchoForge is a strict-open, radar-first, GPU-native synthetic sensing foundry.

It is intended to produce public-proxy object signatures, uncertainty-scored
RCS artifacts, radar-chain products, hard-negative worlds, validation reports,
and dataset/benchmark packages with reproducible provenance.

## Current scaffold

- Governance and agent instructions live in [`AGENTS.md`](./AGENTS.md).
- Ownership, test, and generated-zone maps live in [`agent/`](./agent).
- Bootstrap commands live in [`Justfile`](./Justfile).
- Package manifests live in [`Cargo.toml`](./Cargo.toml),
  [`pyproject.toml`](./pyproject.toml), and [`package.json`](./package.json).

## Repo posture

- Strict-open core, with Python used only where the science stack needs it.
- No exact measured-truth claims for public-proxy signatures without lawful
  measured data and explicit provenance.
- Large solver outputs, array stores, and benchmark artifacts stay out of Git.

## First checks

```bash
rtk just fast
rtk jankurai adapters verify .
rtk just demo
```
