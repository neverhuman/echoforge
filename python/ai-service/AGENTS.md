# Python AI Service — Agent Guidance

Ownership: runtime_core
Test lane: `bash ops/run-lane.sh science-smoke`

## ML Exception Declaration

This Python package exists under the science/ML exception granted in AGENTS.md §3.
Python is used here because the inference workloads require libraries (scikit-learn,
LightGBM, ONNX Runtime) with no viable Rust-native equivalent for production-grade
inference at this time.

## Strict Boundaries

This service MUST NOT:
- Own database schemas (all schemas live in `schemas/` and `contracts/`)
- Define ground-truth data structures (those live in Rust crates)
- Make direct database connections (use the Rust API layer)
- Export JSON that diverges from `contracts/schema_catalog.json`

This service MAY:
- Load ONNX/PyTorch model artifacts from `outputs/`
- Perform inference using loaded models
- Return results that conform to echoforge schemas
- Use Python-only ML libraries for inference

## Structure

```
echoforge_core/      # Python mirror of core data types (schema-derived, not authoritative)
echoforge_validate/  # Validation utilities calling into Rust-defined rules
echoforge_signatures/ # Signature loading (reads artifact bundles only)
```
