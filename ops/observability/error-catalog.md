# EchoForge Error Catalog

Structured error signals emitted by EchoForge crates. Each error produces a machine-readable JSON line on stderr.

## Canonical Error Format

```json
{"error": "<Kind>", "field": "<path.or.key>", "reason": "<human-readable>", "rerun": "<just <lane>>"}
```

## Error Kinds

| Kind | Crate | Rerun Command | Meaning |
|------|-------|--------------|---------|
| `ValidationFailed` | `echoforge-validate` | `just fast` | Input failed schema or domain validation |
| `SchemaViolation` | `echoforge-core` | `just contracts` | Artifact does not conform to JSON schema |
| `ArtifactNotFound` | `echoforge-core` | `just doctor` | Artifact ID missing from store |
| `SerializationFailed` | `echoforge-core` | `just fast` | JSON encode/decode error; see `detail` field |
| `InvalidConfiguration` | `echoforge-studio` | `just fast` | Config key invalid or missing |
| `Io` | any | `just fast` | Filesystem I/O error; check path and permissions |
| `SceneSetupFailed` | `echoforge-radar` | `just science-smoke` | Scene parameters invalid or missing dependency |
| `DetectionThresholdUnmet` | `echoforge-radar` | `just science-smoke` | CFAR or detector threshold not satisfied |
| `TierValidationFailed` | `echoforge-validate` | `just science-validate` | Tier-measured anchored validation failed |

## Telemetry Spans

All production entry points emit `tracing` spans for latency measurement:

| Span | Crate | Description |
|------|-------|-------------|
| `echoforge_core::artifact::load` | `echoforge-core` | Artifact load from store |
| `echoforge_core::artifact::store` | `echoforge-core` | Artifact write to store |
| `echoforge_radar::sim::run` | `echoforge-radar` | Full scene simulation run |
| `echoforge_dataset::pipeline` | `echoforge-dataset` | Dataset generation pipeline |
| `echoforge_validate::validate` | `echoforge-validate` | Tier validation check |

## Repair Evidence Location

| Evidence Type | Path | Format |
|--------------|------|--------|
| Open findings | `target/jankurai/repair-queue.jsonl` | JSONL, one finding per line |
| Score history | `target/jankurai/score-history.jsonl` | JSONL, one run per line |
| Repair receipts | `.agents/receipts/<slice>/<UTC>.md` | Markdown receipt |
| Security evidence | `target/jankurai/security/evidence.json` | JSON |

## Agent Rerun Instructions

Each finding's `Rerun:` field contains the exact lane command needed to reproduce and verify the fix:

```
just fast          # unit tests + adapter verification
just contracts     # schema contract smoke tests
just security      # dependency audit + secret scan
just science-smoke # science/radar tests
just score         # governance audit (advisory mode)
```

All lanes are idempotent and rerunnable without side effects.
