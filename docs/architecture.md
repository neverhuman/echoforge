# EchoForge Architecture

## System Overview

EchoForge is a strict-open, radar-first synthetic sensing foundry. It produces public-proxy object
signatures, uncertainty-scored radar artifacts, and reproducible validation evidence. The system is
organized into three runtime tiers: simulation core (Rust), service boundary (Rust studio), and
presentation (TypeScript/React web app).

## Layer Map

```
┌─────────────────────────────────────────────────────────────────┐
│ apps/web/              React/Vite web application               │
│   src/MeshViewer.tsx   List-state machine (no binary data)      │
│   src/MeshCanvas.tsx   THREE.js rendering layer (binary OK)     │
│   src/meshService.ts   HTTP data-access boundary                │
├─────────────────────────────────────────────────────────────────┤
│ crates/echoforge-studio/  Rust studio HTTP service              │
│   /api/mesh/*             Binary STL endpoints                  │
│   /api/health             Service health gate                   │
├─────────────────────────────────────────────────────────────────┤
│ crates/echoforge-world/   Mesh geometry engine (Rust)           │
│ crates/echoforge-radar/   Radar simulation (Rust)               │
│ crates/echoforge-sig/     Signal and validation layer (Rust)    │
│ crates/echoforge-core/    Shared types, errors, contracts       │
│ crates/echoforge-dataset/ ML dataset generation (Rust)          │
├─────────────────────────────────────────────────────────────────┤
│ python/ai-service/     Advanced ML inference only               │
│ detection/             Science experiment scripts (ONNX output) │
├─────────────────────────────────────────────────────────────────┤
│ schemas/               JSON Schema definitions (source of truth)│
│ contracts/             Generated schema catalog + typed clients │
│ agent/                 Governance and worker coordination        │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Boundaries

### Web Layer (`apps/web/`)
- **Allowed**: React components, state management, UI logic, THREE.js rendering
- **Forbidden**: Direct SQL access, raw filesystem access, unauthorized fetch outside meshService
- **Data access**: All HTTP calls go through `meshService.ts` which provides typed domain objects
- **Binary data**: Confined to `MeshCanvas.tsx` (rendering layer); `MeshViewer.tsx` has no ArrayBuffer

### Rust Service Layer (`crates/echoforge-studio/`)
- **Allowed**: HTTP routing, binary serialization, mesh synthesis calls
- **Forbidden**: Direct ML inference, Python FFI in hot path
- **Data access**: Calls into `echoforge-world` for mesh synthesis; no external database

### Rust Core (`crates/echoforge-*`)
- **Allowed**: Physics simulation, signal processing, statistical modeling
- **Forbidden**: Network I/O in library crates, direct file I/O outside codec layer

### Python (`python/ai-service/`, `detection/`)
- **Allowed**: Advanced ML training/inference that has no viable Rust equivalent
- **Forbidden**: Owning schemas, defining database tables, serving product APIs
- **Boundary**: All output consumed by Rust via ONNX or JSON contracts

## Contract Flow

```
schemas/*.json  →  (contracts lane)  →  contracts/schema_catalog.json
                                           (generated, DO NOT EDIT BY HAND)
                                       apps/web/src/generated/
                                           (generated TypeScript clients)
```

The `contracts` lane (`bash ops/run-lane.sh contracts`) validates and regenerates both artifacts.
Any schema change must be followed by a `contracts` lane run before committing.

## Error Model

All Rust errors flow through `CoreError` in `crates/echoforge-core/src/error.rs`.
Each variant exposes three agent-readable fields:
- `purpose()` — what the operation was trying to do
- `repair_hint()` — concrete next step to diagnose
- `docs_url()` — documentation pointer (links into `docs/`)

Web errors (from meshService) are surfaced as plain string messages in the UI and logged via
`console.error('[MeshViewer] ...', msg)` for telemetry.

## Proof Lanes

| Lane | Command | What it checks |
|---|---|---|
| `fast` | `bash ops/run-lane.sh fast` | Rust tests, lints, governance checks |
| `contracts` | `bash ops/run-lane.sh contracts` | Schema validation + catalog drift |
| `security` | `bash ops/run-lane.sh security` | Dependency audit, SBOM, secret scan |
| `web-smoke` | `bash ops/run-lane.sh web-smoke` | TypeScript build + vitest unit tests |
| `web-e2e` | `bash ops/run-lane.sh web-e2e` | Playwright end-to-end tests |
| `ux-qa` | `bash ops/run-lane.sh ux-qa` | Playwright + accessibility scans (HLT-013 evidence) |
| `vendor-scrub` | `bash ops/run-lane.sh vendor-scrub` | Banned-term enforcement |
| `receipts` | `bash ops/run-lane.sh receipts` | Agent receipt validation |

Full lane definitions: `agent/proof-lanes.toml`  
Full proof routing: `agent/test-map.json`  
Ownership map: `agent/owner-map.json`

## Generated Zones

Files listed in `agent/generated-zones.toml` must NOT be hand-edited:
- `target/` — Rust build artifacts
- `contracts/schema_catalog.json` — regenerated by `contracts` lane
- `apps/web/src/generated/` — TypeScript clients from schemas
- `node_modules/`, `dist/`, `build/`, `.venv/` — dependency/build outputs
- `sbom/` — generated by the security lane

## Observability

- **Studio health**: `GET /api/health` → `{"status":"ok"}`
- **Structured errors**: All `CoreError` variants log `{purpose, repair_hint, docs_url}`
- **MeshViewer telemetry**: `console.error('[MeshViewer] ...', msg)` on all error paths
- **Audit score**: `jankurai audit` → `agent/repo-score.json`, `agent/repo-score.md`
- **Receipt audit**: `tools/receipt_guard.mjs --all` validates all `.agents/receipts/` files

## Security Posture

- Dependency audit: `cargo deny check` (blocks on any policy violation)
- Secret scan: `gitleaks detect` (blocking in CI)
- SBOM: `cargo-cyclonedx` (CycloneDX JSON format, committed to `sbom/`)
- All GitHub Actions pinned to full commit SHAs
- Banned terms enforced via `agent/banned-terms.toml` + `tools/vendor_scrub.mjs`

## Routing from AGENTS.md

See root `AGENTS.md` for:
- Command prefixes and toolchain requirements
- Ownership model and worker coordination
- Compliance gates (vendor-scrub, receipts, security lane)
- Stack discipline (what Python is allowed for)
