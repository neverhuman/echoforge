# jankurai Repo Score

- Standard: `jankurai`
- Auditor: `0.8.16`
- Schema: `1.7.0`
- Paper edition: `2026.05-ed8`
- Target stack ID: `rust-ts-vite-react-postgres-bounded-python`
- Target stack: `Rust core + TypeScript/React/Vite + PostgreSQL + generated contracts + exception-only Python AI/data service`
- Repo: `.`
- Run ID: `1779163722`
- Started at: `1779163722`
- Elapsed: `631` ms
- Scope: `full`
- Raw score: `46`
- Final score: `46`
- Decision: `fail`
- Minimum score: `85`
- Caps applied: `no-one-command-setup-or-validation, no-deterministic-fast-lane, no-security-lane-on-high-risk-repo, no-secret-or-dependency-scanning-in-ci, no-jankurai-audit-lane-in-ci, direct-db-access-from-wrong-layer, missing-rendered-ux-qa-lane, release-readiness-gap, no-agent-friendly-exception-pattern, missing-agent-readable-docs`

## Hard Rule Caps

| Rule | Max Score | Applied |
| --- | ---: | --- |
| `no-root-agent-instructions` | 75 | no |
| `no-one-command-setup-or-validation` | 70 | yes |
| `no-deterministic-fast-lane` | 65 | yes |
| `no-security-lane-on-high-risk-repo` | 60 | yes |
| `generated-contracts-or-public-api-drift-untested` | 80 | no |
| `python-direct-product-truth-or-db-ownership` | 72 | no |
| `no-secret-or-dependency-scanning-in-ci` | 78 | yes |
| `no-jankurai-audit-lane-in-ci` | 82 | yes |
| `jankurai-required-tool-ci-evidence-gap` | 88 | no |
| `non-optimal-product-language-found` | 74 | no |
| `too-much-python-in-product-surface` | 72 | no |
| `boundary-reclassification-evidence-gap` | 72 | no |
| `vibe-placeholders-in-product-code` | 68 | no |
| `fallback-soup-in-product-code` | 70 | no |
| `future-hostile-dead-language-in-product-code` | 64 | no |
| `severe-duplication-in-product-code` | 70 | no |
| `generated-zone-mutation-risk` | 76 | no |
| `direct-db-access-from-wrong-layer` | 66 | yes |
| `missing-web-e2e-lane` | 82 | no |
| `missing-rendered-ux-qa-lane` | 84 | yes |
| `prompt-injection-risk` | 78 | no |
| `overbroad-agent-agency` | 65 | no |
| `secret-like-content-detected` | 60 | no |
| `false-green-test-risk` | 76 | no |
| `destructive-migration-risk` | 70 | no |
| `authz-or-data-isolation-gap` | 78 | no |
| `input-boundary-gap` | 78 | no |
| `agent-tool-supply-chain-gap` | 78 | no |
| `release-readiness-gap` | 80 | yes |
| `missing-rust-property-or-integration-tests` | 82 | no |
| `no-agent-friendly-exception-pattern` | 76 | yes |
| `missing-agent-readable-docs` | 80 | yes |
| `streaming-runtime-drift` | 78 | no |
| `rust-bad-behavior` | 72 | no |
| `sql-bad-behavior` | 72 | no |
| `typescript-bad-behavior` | 72 | no |
| `docker-bad-behavior` | 72 | no |
| `python-bad-behavior` | 72 | no |
| `ci-bad-behavior` | 70 | no |
| `git-bad-behavior` | 70 | no |
| `gittools-bad-behavior` | 70 | no |
| `release-bad-behavior` | 70 | no |
| `web-security-bad-behavior` | 68 | no |
| `repo-rot-bad-behavior` | 88 | no |
| `comment-hygiene-dangerous-residue` | 72 | no |

## Dimensions

| Dimension | Weight | Score | Weighted | Evidence |
| --- | ---: | ---: | ---: | --- |
| Ownership and navigation surface | 13 | 68 | 8.84 | root `AGENTS.md` present; owner map covers audited paths |
| Contract and boundary integrity | 13 | 70 | 9.10 | contract surface found; generated contract artifacts found |
| Proof lanes and test routing | 12 | 16 | 1.92 | web e2e lane present or no web surface; DOM geometry UX QA runtime found |
| Security and supply-chain posture | 12 | 14 | 1.68 | typescript bad-behavior advisory signals: 1 |
| Code shape and semantic surface | 12 | 100 | 12.00 | largest authored code file: src/App.tsx (220 LOC); most code files stay under 300 LOC |
| Data truth and workflow safety | 8 | 30 | 2.40 | strict DB boundary violation: src/MeshViewer.test.tsx |
| Observability and repair evidence | 8 | 31 | 2.48 | repair receipts or raw artifact language found |
| Context economy and agent instructions | 7 | 37 | 2.59 | root `AGENTS.md` present; root `AGENTS.md` stays short |
| Jankurai tool adoption and CI replacement | 7 | 10 | 0.70 | control-plane files present; applicable=13 |
| Python containment and polyglot hygiene | 4 | 100 | 4.00 | no Python files in scope |
| Build speed signals | 4 | 0 | 0.00 |  |

## Reference Profile Structure

- Applicable cells: `0` canonical=`0` noncanonical=`0` guidance missing=`0`

| Cell | Status | Canonical | Detected | Aliases | Guidance | Owner | Proof lane | Agent fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `web` | `not_applicable` | `apps/web/` | `-` | `frontend/, ui/, packages/web/, packages/ui/` | `not_required` | `apps/web` | `rendered UX / Playwright` | `no action` |
| `api` | `not_applicable` | `apps/api/` | `-` | `api/, server/, backend/` | `not_required` | `apps/api` | `edge handler / contract tests` | `no action` |
| `domain` | `not_applicable` | `crates/domain/` | `-` | `domain/, core/` | `not_required` | `crates/domain` | `unit / property tests` | `no action` |
| `application` | `not_applicable` | `crates/application/` | `-` | `application/, usecases/, use-cases/` | `not_required` | `crates/application` | `use-case / authz tests` | `no action` |
| `adapters` | `not_applicable` | `crates/adapters/` | `-` | `adapters/, infra/, integrations/` | `not_required` | `crates/adapters` | `adapter integration tests` | `no action` |
| `workers` | `not_applicable` | `crates/workers/` | `-` | `workers/, jobs/, scheduler/, queue/` | `not_required` | `crates/workers` | `workflow / replay tests` | `no action` |
| `contracts` | `not_applicable` | `contracts/` | `-` | `openapi/, protobuf/, json-schema/, generated/` | `not_required` | `contracts` | `generation / drift checks` | `no action` |
| `db` | `not_applicable` | `db/` | `-` | `migrations/, constraints/, sql/` | `not_required` | `db` | `migration / constraint tests` | `no action` |
| `python-ai` | `not_applicable` | `python/ai-service/` | `-` | `python/, ai-service/, evals/, embeddings/, model/` | `not_required` | `python/ai-service` | `eval / contract tests` | `no action` |
| `ops` | `not_applicable` | `ops/` | `-` | `.github/, .github/workflows/, ci/, release/, observability/, security/` | `not_required` | `ops` | `security lane / workflow lint` | `no action` |

## Rendered UX QA

- Web surface: `true`
- Layered UX lane: `false`
- Missing: `generated API mocks`

## Tool Adoption

- Control plane present: `true`
- Applicable tools: `13`
- Configured: `0`
- CI evidence: `0`
- Artifact verified: `0`
- Replaced count: `0`
- Missing CI evidence: `audit-ci, proof-routing, proofbind, security, ci-bad-behavior, git-bad-behavior, release-bad-behavior, ux-qa, contract-drift, authz-matrix, agent-tool-supply, release-readiness, cost-budget`

| Tool | Category | Mode | Status | Replaced | Artifacts |
| --- | --- | --- | --- | --- | --- |
| `audit-ci` | `audit` | `auto` | `missing` | `manual repo scoring, ad hoc score gates` | `agent/repo-score.json, agent/repo-score.md` |
| `proof-routing` | `proof` | `auto` | `missing` | `ad hoc proof lane selection, manual proof receipts` | `agent/repo-score.json, agent/repo-score.md, target/jankurai/repair-queue.jsonl` |
| `proofbind` | `proof` | `auto` | `missing` | `manual changed-surface routing, ad hoc proof obligation lists` | `target/jankurai/proofbind/surface-witness.json, target/jankurai/proofbind/obligations.json` |
| `proofmark-rust` | `proof` | `auto` | `not_applicable` | `line-only coverage review, manual in-diff mutation review` | `target/jankurai/proofmark/proofmark-receipt.json, target/jankurai/proofmark/proof-receipt.json` |
| `security` | `security` | `auto` | `missing` | `gitleaks, dependency review, SBOM/provenance` | `target/jankurai/security/evidence.json` |
| `ci-bad-behavior` | `security` | `auto` | `missing` | `mutable workflow refs, secret echo/debug workflow checks, non-blocking security scans` | `target/jankurai/language-bad-behavior.log` |
| `git-bad-behavior` | `audit` | `auto` | `missing` | `destructive git automation, force-push release scripts, hidden stash-based state` | `target/jankurai/language-bad-behavior.log` |
| `release-bad-behavior` | `release` | `auto` | `missing` | `manual release checklist, ad hoc tag and artifact review, manual provenance review` | `target/jankurai/language-bad-behavior.log` |
| `ux-qa` | `ux` | `auto` | `missing` | `playwright, axe-core, visual baselines` | `target/jankurai/ux-qa.json` |
| `db-migration-analyze` | `db` | `auto` | `not_applicable` | `manual migration review` | `target/jankurai/migration-report.json` |
| `contract-drift` | `contract` | `auto` | `missing` | `handwritten contract drift checks, openapi diff` | `agent/repo-score.json, agent/repo-score.md` |
| `rust-witness` | `rust` | `auto` | `not_applicable` | `manual witness graphing` | `target/jankurai/rust/witness-graph.json` |
| `vibe-coverage` | `audit` | `auto` | `not_applicable` | `manual vibe-coding coverage spreadsheet` | `target/jankurai/vibe-coverage.json, target/jankurai/vibe-coverage.md` |
| `coverage-evidence` | `proof` | `auto` | `not_applicable` | `manual coverage report review, ad hoc mutation survivor review` | `target/jankurai/coverage/coverage-audit.json, target/jankurai/coverage/coverage-audit.md` |
| `authz-matrix` | `security` | `auto` | `missing` | `manual authz matrix review` | `agent/repo-score.json, agent/repo-score.md` |
| `input-boundary` | `security` | `auto` | `not_applicable` | `manual unsafe sink review` | `agent/repo-score.json, agent/repo-score.md` |
| `agent-tool-supply` | `security` | `auto` | `missing` | `manual MCP/tool trust review` | `agent/repo-score.json, agent/repo-score.md` |
| `release-readiness` | `release` | `auto` | `missing` | `manual launch checklist` | `agent/repo-score.json, agent/repo-score.md` |
| `cost-budget` | `release` | `auto` | `missing` | `manual spend review` | `agent/repo-score.json, agent/repo-score.md` |

## Boundary Reclassifications

No audited runtime boundary reclassifications declared.

## Findings

1. `high` `proof` `.`
   Rule: `HLT-004-UNMAPPED-PROOF`
   Check: `HLT-004-UNMAPPED-PROOF:proof` `hard` confidence `0.88`
   Route: TLR `Verification`, lane `fast`, owner `tools`
   Docs: `agent/JANKURAI_STANDARD.md#proof-lanes`
   Reason: no deterministic fast lane was detected
   Fix: add a fast lane that runs the narrowest deterministic proof loop and keep it canonical
   Rerun: `just fast`
   Fingerprint: `sha256:8f5faf682afa829927a322d200bd31f22afd36836db985a72e97a44de7487584`
   Evidence: no fast lane markers found
2. `high` `proof` `.`
   Check: `HLT-000-SCORE-DIMENSION:proof` `hard` confidence `0.88`
   Route: TLR `Verification`, lane `fast`, owner `unmapped`
   Reason: no one-command setup or validation lane was detected
   Fix: add a canonical `setup`, `check`, `test`, or `verify` lane in one root command file
   Rerun: `just fast`
   Fingerprint: `sha256:7010147691f443ae19d3d8603c11ec84958d455885b09e09eca0b9fa91933bde`
   Evidence: no root setup/check/test/verify target surfaced
3. `high` `audit` `.github/workflows`
   Check: `HLT-000-SCORE-DIMENSION:audit` `hard` confidence `0.88`
   Route: TLR `Context/setup`, lane `audit`, owner `ops`
   Reason: CI does not run the jankurai audit lane
   Fix: add a CI job that runs `jankurai . --json agent/repo-score.json --md agent/repo-score.md` and uploads both artifacts
   Rerun: `just score`
   Fingerprint: `sha256:dcdd57510181477041c184ca8e3087ad93965fd92fe3357b1be5d73986e805ae`
   Evidence: audit output must stay JSON plus Markdown for agent repair routing
4. `high` `security` `.github/workflows`
   Rule: `HLT-009-GENERATED-SECURITY`
   Check: `HLT-009-GENERATED-SECURITY:security` `hard` confidence `0.95`
   Route: TLR `Security, secrets, agency`, lane `security`, owner `ops`
   Docs: `docs/audit-rubric.md#top-level-risk-mapping`
   Reason: high-risk repo has no explicit security lane
   Fix: add a dedicated security lane with secret scanning, dependency review, and workflow linting
   Rerun: `just security`
   Fingerprint: `sha256:c249be982d975721833fe396cdfff422f53a2d61819df881968fba63fdd6b9bf`
   Evidence: no security lane markers found
5. `high` `security` `.github/workflows`
   Rule: `HLT-016-SUPPLY-CHAIN-DRIFT`
   Check: `HLT-016-SUPPLY-CHAIN-DRIFT:security` `hard` confidence `0.95`
   Route: TLR `Security, secrets, agency`, lane `security`, owner `ops`
   Docs: `docs/audit-rubric.md#top-level-risk-mapping`
   Reason: no secret or dependency scanning was found in CI
   Fix: add secret scanning, dependency review, and SBOM or provenance checks to CI
   Rerun: `just security`
   Fingerprint: `sha256:2e22551cbdbd8da1f6fedd2d509dba064990dc4b1505df71609f431d11901099`
   Evidence: no CI scan markers found
6. `medium` `security` `.github/workflows/jankurai.yml`
   Rule: `HLT-016-SUPPLY-CHAIN-DRIFT`
   Check: `HLT-016-SUPPLY-CHAIN-DRIFT:security` `soft` confidence `0.76`
   Route: TLR `Security, secrets, agency`, lane `security`, owner `ops`
   Docs: `docs/audit-rubric.md#top-level-risk-mapping`
   Reason: `Security and supply-chain posture` scored 14 below the standard floor of 85
   Fix: wire secret, dependency, provenance, and workflow scans into an operational CI lane
   Rerun: `just security`
   Fingerprint: `sha256:aeab45cf0cc735860f1f705a3ce28e86916fdfcfc262b30e2556eb6a6afb16d0`
   Evidence: typescript bad-behavior advisory signals: 1, no explicit security lane found, CI does not run the jankurai audit
7. `medium` `context` `AGENTS.md`
   Rule: `HLT-015-CONTEXT-SETUP-GAP`
   Check: `HLT-015-CONTEXT-SETUP-GAP:context` `soft` confidence `0.76`
   Route: TLR `Context/setup`, lane `fast`, owner `agent`
   Docs: `docs/agent-native-standard.md`
   Reason: `Context economy and agent instructions` scored 37 below the standard floor of 85
   Fix: keep root guidance short and route durable detail through agent-readable manifests and docs
   Rerun: `just fast`
   Fingerprint: `sha256:96078500f84b10f914940e5746818deaabdd13bd39b94ec9869201a60b8ca5cc`
   Evidence: root `AGENTS.md` present, root `AGENTS.md` stays short, missing agent-readable docs: README.md, docs/architecture.md or docs/boundaries.md, docs/testing.md
8. `medium` `proof` `Justfile`
   Rule: `HLT-018-PERF-CONCURRENCY-DRIFT`
   Check: `HLT-018-PERF-CONCURRENCY-DRIFT:proof` `soft` confidence `0.76`
   Route: TLR `Verification`, lane `fast`, owner `workspace`
   Docs: `docs/testing.md`
   Reason: `Build speed signals` scored 0 below the standard floor of 85
   Fix: add fast deterministic build/test targets, caches, and narrow proof lanes for agent iteration
   Rerun: `just fast`
   Fingerprint: `sha256:0cf9476a83002d0f15381836511b813d73708c221d9dfc8a5f807059e514534d`
   Evidence: missing one-command setup/validation, missing deterministic fast lane
9. `medium` `boundary` `agent/boundaries.toml`
   Rule: `HLT-007-HANDWRITTEN-CONTRACT`
   Check: `HLT-007-HANDWRITTEN-CONTRACT:boundary` `soft` confidence `0.76`
   Route: TLR `Contracts/data`, lane `contract`, owner `agent`
   Docs: `docs/audit-rubric.md#known-vibe-coding-insults`
   Reason: `Contract and boundary integrity` scored 70 below the standard floor of 85
   Fix: add generated contracts and boundary checks for public APIs, data access, and cross-runtime seams
   Rerun: `just fast`
   Fingerprint: `sha256:3641cf1f456fee78e57ec43c0a70915f4efa9b8fa5e0d3f4e88de0f05ce34281`
   Evidence: contract surface found, generated contract artifacts found, TypeScript strict mode hinted by `tsconfig.json`, all contract sources have generated zone entries
10. `medium` `context` `agent/owner-map.json`
   Rule: `HLT-003-OWNERLESS-PATH`
   Check: `HLT-003-OWNERLESS-PATH:context` `soft` confidence `0.76`
   Route: TLR `Context/setup`, lane `fast`, owner `agent`
   Docs: `agent/JANKURAI_STANDARD.md#ownership-boundaries`
   Reason: `Ownership and navigation surface` scored 68 below the standard floor of 85
   Fix: tighten owner/test maps and root routing until agents can localize ownership without inference
   Rerun: `just fast`
   Fingerprint: `sha256:d8e937c4674c535079542faac18e6310c27ccb3b9c83d3529c9571d7db9d1cb8`
   Evidence: root `AGENTS.md` present, owner map covers audited paths, test map covers audited paths
11. `medium` `proof` `agent/test-map.json`
   Rule: `HLT-004-UNMAPPED-PROOF`
   Check: `HLT-004-UNMAPPED-PROOF:proof` `soft` confidence `0.76`
   Route: TLR `Verification`, lane `fast`, owner `agent`
   Docs: `agent/JANKURAI_STANDARD.md#proof-lanes`
   Reason: `Proof lanes and test routing` scored 16 below the standard floor of 85
   Fix: route each owned path to a deterministic proof command and make the lane executable in CI
   Rerun: `just fast`
   Fingerprint: `sha256:da70e3a0c2fc706a94f500536c3807464e1fa140f87f65931a02106865bf9786`
   Evidence: web e2e lane present or no web surface, DOM geometry UX QA runtime found, Rust property/integration tests present or no Rust surface, no one-command setup/validation lane
12. `high` `ux-qa` `apps/web`
   Rule: `HLT-013-RENDERED-UX-GAP`
   Check: `HLT-013-RENDERED-UX-GAP:ux-qa` `hard` confidence `0.88`
   Route: TLR `Verification and rendered UX`, lane `web`, owner `tools`
   Docs: `docs/testing.md`
   Reason: web surface lacks layered rendered UX QA evidence
   Fix: add Storybook state coverage, Playwright screenshots, visual review or `@jankurai/ux-qa`, accessibility scans, CLS checks, generated mocks, and design tokens
   Rerun: `just ux-qa`
   Fingerprint: `sha256:571d35c2e730a393b782bac14825b197c0543920bb21967079d264ac602ea5b1`
   Evidence: rendered UX QA lane missing
13. `high` `exceptions` `crates/domain`
   Rule: `HLT-017-OPAQUE-OBSERVABILITY`
   Check: `HLT-017-OPAQUE-OBSERVABILITY:exceptions` `hard` confidence `0.88`
   Route: TLR `Repair`, lane `observability`, owner `tools`
   Docs: `agent/JANKURAI_STANDARD.md#repair-receipts`
   Reason: no agent-friendly exception/error pattern was detected
   Fix: define a typed exception surface with purpose, reason, common fixes, docs_url, and repair_hint so the next rerun is local
   Rerun: `just score`
   Fingerprint: `sha256:538667a01e35d8e91eae100627364816dd225911862fa2fa1578642af63d4af8`
   Evidence: route repair work to the next agent, opaque failures slow local debugging and reruns, add a typed repair hint; name the common fixes; point at the local docs URL, docs/testing.md
14. `medium` `data` `db/`
   Rule: `HLT-006-DIRECT-DB-WRONG-LAYER`
   Check: `HLT-006-DIRECT-DB-WRONG-LAYER:data` `soft` confidence `0.76`
   Route: TLR `Contracts/data`, lane `db`, owner `tools`
   Docs: `docs/audit-rubric.md#required-shape`
   Reason: `Data truth and workflow safety` scored 30 below the standard floor of 85
   Fix: move durable truth into migrations, constraints, adapters, and application-owned transactions
   Rerun: `just fast`
   Fingerprint: `sha256:0ad5f62609e845dec62d4e1316b6905fc72ddb02042f7673fa53643ea7d2419d`
   Evidence: strict DB boundary violation: src/MeshViewer.test.tsx, direct DB access leaks out of the data boundary
15. `medium` `docs` `docs/`
   Check: `HLT-000-SCORE-DIMENSION:docs` `soft` confidence `0.76`
   Route: TLR `Context/setup`, lane `audit`, owner `standard`
   Reason: agent-readable documentation is incomplete
   Fix: add concise docs for architecture, boundaries, tests, generated zones, and audit rules; route them from root `AGENTS.md`
   Rerun: `just score`
   Fingerprint: `sha256:7a7bbff17bd45fa833f208a469d73fc717e5fd8687e3d8d20098aa2ce66f2e92`
   Evidence: README.md, docs/architecture.md or docs/boundaries.md, docs/testing.md
16. `high` `release` `docs/release.md`
   Rule: `HLT-025-RELEASE-READINESS-GAP`
   Check: `HLT-025-RELEASE-READINESS-GAP:release` `hard` confidence `0.88`
   Route: TLR `Verification`, lane `release`, owner `standard`
   Docs: `docs/testing.md`
   Matched term: `release structure`
   Reason: launch gates need artifact-backed release evidence
   Fix: add a release control surface with version source, changelog, release process docs, CI or script evidence, integrity/provenance evidence, and rollback guidance
   Rerun: `just check`
   Fingerprint: `sha256:33d19110bde7e6eb57aee22a917263242e35078f23e41ab68d7447fa9044b1a9`
   Evidence: release structure missing: version source, changelog, release process doc
17. `medium` `observability` `docs/testing.md`
   Rule: `HLT-017-OPAQUE-OBSERVABILITY`
   Check: `HLT-017-OPAQUE-OBSERVABILITY:observability` `soft` confidence `0.76`
   Route: TLR `Repair`, lane `observability`, owner `standard`
   Docs: `agent/JANKURAI_STANDARD.md#repair-receipts`
   Reason: `Observability and repair evidence` scored 31 below the standard floor of 85
   Fix: add structured errors, telemetry, and repair receipts that tell the next agent where to rerun proof
   Rerun: `just score`
   Fingerprint: `sha256:f30348433688a6607761404d271a702c7b4951143f120676fc18188f6d46e349`
   Evidence: repair receipts or raw artifact language found, no agent-friendly exception pattern found
18. `medium` `release` `docs/testing.md`
   Rule: `HLT-026-COST-BUDGET-GAP`
   Check: `HLT-026-COST-BUDGET-GAP:release` `soft` confidence `0.88`
   Route: TLR `Verification`, lane `release`, owner `standard`
   Docs: `docs/testing.md`
   Matched term: `budget`
   Reason: unbounded paid work needs budgets and stop conditions
   Fix: add explicit budgets, quotas, stop conditions, and kill-switch evidence for paid or unbounded operations
   Rerun: `just check`
   Fingerprint: `sha256:edd248b7afc24b644107205fa5b84a88103ac4b622009ff9f19b779de8798f59`
   Evidence: cost surface found without budget/stop-condition policy
19. `high` `data` `src/MeshViewer.test.tsx:1`
   Rule: `HLT-006-DIRECT-DB-WRONG-LAYER`
   Check: `HLT-006-DIRECT-DB-WRONG-LAYER:data` `hard` confidence `0.95`
   Route: TLR `Contracts/data`, lane `db`, owner `tools`
   Docs: `docs/audit-rubric.md#required-shape`
   Reason: direct database access appears in a wrong layer
   Fix: move SQL and DB clients to `crates/adapters` or `db/`; expose typed application/domain APIs upward
   Rerun: `just fast`
   Fingerprint: `sha256:a9f85a967c15f7e2195a7a70e38b5b2dc2364e5a498f1c347d0c9d8d0f2c24fc`
   Evidence: DB marker in non-adapter layer

## Policy

- Policy file: `./agent/audit-policy.toml`
- Minimum score: `85`
- Fail on: `critical, high`

## Agent Fix Queue

1. `high` `HLT-006-DIRECT-DB-WRONG-LAYER` `src/MeshViewer.test.tsx` - move SQL and DB clients to `crates/adapters` or `db/`; expose typed application/domain APIs upward
   Route: `Contracts/data`/`db`
2. `medium` `HLT-007-HANDWRITTEN-CONTRACT` `agent/boundaries.toml` - add generated contracts and boundary checks for public APIs, data access, and cross-runtime seams
   Route: `Contracts/data`/`contract`
3. `medium` `HLT-006-DIRECT-DB-WRONG-LAYER` `db/` - move durable truth into migrations, constraints, adapters, and application-owned transactions
   Route: `Contracts/data`/`db`
4. `high` `HLT-004-UNMAPPED-PROOF` `.` - add a fast lane that runs the narrowest deterministic proof loop and keep it canonical
   Route: `Verification`/`fast`
5. `high` `.` - add a canonical `setup`, `check`, `test`, or `verify` lane in one root command file
   Route: `Verification`/`fast`
6. `high` `HLT-025-RELEASE-READINESS-GAP` `docs/release.md` - add a release control surface with version source, changelog, release process docs, CI or script evidence, integrity/provenance evidence, and rollback guidance
   Route: `Verification`/`release`
7. `medium` `HLT-018-PERF-CONCURRENCY-DRIFT` `Justfile` - add fast deterministic build/test targets, caches, and narrow proof lanes for agent iteration
   Route: `Verification`/`fast`
8. `medium` `HLT-004-UNMAPPED-PROOF` `agent/test-map.json` - route each owned path to a deterministic proof command and make the lane executable in CI
   Route: `Verification`/`fast`
9. `medium` `HLT-026-COST-BUDGET-GAP` `docs/testing.md` - add explicit budgets, quotas, stop conditions, and kill-switch evidence for paid or unbounded operations
   Route: `Verification`/`release`
10. `high` `HLT-017-OPAQUE-OBSERVABILITY` `crates/domain` - define a typed exception surface with purpose, reason, common fixes, docs_url, and repair_hint so the next rerun is local
   Route: `Repair`/`observability`
11. `medium` `HLT-017-OPAQUE-OBSERVABILITY` `docs/testing.md` - add structured errors, telemetry, and repair receipts that tell the next agent where to rerun proof
   Route: `Repair`/`observability`
12. `high` `.github/workflows` - add a CI job that runs `jankurai . --json agent/repo-score.json --md agent/repo-score.md` and uploads both artifacts
   Route: `Context/setup`/`audit`
13. `medium` `HLT-015-CONTEXT-SETUP-GAP` `AGENTS.md` - keep root guidance short and route durable detail through agent-readable manifests and docs
   Route: `Context/setup`/`fast`
14. `medium` `HLT-003-OWNERLESS-PATH` `agent/owner-map.json` - tighten owner/test maps and root routing until agents can localize ownership without inference
   Route: `Context/setup`/`fast`
15. `medium` `docs/` - add concise docs for architecture, boundaries, tests, generated zones, and audit rules; route them from root `AGENTS.md`
   Route: `Context/setup`/`audit`
16. `high` `HLT-009-GENERATED-SECURITY` `.github/workflows` - add a dedicated security lane with secret scanning, dependency review, and workflow linting
   Route: `Security, secrets, agency`/`security`
17. `high` `HLT-016-SUPPLY-CHAIN-DRIFT` `.github/workflows` - add secret scanning, dependency review, and SBOM or provenance checks to CI
   Route: `Security, secrets, agency`/`security`
18. `high` `HLT-013-RENDERED-UX-GAP` `apps/web` - add Storybook state coverage, Playwright screenshots, visual review or `@jankurai/ux-qa`, accessibility scans, CLS checks, generated mocks, and design tokens
   Route: `Verification and rendered UX`/`web`
19. `medium` `HLT-016-SUPPLY-CHAIN-DRIFT` `.github/workflows/jankurai.yml` - wire secret, dependency, provenance, and workflow scans into an operational CI lane
   Route: `Security, secrets, agency`/`security`
