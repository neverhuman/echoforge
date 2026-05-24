# Changelog

All notable changes to EchoForge are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
EchoForge adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.13] - 2026-05-24

### Changed
- Added the v8 paper feedback bundle, including the new detection helpers, paper evidence builders, source appendix generator and manifest, regenerated figures and PDF, refreshed validation scripts, and updated SBOM output.
- Bumped the workspace release metadata to `0.2.13`.
- Pinned CI workflows to the repo's local Rust `1.95.0` and Node `26.1.0` toolchain baseline on Ubuntu 24.04.

## [0.2.12] - 2026-05-23

### Changed
- Added the v7 paper traceability pass with data-processing evidence rows, a processing-flow figure, radar-review narrative clarifications, and validator guards.
- Bumped the workspace release metadata to `0.2.12`.

## [0.2.10] - 2026-05-22

### Changed
- Completed the v2 paper feedback package and aligned the release bundle with the updated paper, figures, and validation outputs.
- Bumped the workspace release metadata to `0.2.10`.

## [0.2.9] - 2026-05-21

### Changed
- Added the Shahed-136/Geran-2 `runit` main-run generator, detector baseline, and regression test coverage.
- Documented the main-run split policy, artifact layout, and detector outputs in `docs/main_run.md`.
- Added the README main-run performance example and release-payload command pair for the public-proxy corpus.
- Bumped the workspace release version to `0.2.9`.

## [0.2.8] - 2026-05-21

### Changed
- Improved the Studio README media capture flow to build and verify a clearer `1280x720` live radar GIF.
- Bumped the workspace release version to `0.2.8`.

## [0.2.7] - 2026-05-21

### Changed
- Bumped the workspace release version to `0.2.7`.

## [0.2.6] - 2026-05-20

### Added
- Native scene rosters for public-proxy hard-negative families, including static glint, stationary return, flock, ghost, and zero-target clutter-only cases.
- Radar noise, clutter, interference, and validation backlog notes for future public-proxy robustness work.

### Changed
- Dataset frame features and micro-Doppler descriptors now derive from synthesized range/IQ products instead of envelope-only proxies.

### Fixed
- Radar scene synthesis now supports empty target rosters for clutter-only scenarios while preserving finite diagnostic link-budget metadata.

## [0.2.5] - 2026-05-20

### Added
- Real-data measured-anchor registry, reference-only adapter lane, observable-only feature policy, and realism gate tests.
- Jankurai governance compliance (score >= 85)
- Playwright E2E test lane (`web-e2e`)
- Structured CoreError with JSON-parseable machine-readable output
- Property-based tests for echoforge-core and echoforge-radar
- Agent-readable AGENTS.md in contracts/, ops/, apps/web/, python/ai-service/, detection/
- docs/release.md and docs/testing.md

### Changed
- Renamed ValidationTier::Placeholder to Unscored throughout echoforge-sig
- Renamed stub adapter version to unwired-0.1.0 in echoforge-solver-sagitta
- Replaced deprecated API usage in echoforge-radar with active equivalents

### Fixed
- 25 malformed agent receipts (added required sections)
- CI workflows pinned to full GitHub Actions commit SHAs
- MeshViewer.tsx ESLint suppression removed

## [0.2.2] - 2026-05-20

### Changed
- Bumped the workspace release version to `0.2.2`.
- Captured PR-ready validation for the radar console streaming branch.

## [0.2.1] - 2026-05-20

### Changed
- Bumped the workspace release version to `0.2.1`.
- Updated the README to show the current detection lane entrypoint and the top-three public-proxy modeling results.

### Fixed
- Consolidated AUC report generation now retains all method/horizon rows instead of replacing prior horizons for the same method.
- Added ignore coverage for generated `.joblib` model artifacts.

## [0.2.0] - 2026-05-20

### Added
- Rust-backed studio run history, replay, and download metadata endpoints
- Validation-gated public-proxy run cards with reproducibility metadata
- Operator-studio views for live console, runs, detectors, and dataset export
- Fast staged-file jankurai ratchet hook for pre-commit workflows

### Changed
- EchoForge workspace version moved to `0.2.0`
- WebSocket control frames now carry run lifecycle, artifact, validation, and backpressure notices
- The studio UI now surfaces stable run IDs, seeded artifacts, and export gates

### Fixed
- Broke the pre-commit jankurai audit into staged-file ratchet behavior
