# Changelog

All notable changes to EchoForge are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
EchoForge adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
