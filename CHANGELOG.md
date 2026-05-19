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
