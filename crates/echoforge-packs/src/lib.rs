//! `echoforge-packs` — pluggable pack loader for EchoForge.
//!
//! Discovers packs under `object-packs/<pack>/pack.manifest.json`, validates
//! cards against `schemas/*.schema.json`, and exposes a typed in-memory
//! [`Registry`] keyed by `(pack_slug, card_slug)`. Pack manifests use either
//! the prior v1 format (`object_pack_manifest.schema.json` /
//! `hard_negative_pack.schema.json`) or the unified v2 format
//! (`pack_v2.schema.json`); the loader auto-detects via the `schema_version`
//! discriminator.
//!
//! ## Design contracts
//!
//! - **Strictly additive**: the loader runs alongside the existing hard-coded
//!   `RadarSimConfig::default()` and `TargetClass` enum paths. Nothing in
//!   `echoforge-radar` is rewired to depend on the registry; downstream
//!   crates that want card-driven behaviour opt in via the
//!   `echoforge-radar/pack-registry` feature flag (not landed in this
//!   wave).
//! - **Strict-open posture**: card validation refuses any payload that fails
//!   its declared schema; the registry never silently coerces or back-fills
//!   missing fields.
//! - **Deterministic ids**: when cards arrive with the pending
//!   fingerprint (`0000…0000`), the loader recomputes it via
//!   [`echoforge_core::fingerprint_sha256`] over the card payload and
//!   rewrites the `id` field in memory.
//! - **Trait factories**: per-card-kind `Box<dyn Source>` / `Box<dyn
//!   Detector>` / `Box<dyn RadarPlatform>` construction is the contract
//!   that downstream crates implement; this crate only ships the trait
//!   definitions + the registry that stores `Arc<CardPayload>` references.
//!
//! ## Module layout
//!
//! - [`registry`] — typed in-memory `Registry` and `PackEntry` shapes.
//! - [`discovery`] — filesystem walker that finds `object-packs/*/pack.manifest.json`.
//! - [`validation`] — JSON-schema validation against the EchoForge schema set.
//! - [`traits`] — `Source` / `Detector` / `RadarPlatform` trait definitions.
//!
//! ## Strict-open evidence ladder
//!
//! Every registered card carries its declared `validation.tier` (V0–V5)
//! and `validation.fidelity_class` (F0–F5). The registry exposes a
//! [`Registry::min_tier_across_pack`] helper so consumers can refuse to
//! emit downstream artefacts below a required evidence floor.

pub mod discovery;
pub mod error;
pub mod manifest;
pub mod registry;
pub mod traits;
pub mod validation;

pub use error::PackError;
pub use manifest::{ManifestVersion, PackManifest};
pub use registry::{CardPayload, PackEntry, Registry};
pub use traits::{Detector, DetectorMetadata, RadarPlatform, Source, SourceMetadata};

/// Discover packs under `<repo_root>/object-packs/` and return a populated
/// [`Registry`]. Convenience wrapper around [`Registry::discover`].
pub fn discover_default(repo_root: &std::path::Path) -> Result<Registry, PackError> {
    Registry::discover(&repo_root.join("object-packs"))
}
