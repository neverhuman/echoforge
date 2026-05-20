//! `Detector` trait — loadable detection algorithm card.
//!
//! Concrete impls (CA-CFAR, OS-CFAR, MTI/MTD, TBD/Hough, micro-Doppler
//! LRT classifier, acoustic-radar Bayesian fusion, ...) live in
//! `crates/echoforge-radar/src/detectors/` and the loader's factory
//! table maps each `algorithm_family` slug to the appropriate
//! constructor.

use std::any::Any;
use std::fmt::Debug;

/// Per-card metadata surfaced for `ef pack show detector <slug>`.
#[derive(Debug, Clone)]
pub struct DetectorMetadata {
    pub pack_slug: String,
    pub card_slug: String,
    pub display_name: String,
    pub algorithm_family: String,
    pub intended_target_classes: Vec<String>,
    pub intended_sensor_archetypes: Vec<String>,
    pub validation_tier: String,
    pub fidelity_class: Option<String>,
}

/// Object-safe trait implemented by `echoforge-radar` once Wave 13
/// lands. The `process(...)` signature deliberately uses opaque types
/// here so `echoforge-packs` does not depend on `echoforge-radar` (the
/// dependency is the other way around to avoid a cycle).
pub trait Detector: Debug + Send + Sync {
    fn metadata(&self) -> &DetectorMetadata;
    fn as_any(&self) -> &dyn Any;
}
