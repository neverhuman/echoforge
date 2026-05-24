//! `Source` trait — loadable adversary / confuser / commercial-reference platform.
//!
//! A `Source` instance is constructed from a `source_platform_card`
//! payload (see `schemas/source_platform_card.schema.json`) and exposes
//! the per-target physics fields that the radar / EO-IR / acoustic /
//! passive-RF chains consume.

use std::any::Any;
use std::fmt::Debug;

/// Per-card metadata surfaced for `ef pack show source <slug>` and
/// downstream debugging.
#[derive(Debug, Clone)]
pub struct SourceMetadata {
    pub pack_slug: String,
    pub card_slug: String,
    pub display_name: String,
    pub platform_family: String,
    pub platform_role: String,
    pub validation_tier: String,
    pub fidelity_class: Option<String>,
}

/// Object-safe trait implemented by downstream crates (typically
/// `echoforge-radar`) once Wave 13 lands the runtime migration. This
/// crate only owns the trait *signature*; impl details live with the
/// physics modules that consume the card payload.
pub trait Source: Debug + Send + Sync {
    /// Stable per-card metadata.
    fn metadata(&self) -> &SourceMetadata;

    /// Downcast escape hatch for consumers that need typed access to
    /// the underlying card payload (e.g. the radar chain needs the
    /// `rcs_signature` block as a typed struct, not the raw JSON
    /// `Value`).
    fn as_any(&self) -> &dyn Any;
}
