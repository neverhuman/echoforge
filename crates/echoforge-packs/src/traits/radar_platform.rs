//! `RadarPlatform` trait — loadable named radar platform card.
//!
//! A `RadarPlatform` instance is constructed from a `radar_platform_card`
//! payload (see `schemas/radar_platform_card.schema.json`). Concrete
//! implementations in `echoforge-radar` build a `RadarSimConfig` from
//! the card's antenna / transmit / receiver / scan blocks.

use std::any::Any;
use std::fmt::Debug;

use crate::traits::source::SourceMetadata;

/// Object-safe trait implemented by `echoforge-radar` once Wave 13
/// lands. The trait surface includes only the metadata accessor here;
/// the typed accessors (antenna gain, PRF, dwell, declared detection
/// envelope) live on the concrete `RadarPlatformCard` struct in
/// `echoforge-radar`.
pub trait RadarPlatform: Debug + Send + Sync {
    /// Re-uses the `SourceMetadata` shape (display name, validation
    /// tier, fidelity class) — pack and card slugs identify the
    /// platform, family is "radar_platform".
    fn metadata(&self) -> &SourceMetadata;
    fn as_any(&self) -> &dyn Any;
}
