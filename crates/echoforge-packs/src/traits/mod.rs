//! Trait definitions for pack-loadable platform components.
//!
//! Wave 8 ships only the trait *signatures*. Concrete implementations
//! that read a card payload and produce a working `Box<dyn Source>` /
//! `Box<dyn Detector>` / `Box<dyn RadarPlatform>` live in
//! `crates/echoforge-radar/` and are wired behind a future
//! `pack-registry` feature flag (Wave 13 — runtime migration).
//!
//! These traits exist now so that Wave 9 (`named-platform-case-study-v1`)
//! and downstream consumers can compile against the loader's surface
//! without needing the full implementation. The Wave 13 migration
//! converts existing hard-coded constructors into card-driven factory
//! methods that satisfy these traits.

mod detector;
mod radar_platform;
mod source;

pub use detector::{Detector, DetectorMetadata};
pub use radar_platform::RadarPlatform;
pub use source::{Source, SourceMetadata};
