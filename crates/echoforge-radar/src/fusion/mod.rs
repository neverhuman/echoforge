//! Fusion: combine outputs from multiple detectors into a single, dedupd
//! event stream.
//!
//! See `graph.rs` for the runtime; `mod.rs` only re-exports the public
//! surface so callers can `use echoforge_radar::fusion::DetectorGraphRuntime`.

pub mod graph;

pub use graph::{DetectorGraphRuntime, FusedDetections};
