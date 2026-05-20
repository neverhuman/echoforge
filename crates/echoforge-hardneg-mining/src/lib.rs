//! Hard-negative mining loop.
//!
//! Implements the closed-loop curriculum described in FUCKIT.md's Critical
//! Review section I ("Hard-negative mining loop is the four-card pending,
//! not a loop"). One iteration looks like:
//!
//! ```text
//!   campaign rollups  --[cluster]-->  trouble clusters
//!                                          |
//!                                  [synthesize_batch]
//!                                          v
//!                            proposed variant deltas
//!                                          |
//!                                  curator review (apply_variants packet,
//!                                                  not in this crate's scope)
//!                                          |
//!                                          v
//!                            airspace-objects.json grows
//! ```
//!
//! See the per-module docs for the data structures involved.
//!
//! The crate intentionally has **zero dependencies** on `echoforge-radar`
//! or `echoforge-dataset` — it consumes the on-disk JSON shapes that those
//! crates emit, not their Rust types. That keeps the mining loop usable
//! against any campaign root regardless of which streaming runner version
//! produced it.

pub mod cluster;
pub mod error;
pub mod loop_driver;
pub mod synthesize;
pub mod types;

pub use cluster::{FailureCluster, FailureClusterStore, MAX_SAMPLE_RECORDS_PER_CLUSTER};
pub use error::MiningError;
pub use loop_driver::{
    run_one_iteration, IterationManifest, MiningIterationReport, MiningLoopConfig,
};
pub use synthesize::{
    compute_widen_factor, synthesize_batch, synthesize_variant, ObjectClassDelta,
    DEFAULT_WIDEN_GAIN, RCS_UPPER_BUMP_DB, WIDEN_MAX, WIDEN_MIN,
};
pub use types::{
    AirspaceObjectsConfig, KinematicsBounds, MicroMotionBounds, ModelEvalRollup, ObjectClass,
    SensorObservableBounds,
};
