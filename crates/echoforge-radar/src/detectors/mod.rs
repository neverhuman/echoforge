//! Detector graph: pluggable detection primitives that emit a common
//! `DetectionEvent`. Each detector implements the `Detector` trait so the
//! fusion runtime can chain heterogeneous detectors uniformly.
//!
//! All detectors are deterministic, allocation-explicit, and CPU only. The
//! 1-D CFAR variants (CA/OS/GO/SO) consume `&[f32]` power vectors; the 2-D
//! detectors operate on `RangeDoppler` grids.

use serde::{Deserialize, Serialize};

pub mod blob_rd;
pub mod ca_cfar;
pub mod cfar_alpha;
pub mod cfar_closed_forms;
pub mod go_cfar;
pub mod micro_doppler_classifier;
pub mod micro_doppler_feature;
pub mod os_cfar;
pub mod phase_tiered;
pub mod so_cfar;
pub mod tbd;

pub use blob_rd::{BlobRdDetector, BlobRdParams};
pub use ca_cfar::CaCfarDetector;
pub use go_cfar::GoCfarDetector;
pub use micro_doppler_feature::{MicroDopplerDetector, MicroDopplerParams};
pub use os_cfar::{order_index_for_quantile, OsCfarDetector, OsCfarParams};
pub use phase_tiered::{
    BoostDecision, BoostTierConfig, BoostTierDetector, ClimbDecision, ClimbOutTierDetector,
    ClimbTierConfig, CruiseDecision, CruiseTierConfig, CruiseTierDetector, KinematicGate,
    KinematicObservation, KinematicSample, PhaseTieredConfig, PhaseTieredDecision,
    PhaseTieredDetector, PropulsionClass, SpeedClassifier, Tier, TierArbiter, TierTransition,
};
pub use so_cfar::SoCfarDetector;

/// Tag identifying which detector produced an event. Useful for diagnostics
/// and for the fusion runtime to attribute per-detector counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DetectionKind {
    Threshold,
    CaCfar,
    OsCfar,
    GoCfar,
    SoCfar,
    Blob,
    MicroDoppler,
}

/// A single detection emitted by any detector in the graph. `doppler_bin`
/// is `None` for 1-D (range-only) detectors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionEvent {
    pub range_bin: usize,
    pub doppler_bin: Option<usize>,
    pub magnitude_db: f64,
    pub kind: DetectionKind,
}

impl DetectionEvent {
    pub fn new(
        range_bin: usize,
        doppler_bin: Option<usize>,
        magnitude_db: f64,
        kind: DetectionKind,
    ) -> Self {
        Self {
            range_bin,
            doppler_bin,
            magnitude_db,
            kind,
        }
    }
}

/// Convert linear magnitude (or power) into decibels. Uses a 1e-12 floor to
/// keep zero-magnitude bins finite.
pub fn magnitude_to_db(linear: f64) -> f64 {
    10.0 * (linear.max(1e-12)).log10()
}

/// Common detector contract. `Input` is whatever shape the detector consumes
/// (e.g. `Vec<f32>` for 1-D power, `RangeDoppler` for 2-D grids).
pub trait Detector {
    type Input;
    fn detect(&self, input: &Self::Input) -> Vec<DetectionEvent>;
    fn kind(&self) -> DetectionKind;
}

/// Minimal 2-D range-Doppler grid. `data[range_bin * doppler_bins +
/// doppler_bin]` is the (linear) magnitude at that cell. Rows are range,
/// columns are doppler.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeDoppler {
    pub range_bins: usize,
    pub doppler_bins: usize,
    pub data: Vec<f32>,
}

impl RangeDoppler {
    pub fn new(range_bins: usize, doppler_bins: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            range_bins * doppler_bins,
            "RangeDoppler data length must equal range_bins * doppler_bins"
        );
        Self {
            range_bins,
            doppler_bins,
            data,
        }
    }

    pub fn zeros(range_bins: usize, doppler_bins: usize) -> Self {
        Self::new(
            range_bins,
            doppler_bins,
            vec![0.0f32; range_bins * doppler_bins],
        )
    }

    #[inline]
    pub fn get(&self, range_bin: usize, doppler_bin: usize) -> f32 {
        self.data[range_bin * self.doppler_bins + doppler_bin]
    }

    #[inline]
    pub fn set(&mut self, range_bin: usize, doppler_bin: usize, value: f32) {
        self.data[range_bin * self.doppler_bins + doppler_bin] = value;
    }
}
