mod backend;
mod cfar;
mod chain;
pub mod detectors;
pub mod fusion;
mod pulse_compression;
mod sim;
mod tracking_fusion;
mod waveform;

pub use backend::{ArrayBackend, CpuBackend, GpuBackendUnavailable};
pub use cfar::{ca_cfar_1d, ca_cfar_scale, CfarDecision, CfarParams};
pub use chain::{RadarChain, RadarChainOutput};
pub use detectors::{
    BlobRdDetector, BlobRdParams, CaCfarDetector, DetectionEvent, DetectionKind, Detector,
    GoCfarDetector, MicroDopplerDetector, MicroDopplerParams, OsCfarDetector, OsCfarParams,
    RangeDoppler, SoCfarDetector,
};
pub use fusion::{DetectorGraphRuntime, FusedDetections};
pub use pulse_compression::{magnitude, matched_filter, pulse_compress};
pub use sim::{
    range_bin_to_m, synthesize_takeoff_episode, DetectionRecord, EpisodeSeed, NoiseProfile,
    RadarSimConfig, SyntheticEpisode, TakeoffProfile, TargetState,
};
pub use tracking_fusion::{TrackingFusionAdapter, TrackingFusionReport, TrackingTrack};
pub use waveform::{lfm_chirp, LfmChirp};

pub type ComplexSample = num_complex::Complex<f32>;
