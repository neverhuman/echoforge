mod backend;
mod cfar;
mod chain;
mod pulse_compression;
mod waveform;

pub use backend::{ArrayBackend, CpuBackend};
pub use cfar::{ca_cfar_1d, ca_cfar_scale, CfarDecision, CfarParams};
pub use chain::{RadarChain, RadarChainOutput};
pub use pulse_compression::{magnitude, matched_filter, pulse_compress};
pub use waveform::{lfm_chirp, LfmChirp};

pub type ComplexSample = num_complex::Complex<f32>;

