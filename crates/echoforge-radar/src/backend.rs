use crate::cfar::{ca_cfar_1d, CfarDecision, CfarParams};
use crate::pulse_compression::matched_filter;
use crate::waveform::{lfm_chirp, LfmChirp};
use crate::ComplexSample;

pub trait ArrayBackend {
    fn name(&self) -> &'static str;
    fn chirp(&self, config: &LfmChirp) -> Vec<ComplexSample>;
    fn pulse_compress(
        &self,
        received: &[ComplexSample],
        reference: &[ComplexSample],
    ) -> Vec<ComplexSample>;
    fn ca_cfar_1d(&self, power: &[f32], params: CfarParams) -> Vec<CfarDecision>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl ArrayBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn chirp(&self, config: &LfmChirp) -> Vec<ComplexSample> {
        lfm_chirp(config)
    }

    fn pulse_compress(
        &self,
        received: &[ComplexSample],
        reference: &[ComplexSample],
    ) -> Vec<ComplexSample> {
        matched_filter(received, reference)
    }

    fn ca_cfar_1d(&self, power: &[f32], params: CfarParams) -> Vec<CfarDecision> {
        ca_cfar_1d(power, params)
    }
}

