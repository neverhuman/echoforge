use crate::backend::ArrayBackend;
use crate::cfar::{ca_cfar_1d, CfarDecision, CfarParams};
use crate::pulse_compression::magnitude;
use crate::waveform::LfmChirp;
use crate::ComplexSample;

#[derive(Debug, Clone)]
pub struct RadarChainOutput {
    pub reference: Vec<ComplexSample>,
    pub compressed: Vec<ComplexSample>,
    pub magnitudes: Vec<f32>,
    pub cfar: Vec<CfarDecision>,
}

#[derive(Debug, Clone)]
pub struct RadarChain<B: ArrayBackend> {
    backend: B,
    cfar_params: CfarParams,
}

impl<B: ArrayBackend> RadarChain<B> {
    pub fn new(backend: B, cfar_params: CfarParams) -> Self {
        Self {
            backend,
            cfar_params,
        }
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    pub fn chirp(&self, config: &LfmChirp) -> Vec<ComplexSample> {
        self.backend.chirp(config)
    }

    pub fn pulse_compress(
        &self,
        received: &[ComplexSample],
        reference: &[ComplexSample],
    ) -> Vec<ComplexSample> {
        self.backend.pulse_compress(received, reference)
    }

    pub fn detect(&self, received: &[ComplexSample], reference: &[ComplexSample]) -> RadarChainOutput {
        let compressed = self.pulse_compress(received, reference);
        let magnitudes = magnitude(&compressed);
        let cfar = ca_cfar_1d(&magnitudes, self.cfar_params);

        RadarChainOutput {
            reference: reference.to_vec(),
            compressed,
            magnitudes,
            cfar,
        }
    }
}

