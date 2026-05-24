use crate::backend::ArrayBackend;
use crate::cfar::{ca_cfar_1d, CfarDecision, CfarParams};
use crate::pulse_compression::{coefficients, magnitude, CompressionWindow};
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
    compression_window: CompressionWindow,
}

impl<B: ArrayBackend> RadarChain<B> {
    /// Construct a chain with the default Taylor-35 amplitude weighting
    /// applied to the matched-filter reference. This is the Lane E
    /// credibility win: range sidelobes sit near -33 dB instead of the
    /// raw matched-filter -13 dB.
    pub fn new(backend: B, cfar_params: CfarParams) -> Self {
        Self {
            backend,
            cfar_params,
            compression_window: CompressionWindow::taylor_default(),
        }
    }

    /// Override the compression window. Pass [`CompressionWindow::None`]
    /// to recover the byte-stable pre-Lane-E behaviour (raw matched
    /// filter, no taper).
    pub fn with_compression_window(mut self, window: CompressionWindow) -> Self {
        self.compression_window = window;
        self
    }

    pub fn compression_window(&self) -> CompressionWindow {
        self.compression_window
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    pub fn chirp(&self, config: &LfmChirp) -> Vec<ComplexSample> {
        self.backend.chirp(config)
    }

    /// Raw pulse compression — no window — exposed so existing callers
    /// (e.g. detector graph integration tests) keep their byte-stable
    /// behaviour even after the chain default switched to Taylor-35.
    pub fn pulse_compress(
        &self,
        received: &[ComplexSample],
        reference: &[ComplexSample],
    ) -> Vec<ComplexSample> {
        self.backend.pulse_compress(received, reference)
    }

    pub fn detect(
        &self,
        received: &[ComplexSample],
        reference: &[ComplexSample],
    ) -> RadarChainOutput {
        // Pre-window the reference so any backend (CPU, GPU unimplemented) sees a
        // tapered template. This keeps the `ArrayBackend` trait surface
        // unchanged but threads the Taylor-35 default through.
        let windowed_reference = apply_window(reference, self.compression_window);
        let compressed = self.backend.pulse_compress(received, &windowed_reference);
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

fn apply_window(reference: &[ComplexSample], window: CompressionWindow) -> Vec<ComplexSample> {
    if matches!(window, CompressionWindow::None) {
        return reference.to_vec();
    }
    let coeffs = coefficients(window, reference.len());
    reference
        .iter()
        .zip(coeffs.iter())
        .map(|(sample, w)| ComplexSample::new(sample.re * w, sample.im * w))
        .collect()
}
