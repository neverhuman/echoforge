use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::ComplexSample;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LfmChirp {
    pub sample_rate_hz: f64,
    pub pulse_width_s: f64,
    pub bandwidth_hz: f64,
    pub carrier_hz: f64,
    pub initial_phase_rad: f64,
}

impl LfmChirp {
    pub fn sample_count(&self) -> usize {
        (self.sample_rate_hz * self.pulse_width_s).round() as usize
    }

    pub fn samples(&self) -> Vec<ComplexSample> {
        lfm_chirp(self)
    }
}

pub fn lfm_chirp(config: &LfmChirp) -> Vec<ComplexSample> {
    let sample_count = config.sample_count();
    if sample_count == 0 {
        return Vec::new();
    }

    let dt = 1.0 / config.sample_rate_hz;
    let slope = config.bandwidth_hz / config.pulse_width_s;
    let center = config.pulse_width_s / 2.0;

    (0..sample_count)
        .map(|index| {
            let t = index as f64 * dt;
            let centered_t = t - center;
            let phase = config.initial_phase_rad
                + 2.0 * PI * (config.carrier_hz * t + 0.5 * slope * centered_t * centered_t);
            ComplexSample::new(phase.cos() as f32, phase.sin() as f32)
        })
        .collect()
}
