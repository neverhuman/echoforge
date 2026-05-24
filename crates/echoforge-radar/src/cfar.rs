use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CfarParams {
    pub training_cells: usize,
    pub guard_cells: usize,
    pub pfa: f32,
}

impl CfarParams {
    pub fn new(training_cells: usize, guard_cells: usize, pfa: f32) -> Self {
        Self {
            training_cells,
            guard_cells,
            pfa,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CfarDecision {
    pub index: usize,
    pub evaluated: bool,
    pub detected: bool,
    pub statistic: f32,
    pub threshold: f32,
    pub noise_estimate: f32,
}

/// CA-CFAR threshold scale factor for **Gaussian** noise / Rayleigh
/// amplitude.
///
/// `alpha = N * (Pfa^(-1/N) - 1)` (Skolnik, *Introduction to Radar Systems*,
/// 3rd ed. §7.7.2). This is the textbook Cell-Averaging CFAR closed form
/// derived under the assumption that the training cells are i.i.d.
/// exponential (= power-domain Gaussian I+Q / Rayleigh amplitude).
///
/// **WARNING — Lane G_a, Wave 1 Expert Credibility Sweep**: applying this
/// formula to K-distributed, Weibull or log-normal clutter under-counts the
/// observed Pfa by 1–3 orders of magnitude (heavy tails produce sample
/// means much smaller than the population mean, which the Gaussian alpha
/// does not compensate for). It is also wrong for non-CA CFAR variants
/// (OS-CFAR uses the Rohling 1983 implicit equation instead). For any
/// non-Gaussian clutter or non-CA variant call
/// [`crate::detectors::cfar_alpha::resolve_alpha`] instead, which dispatches
/// to the correct closed form, library lookup, or Monte-Carlo calibration
/// per (variant, distribution) pair.
pub fn ca_cfar_scale(training_cells: usize, pfa: f32) -> f32 {
    if training_cells == 0 {
        return 0.0;
    }
    let n = training_cells as f32;
    n * (pfa.powf(-1.0 / n) - 1.0)
}

pub fn ca_cfar_1d(power: &[f32], params: CfarParams) -> Vec<CfarDecision> {
    let mut decisions = Vec::with_capacity(power.len());
    let alpha = ca_cfar_scale(params.training_cells, params.pfa);
    let window = params.training_cells + params.guard_cells;

    for index in 0..power.len() {
        if index < window || index + window >= power.len() {
            decisions.push(CfarDecision {
                index,
                evaluated: false,
                detected: false,
                statistic: power[index],
                threshold: f32::INFINITY,
                noise_estimate: 0.0,
            });
            continue;
        }

        let left_start = index - window;
        let left_end = index - params.guard_cells;
        let right_start = index + params.guard_cells + 1;
        let right_end = index + window + 1;

        let mut sum = 0.0f32;
        let mut count = 0usize;

        for value in &power[left_start..left_end] {
            sum += *value;
            count += 1;
        }
        for value in &power[right_start..right_end] {
            sum += *value;
            count += 1;
        }

        let noise_estimate = if count == 0 { 0.0 } else { sum / count as f32 };
        let threshold = alpha * noise_estimate;
        let statistic = power[index];

        decisions.push(CfarDecision {
            index,
            evaluated: true,
            detected: statistic > threshold,
            statistic,
            threshold,
            noise_estimate,
        });
    }

    decisions
}
