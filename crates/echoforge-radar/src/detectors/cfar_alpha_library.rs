//! CFAR alpha library: Monte-Carlo calibrator and precomputed lookup table.
//! Extracted from cfar_alpha.rs for LOC compliance.

use super::cfar_alpha_helpers::{distribution_matches, pfa_close, sample_distribution, AlphaRng};
use super::{CfarVariant, NoiseDistribution};

/// Monte-Carlo calibration: draws `trials` windows, computes the noise
/// estimator per-variant, collects the (CUT / noise) ratios, and returns
/// the empirical `(1 - pfa)`-quantile of the ratio distribution.
pub fn calibrate_alpha_monte_carlo(
    variant: CfarVariant,
    distribution: NoiseDistribution,
    training_cells: usize,
    guard_cells: usize,
    pfa: f32,
    trials: usize,
    seed: u64,
) -> f32 {
    if training_cells == 0 || trials == 0 {
        return 0.0;
    }
    if let CfarVariant::OrderedStatistic { rank } = variant {
        if rank == 0 || rank > 2 * training_cells {
            return 0.0;
        }
    }
    let mut rng = AlphaRng::new(seed);
    let mut ratios: Vec<f64> = Vec::with_capacity(trials);
    let window_size = 2 * training_cells + 2 * guard_cells + 1;
    let mut buf = vec![0.0f64; window_size];

    for _ in 0..trials {
        for slot in buf.iter_mut() {
            *slot = sample_distribution(&mut rng, distribution);
        }
        let cut = buf[training_cells + guard_cells];
        let lead = &buf[0..training_cells];
        let lag = &buf[training_cells + 2 * guard_cells + 1..window_size];
        let noise = match variant {
            CfarVariant::CellAveraging => {
                let s: f64 = lead.iter().sum::<f64>() + lag.iter().sum::<f64>();
                s / (lead.len() + lag.len()) as f64
            }
            CfarVariant::GreatestOf => {
                let m1 = lead.iter().sum::<f64>() / lead.len() as f64;
                let m2 = lag.iter().sum::<f64>() / lag.len() as f64;
                m1.max(m2)
            }
            CfarVariant::SmallestOf => {
                let m1 = lead.iter().sum::<f64>() / lead.len() as f64;
                let m2 = lag.iter().sum::<f64>() / lag.len() as f64;
                m1.min(m2)
            }
            CfarVariant::OrderedStatistic { rank } => {
                let mut combined: Vec<f64> = Vec::with_capacity(lead.len() + lag.len());
                combined.extend_from_slice(lead);
                combined.extend_from_slice(lag);
                combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = (rank - 1).min(combined.len() - 1);
                combined[idx]
            }
        };
        if noise.abs() < 1e-30 {
            continue;
        }
        ratios.push(cut / noise);
    }

    if ratios.is_empty() {
        return 0.0;
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = (1.0 - pfa as f64).clamp(0.0, 1.0);
    let idx = ((q * ratios.len() as f64).round() as isize - 1).clamp(0, ratios.len() as isize - 1)
        as usize;
    ratios[idx] as f32
}

// =========================================================================
// Library lookup for K / Weibull (precomputed via the MC calibrator above).
// =========================================================================

#[derive(Debug, Clone, Copy)]
pub(super) struct AlphaEntry {
    pub variant: CfarVariant,
    pub distribution: NoiseDistribution,
    pub training_cells: usize,
    pub pfa: f32,
    pub alpha: f32,
}

const fn ca_entry(
    distribution: NoiseDistribution,
    training_cells: usize,
    alpha: f32,
) -> AlphaEntry {
    AlphaEntry {
        variant: CfarVariant::CellAveraging,
        distribution,
        training_cells,
        pfa: 1e-3,
        alpha,
    }
}

const fn os_entry(
    rank: usize,
    distribution: NoiseDistribution,
    training_cells: usize,
    alpha: f32,
) -> AlphaEntry {
    AlphaEntry {
        variant: CfarVariant::OrderedStatistic { rank },
        distribution,
        training_cells,
        pfa: 1e-3,
        alpha,
    }
}

pub(super) const ALPHA_LIBRARY: &[AlphaEntry] = &[
    ca_entry(NoiseDistribution::Weibull { shape: 1.2 }, 16, 20.43),
    ca_entry(NoiseDistribution::Weibull { shape: 1.2 }, 24, 18.71),
    ca_entry(NoiseDistribution::Weibull { shape: 2.0 }, 16, 7.59),
    ca_entry(NoiseDistribution::Weibull { shape: 2.0 }, 24, 7.27),
    ca_entry(NoiseDistribution::KDistribution { shape: 0.8 }, 16, 23.06),
    ca_entry(NoiseDistribution::KDistribution { shape: 0.8 }, 24, 21.59),
    ca_entry(NoiseDistribution::KDistribution { shape: 2.0 }, 16, 14.38),
    ca_entry(NoiseDistribution::KDistribution { shape: 2.0 }, 24, 14.02),
    os_entry(12, NoiseDistribution::Weibull { shape: 1.2 }, 16, 147.0),
    os_entry(18, NoiseDistribution::Weibull { shape: 1.2 }, 24, 122.2),
    os_entry(
        12,
        NoiseDistribution::KDistribution { shape: 0.8 },
        16,
        152.25,
    ),
    os_entry(
        18,
        NoiseDistribution::KDistribution { shape: 0.8 },
        24,
        134.47,
    ),
];

pub fn alpha_library_lookup(
    variant: CfarVariant,
    distribution: NoiseDistribution,
    training_cells: usize,
    pfa: f32,
) -> Option<f32> {
    for entry in ALPHA_LIBRARY {
        if entry.variant == variant
            && distribution_matches(entry.distribution, distribution)
            && entry.training_cells == training_cells
            && pfa_close(entry.pfa, pfa)
        {
            return Some(entry.alpha);
        }
    }
    None
}

pub fn alpha_library_len() -> usize {
    ALPHA_LIBRARY.len()
}
