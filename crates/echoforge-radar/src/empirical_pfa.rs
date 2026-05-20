//! Empirical Pfa calibration: Monte-Carlo measurement of observed
//! false-alarm probability for each (`ClutterRegime`, `CfarVariant`) tuple.
//!
//! The headline credibility artifact: a radar engineer can read
//! `outputs/empirical_pfa/<UTC>.jsonl` (or .md) and see, per clutter
//! regime x CFAR variant, the (nominal_pfa, observed_pfa, lower_ci,
//! upper_ci, ratio) row. If any nominal lies outside the 95% Wilson CI
//! of observed, the run fails.
//!
//! Pipeline:
//!  1. For each (regime, variant) pair, pre-generate a clutter-amplitude
//!     vector of length `n_cells` using `clutter::sample_clutter_amplitude`.
//!  2. Square to power domain (matches the convention used by
//!     `detectors::cfar_alpha`).
//!  3. Compute the CFAR alpha via `resolve_alpha` for the corresponding
//!     `NoiseDistribution`.
//!  4. Slide the CFAR window across the buffer (cell-under-test = the cell
//!     immediately after `training + guard` cells on the leading side, etc.).
//!     For each window, compute the variant-specific noise estimate, multiply
//!     by alpha, and compare to the CUT power. Count exceedances.
//!  5. Wrap the count in a 95% Wilson CI and compare to nominal Pfa.
//!
//! References:
//!   - Rohling, "Radar CFAR Thresholding in Clutter and Multiple
//!     Target Situations", IEEE Trans AES 1983 (CFAR variants).
//!   - Ward, Tough, Watts, *Sea Clutter*, IET 2013 (K-distribution
//!     calibration methodology).
//!   - Wilson, "Probable Inference, the Law of Succession, and
//!     Statistical Inference", JASA 1927 (binomial CI bounds).
//!
//! Strict-open posture: every reported value is a simulator-internal
//! Monte-Carlo measurement. No measured-truth claim is implied; the
//! calibrator's job is to verify that the alpha tables and CFAR estimators
//! in `crate::detectors::cfar_alpha` are internally self-consistent under
//! the synthetic distributions they were derived for.

use crate::clutter::{
    sample_clutter_amplitude, ClutterDistribution, ClutterRegime, TerrainClass,
};
use crate::detectors::cfar_alpha::{resolve_alpha, CfarVariant, NoiseDistribution};

#[path = "empirical_pfa_render.rs"]
mod empirical_pfa_render;
pub use empirical_pfa_render::{render_jsonl, render_markdown};
use empirical_pfa_render::{cfar_decision, layout_for};

/// One Pfa-trial request. The combination of `clutter_regime` + `cfar_variant`
/// + `training_cells` + `guard_cells` + `nominal_pfa` + `trials` + `seed` is
/// the deterministic input; the same tuple always yields the same
/// `PfaObservation`.
#[derive(Debug, Clone, Copy)]
pub struct PfaTrial {
    /// Clutter regime used to draw target-free cell amplitudes.
    pub clutter_regime: ClutterRegime,
    /// CFAR variant under test.
    pub cfar_variant: CfarVariant,
    /// Total training cells (`N`) used by the CFAR estimator; split half/half
    /// across leading / lagging windows for CA / GO / SO variants.
    pub training_cells: usize,
    /// Guard cells (each side of the CUT) excluded from the training window.
    pub guard_cells: usize,
    /// Nominal Pfa the threshold was designed for.
    pub nominal_pfa: f64,
    /// Number of CFAR decisions to run. Each one consumes one CUT plus the
    /// surrounding window; the underlying clutter buffer is sized to provide
    /// `trials` complete decisions.
    pub trials: u64,
    /// Master seed; per-cell amplitudes use `seed ^ cell_idx` so the same
    /// `(regime, seed)` pair produces the same global clutter realisation.
    pub seed: u64,
}

/// One Pfa-trial observation. `observed_pfa = observed_count / trials`, and
/// `passes` is `nominal_pfa in [wilson_ci_low, wilson_ci_high]`.
#[derive(Debug, Clone)]
pub struct PfaObservation {
    /// Human-readable terrain label (e.g. "Sea", "Urban").
    pub clutter_regime_name: String,
    /// The amplitude distribution used (echoed from the regime).
    pub clutter_distribution: ClutterDistribution,
    /// CFAR variant under test (echoed from the request).
    pub cfar_variant: CfarVariant,
    /// Training cells used (echoed from the request).
    pub training_cells: usize,
    /// Guard cells used (echoed from the request).
    pub guard_cells: usize,
    /// Nominal Pfa the threshold was designed for (echoed from the request).
    pub nominal_pfa: f64,
    /// Number of CUT samples that exceeded the CFAR threshold.
    pub observed_count: u64,
    /// Total CFAR decisions evaluated.
    pub trials: u64,
    /// `observed_count / trials`.
    pub observed_pfa: f64,
    /// Wilson 95% CI lower bound for the observed proportion.
    pub wilson_ci_low: f64,
    /// Wilson 95% CI upper bound for the observed proportion.
    pub wilson_ci_high: f64,
    /// `nominal_pfa in [wilson_ci_low, wilson_ci_high]`.
    pub passes: bool,
    /// `observed_pfa / nominal_pfa`. `f64::NAN` when `nominal_pfa == 0.0`.
    pub ratio_observed_to_nominal: f64,
}

/// Wilson 95% confidence interval for a binomial proportion.
///
/// Returns `(lower, upper)`. For `trials == 0` returns `(0.0, 1.0)` (maximally
/// uninformative). The Wilson interval is preferred over the normal-
/// approximation interval when the observed proportion is near 0 or 1; see
/// Wilson 1927.
pub fn wilson_ci_95(successes: u64, trials: u64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }
    // z for two-sided 95% CI.
    const Z: f64 = 1.959_963_984_540_054;
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = Z * Z;
    let denom = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denom;
    let half_width = (Z / denom) * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)).sqrt();
    let lo = (centre - half_width).max(0.0);
    let hi = (centre + half_width).min(1.0);
    (lo, hi)
}

/// Map a `ClutterDistribution` to the `NoiseDistribution` that the CFAR
/// alpha resolver expects. The two enums are intentionally separate (see
/// `cfar_alpha::NoiseDistribution` doc comment); this function is the only
/// place that bridges them.
fn clutter_to_noise(distribution: ClutterDistribution) -> NoiseDistribution {
    match distribution {
        ClutterDistribution::Rayleigh => NoiseDistribution::Rayleigh,
        ClutterDistribution::Weibull { shape, .. } => NoiseDistribution::Weibull {
            shape: shape as f32,
        },
        ClutterDistribution::KDistribution { shape, .. } => NoiseDistribution::KDistribution {
            shape: shape as f32,
        },
        // `cfar_alpha::NoiseDistribution::LogNormal` is parameterised by the
        // standard deviation of the underlying normal and assumes mean = 0;
        // the median is therefore 1. `ClutterDistribution::LogNormal` carries
        // both mean_log and std_log. The alpha library was calibrated for the
        // mean=0 family; we pass `std_log` through and document the
        // implication in the receipt.
        ClutterDistribution::LogNormal { std_log, .. } => NoiseDistribution::LogNormal {
            sigma: std_log as f32,
        },
    }
}

/// Human label for a terrain class (used in the `clutter_regime_name`
/// field of `PfaObservation`).
fn terrain_label(terrain: TerrainClass) -> &'static str {
    match terrain {
        TerrainClass::OpenSky => "OpenSky",
        TerrainClass::Desert => "Desert",
        TerrainClass::Forest => "Forest",
        TerrainClass::Urban => "Urban",
        TerrainClass::Sea => "Sea",
        TerrainClass::Mountain => "Mountain",
        TerrainClass::Agricultural => "Agricultural",
        TerrainClass::Suburban => "Suburban",
        TerrainClass::CoastalSea => "CoastalSea",
    }
}

/// Run one `(regime, variant)` Pfa trial; returns the observation.
///
/// Implementation: pre-allocates a `trials + window_len - 1` element power
/// buffer (so we get exactly `trials` complete CFAR windows), draws each cell
/// from the regime's amplitude distribution using a per-cell-derived seed,
/// then slides the CFAR window across counting exceedances.
pub fn measure_pfa(trial: &PfaTrial) -> PfaObservation {
    let layout = layout_for(trial);
    let trials = trial.trials.max(1);
    let buffer_len = trials as usize + layout.window_len - 1;

    // Pre-generate the power buffer. We square each amplitude to convert to
    // power; CFAR thresholds in `cfar_alpha` are defined in the power domain.
    // Per-cell seed = `seed ^ (cell_idx * 0x9e37_79b9_7f4a_7c15)` so adjacent
    // cells use uncorrelated SplitMix64 streams.
    let mut buffer: Vec<f64> = Vec::with_capacity(buffer_len);
    let seed = trial.seed;
    for i in 0..buffer_len {
        let s = seed ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let amp = sample_clutter_amplitude(&trial.clutter_regime.distribution, s);
        buffer.push(amp * amp);
    }

    // Resolve alpha for this (variant, distribution, N, pfa). cfar_alpha owns
    // the closed-form fast path for Gaussian/Rayleigh and the library lookup
    // / Monte-Carlo recovery for K/Weibull/log-normal.
    let noise = clutter_to_noise(trial.clutter_regime.distribution);
    let alpha = resolve_alpha(
        trial.cfar_variant,
        noise,
        trial.training_cells,
        trial.nominal_pfa as f32,
    ) as f64;

    // Slide the CFAR window. For each CUT, the lead window sits to the LEFT
    // and the lag window sits to the RIGHT, separated from the CUT by
    // `guard_cells` cells on each side.
    let mut exceedances: u64 = 0;
    for cut_start in 0..trials as usize {
        let lead_start = cut_start;
        let lead_end = lead_start + layout.half_train;
        let guard_left_end = lead_end + trial.guard_cells;
        let cut_idx = guard_left_end;
        let guard_right_end = cut_idx + 1 + trial.guard_cells;
        let lag_start = guard_right_end;
        let lag_end = lag_start + layout.half_train;

        let lead = &buffer[lead_start..lead_end];
        let lag = &buffer[lag_start..lag_end];
        let cut = buffer[cut_idx];

        if cfar_decision(trial.cfar_variant, cut, lead, lag, alpha) {
            exceedances += 1;
        }
    }

    let observed_pfa = exceedances as f64 / trials as f64;
    let (wilson_lo, wilson_hi) = wilson_ci_95(exceedances, trials);
    let passes = trial.nominal_pfa >= wilson_lo && trial.nominal_pfa <= wilson_hi;
    let ratio = if trial.nominal_pfa > 0.0 {
        observed_pfa / trial.nominal_pfa
    } else {
        f64::NAN
    };

    PfaObservation {
        clutter_regime_name: terrain_label(trial.clutter_regime.terrain).to_string(),
        clutter_distribution: trial.clutter_regime.distribution,
        cfar_variant: trial.cfar_variant,
        training_cells: trial.training_cells,
        guard_cells: trial.guard_cells,
        nominal_pfa: trial.nominal_pfa,
        observed_count: exceedances,
        trials,
        observed_pfa,
        wilson_ci_low: wilson_lo,
        wilson_ci_high: wilson_hi,
        passes,
        ratio_observed_to_nominal: ratio,
    }
}

/// The standard `(ClutterRegime, CfarVariant)` table. Eight regimes x four
/// variants = thirty-two combos. Training / guard / nominal Pfa picked to be
/// representative of the literature defaults; OS rank is 3N/4 per Rohling.
///
/// The Wave 4.5 H1 `TerrainClass::CoastalSea` sea-spray regimes are
/// intentionally NOT in the baseline calibration table — they ship with
/// their own Pfa gates (see `tests/physics_correctness.rs` Wave 4.5 H1
/// section). Including them here would silently rebaseline the C12
/// receipt and obscure regressions on the original 8-regime surface.
fn standard_table() -> Vec<(ClutterRegime, CfarVariant, usize, usize, f64)> {
    let regimes: Vec<ClutterRegime> = ClutterRegime::library()
        .into_iter()
        .filter(|r| r.terrain != TerrainClass::CoastalSea)
        .collect();
    let mut combos: Vec<(ClutterRegime, CfarVariant, usize, usize, f64)> = Vec::new();
    let training_cells: usize = 24;
    let guard_cells: usize = 4;
    // Use a relatively generous Pfa so that 10^7 trials sees ~10^4 false
    // alarms on average (enough for tight Wilson CIs). 1e-3 is the textbook
    // CFAR design value used by Rohling 1983 and Skolnik chap. 7.
    let nominal_pfa: f64 = 1e-3;
    let rank = (3 * training_cells) / 4; // 18 of 24
    let variants = [
        CfarVariant::CellAveraging,
        CfarVariant::OrderedStatistic { rank },
        CfarVariant::GreatestOf,
        CfarVariant::SmallestOf,
    ];
    for regime in regimes {
        for variant in variants {
            combos.push((regime, variant, training_cells, guard_cells, nominal_pfa));
        }
    }
    combos
}

/// Calibrate the standard 8 `ClutterRegime` x 4 `CfarVariant` table. Default
/// `trials_per_combo` is recommended at 1e7 (the C12 production gate);
/// callers may pass smaller values for smoke runs. Returns a Vec of 32
/// observations.
///
/// Seed scheme: each combo gets `seed ^ (combo_idx * 0x12345_67890_ABCDEF)`
/// so combos are seeded independently.
pub fn calibrate_standard_table(trials_per_combo: u64, seed: u64) -> Vec<PfaObservation> {
    let table = standard_table();
    let mut observations = Vec::with_capacity(table.len());
    for (idx, (regime, variant, training, guard, pfa)) in table.into_iter().enumerate() {
        let combo_seed = seed ^ (idx as u64).wrapping_mul(0x1234_5678_90AB_CDEF);
        let trial = PfaTrial {
            clutter_regime: regime,
            cfar_variant: variant,
            training_cells: training,
            guard_cells: guard,
            nominal_pfa: pfa,
            trials: trials_per_combo,
            seed: combo_seed,
        };
        observations.push(measure_pfa(&trial));
    }
    observations
}

#[cfg(test)]
#[path = "empirical_pfa_tests.rs"]
mod tests;
