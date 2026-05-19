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

/// Window layout used during the CFAR slide.
struct WindowLayout {
    /// Half of the training cells (lead side and lag side each get this many).
    half_train: usize,
    /// Total cells consumed by one CFAR decision (`2*half_train + 2*guard + 1`).
    window_len: usize,
}

fn layout_for(trial: &PfaTrial) -> WindowLayout {
    // Round odd training counts up so each side gets a full half. The slide
    // produces `total_cells - window_len + 1` decisions when the buffer length
    // is exactly that; we size the buffer to give `trials` decisions.
    let half_train = trial.training_cells.div_ceil(2);
    let window_len = 2 * half_train + 2 * trial.guard_cells + 1;
    WindowLayout {
        half_train,
        window_len,
    }
}

/// One CFAR decision. Returns `true` if the CUT power exceeds `alpha * stat`.
fn cfar_decision(
    variant: CfarVariant,
    cut: f64,
    lead: &[f64],
    lag: &[f64],
    alpha: f64,
) -> bool {
    let n = lead.len() + lag.len();
    if n == 0 {
        return false;
    }
    let stat = match variant {
        CfarVariant::CellAveraging => {
            let sum: f64 = lead.iter().chain(lag.iter()).sum();
            sum / n as f64
        }
        CfarVariant::GreatestOf => {
            let lead_mean = if lead.is_empty() {
                0.0
            } else {
                lead.iter().sum::<f64>() / lead.len() as f64
            };
            let lag_mean = if lag.is_empty() {
                0.0
            } else {
                lag.iter().sum::<f64>() / lag.len() as f64
            };
            lead_mean.max(lag_mean)
        }
        CfarVariant::SmallestOf => {
            let lead_mean = if lead.is_empty() {
                f64::INFINITY
            } else {
                lead.iter().sum::<f64>() / lead.len() as f64
            };
            let lag_mean = if lag.is_empty() {
                f64::INFINITY
            } else {
                lag.iter().sum::<f64>() / lag.len() as f64
            };
            lead_mean.min(lag_mean)
        }
        CfarVariant::OrderedStatistic { rank } => {
            // Sort the combined training window in ascending order; take the
            // `rank`-th sample (1-indexed). Rohling 1983 recommends k = 3N/4.
            let mut combined: Vec<f64> = Vec::with_capacity(n);
            combined.extend_from_slice(lead);
            combined.extend_from_slice(lag);
            // Use partial sort for the k-th order statistic. We pull a stable
            // O(N log N) sort here because N is small (<= ~32) in practice and
            // the simpler code dominates the per-window cost.
            combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = rank.clamp(1, n) - 1;
            combined[idx]
        }
    };
    cut > alpha * stat
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

/// Render observations to a Markdown table for receipts. Columns:
/// regime | distribution | variant | N | guard | nominal | observed | CI | ratio | passes.
pub fn render_markdown(observations: &[PfaObservation]) -> String {
    let mut out = String::new();
    out.push_str(
        "| Regime | Distribution | Variant | N | Guard | Nominal Pfa | Observed Pfa | 95% CI | Ratio | Pass |\n",
    );
    out.push_str(
        "|--------|--------------|---------|---|-------|-------------|--------------|--------|-------|------|\n",
    );
    for obs in observations {
        let dist_str = match obs.clutter_distribution {
            ClutterDistribution::Rayleigh => "Rayleigh".to_string(),
            ClutterDistribution::Weibull { shape, scale } => {
                format!("Weibull(c={shape:.2}, s={scale:.2})")
            }
            ClutterDistribution::KDistribution { shape, scale } => {
                format!("K(nu={shape:.2}, s={scale:.2})")
            }
            ClutterDistribution::LogNormal { mean_log, std_log } => {
                format!("LogN(mu={mean_log:.2}, sig={std_log:.2})")
            }
        };
        let var_str = match obs.cfar_variant {
            CfarVariant::CellAveraging => "CA".to_string(),
            CfarVariant::OrderedStatistic { rank } => format!("OS(k={rank})"),
            CfarVariant::GreatestOf => "GO".to_string(),
            CfarVariant::SmallestOf => "SO".to_string(),
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.2e} | {:.2e} | [{:.2e}, {:.2e}] | {:.2}x | {} |\n",
            obs.clutter_regime_name,
            dist_str,
            var_str,
            obs.training_cells,
            obs.guard_cells,
            obs.nominal_pfa,
            obs.observed_pfa,
            obs.wilson_ci_low,
            obs.wilson_ci_high,
            obs.ratio_observed_to_nominal,
            if obs.passes { "PASS" } else { "FAIL" },
        ));
    }
    out
}

/// Render observations to a JSON Lines blob (one row per observation).
/// Each row is a single-line JSON object with explicit primitive fields so
/// downstream tooling can parse with serde / jq / pandas without a custom
/// schema crate.
pub fn render_jsonl(observations: &[PfaObservation]) -> String {
    let mut out = String::new();
    for obs in observations {
        let (dist_kind, dist_param_a, dist_param_b) = match obs.clutter_distribution {
            ClutterDistribution::Rayleigh => ("rayleigh", f64::NAN, f64::NAN),
            ClutterDistribution::Weibull { shape, scale } => ("weibull", shape, scale),
            ClutterDistribution::KDistribution { shape, scale } => ("k", shape, scale),
            ClutterDistribution::LogNormal { mean_log, std_log } => {
                ("lognormal", mean_log, std_log)
            }
        };
        let (var_kind, var_rank) = match obs.cfar_variant {
            CfarVariant::CellAveraging => ("ca", 0usize),
            CfarVariant::OrderedStatistic { rank } => ("os", rank),
            CfarVariant::GreatestOf => ("go", 0),
            CfarVariant::SmallestOf => ("so", 0),
        };
        // Hand-written JSON to avoid pulling in `serde_json` from the radar
        // crate. Fields are emitted as JSON numbers (so `null` for NaN, to
        // keep strict parsers happy).
        let f = |v: f64| -> String {
            if v.is_finite() {
                format!("{v}")
            } else {
                "null".to_string()
            }
        };
        out.push_str(&format!(
            r#"{{"regime":"{}","distribution_kind":"{}","distribution_param_a":{},"distribution_param_b":{},"variant_kind":"{}","variant_rank":{},"training_cells":{},"guard_cells":{},"nominal_pfa":{},"observed_count":{},"trials":{},"observed_pfa":{},"wilson_ci_low":{},"wilson_ci_high":{},"ratio_observed_to_nominal":{},"passes":{}}}"#,
            obs.clutter_regime_name,
            dist_kind,
            f(dist_param_a),
            f(dist_param_b),
            var_kind,
            var_rank,
            obs.training_cells,
            obs.guard_cells,
            f(obs.nominal_pfa),
            obs.observed_count,
            obs.trials,
            f(obs.observed_pfa),
            f(obs.wilson_ci_low),
            f(obs.wilson_ci_high),
            f(obs.ratio_observed_to_nominal),
            obs.passes,
        ));
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_ci_centred_proportion() {
        // 500 / 1000 = 0.5 should give a CI roughly centred on 0.5 with
        // ~3% half-width.
        let (lo, hi) = wilson_ci_95(500, 1000);
        assert!(lo < 0.5 && hi > 0.5, "CI should bracket 0.5: [{lo}, {hi}]");
        assert!(hi - lo < 0.10, "CI half-width should be small: [{lo}, {hi}]");
    }

    #[test]
    fn wilson_ci_zero_trials_is_uninformative() {
        let (lo, hi) = wilson_ci_95(0, 0);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 1.0);
    }

    #[test]
    fn wilson_ci_zero_successes_lower_bound_zero() {
        // Wilson CI on (0, N) gives a centre + half-width that subtract to
        // ~ -eps due to floating-point round-off; we clamp the lower bound
        // to >= 0 in wilson_ci_95, so the returned value should be either
        // exactly 0.0 or a non-negative ~ eps value.
        let (lo, _hi) = wilson_ci_95(0, 1000);
        assert!(lo >= 0.0 && lo < 1e-12, "lower bound should clamp to ~0, got {lo}");
    }

    #[test]
    fn render_markdown_has_header_and_one_row_per_observation() {
        // Synthetic minimal observation set; just verify table structure.
        let observations = calibrate_standard_table(1_000, 0xABCD);
        let md = render_markdown(&observations);
        assert!(md.starts_with("| Regime"), "markdown should start with header");
        // 2 header rows + N data rows.
        let line_count = md.lines().count();
        assert_eq!(line_count, observations.len() + 2);
    }

    #[test]
    fn render_jsonl_one_line_per_observation() {
        let observations = calibrate_standard_table(1_000, 0xABCD);
        let jsonl = render_jsonl(&observations);
        let line_count = jsonl.lines().count();
        assert_eq!(line_count, observations.len());
    }

    #[test]
    fn standard_table_has_thirty_two_combos() {
        let table = standard_table();
        assert_eq!(table.len(), 32);
    }

    #[test]
    fn measure_pfa_deterministic_for_fixed_seed() {
        let regime = ClutterRegime::for_terrain(TerrainClass::OpenSky, 30.0);
        let trial = PfaTrial {
            clutter_regime: regime,
            cfar_variant: CfarVariant::CellAveraging,
            training_cells: 24,
            guard_cells: 4,
            nominal_pfa: 1e-2,
            trials: 10_000,
            seed: 0xDEAD_BEEF,
        };
        let a = measure_pfa(&trial);
        let b = measure_pfa(&trial);
        assert_eq!(a.observed_count, b.observed_count);
        assert_eq!(a.trials, b.trials);
    }
}
