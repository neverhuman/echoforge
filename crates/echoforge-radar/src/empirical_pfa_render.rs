//! Rendering helpers for `empirical_pfa.rs` — extracted for LOC compliance.

use super::{PfaObservation, PfaTrial};
use crate::clutter::ClutterDistribution;
use crate::detectors::cfar_alpha::CfarVariant;

// ---------------------------------------------------------------------------
// CFAR window helpers (used by measure_pfa in empirical_pfa.rs)
// ---------------------------------------------------------------------------

/// Window layout used during the CFAR slide.
pub(super) struct WindowLayout {
    /// Half of the training cells (lead side and lag side each get this many).
    pub(super) half_train: usize,
    /// Total cells consumed by one CFAR decision (`2*half_train + 2*guard + 1`).
    pub(super) window_len: usize,
}

pub(super) fn layout_for(trial: &PfaTrial) -> WindowLayout {
    let half_train = trial.training_cells.div_ceil(2);
    let window_len = 2 * half_train + 2 * trial.guard_cells + 1;
    WindowLayout { half_train, window_len }
}

/// One CFAR decision. Returns `true` if the CUT power exceeds `alpha * stat`.
pub(super) fn cfar_decision(
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
            let lead_mean = if lead.is_empty() { 0.0 } else {
                lead.iter().sum::<f64>() / lead.len() as f64
            };
            let lag_mean = if lag.is_empty() { 0.0 } else {
                lag.iter().sum::<f64>() / lag.len() as f64
            };
            lead_mean.max(lag_mean)
        }
        CfarVariant::SmallestOf => {
            let lead_mean = if lead.is_empty() { f64::INFINITY } else {
                lead.iter().sum::<f64>() / lead.len() as f64
            };
            let lag_mean = if lag.is_empty() { f64::INFINITY } else {
                lag.iter().sum::<f64>() / lag.len() as f64
            };
            lead_mean.min(lag_mean)
        }
        CfarVariant::OrderedStatistic { rank } => {
            let mut combined: Vec<f64> = Vec::with_capacity(n);
            combined.extend_from_slice(lead);
            combined.extend_from_slice(lag);
            combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = rank.clamp(1, n) - 1;
            combined[idx]
        }
    };
    cut > alpha * stat
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
        // crate. Fields are emitted as JSON numbers (so `null` for NaN).
        let f = |v: f64| -> String {
            if v.is_finite() { format!("{v}") } else { "null".to_string() }
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
