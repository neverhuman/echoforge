//! V4 (measured-anchored) validation tier gate.
//!
//! Per `docs/validation-tiers.md`, V4 = "method calibrated or anchored
//! against lawful measured data, but not fully validated". The
//! string-alias for V4 is `measured_anchored` (V5 reserves the
//! `measured` alias for full lawful-measured validation).
//!
//! This module ships the GATE LOGIC: given a set of distribution
//! anchors (citations to public radar / UAS / bird datasets) and a
//! set of observed distributions extracted from simulator output, the
//! gate computes a 1-D distribution distance per anchor (Wasserstein-1
//! or Kolmogorov-Smirnov) and decides whether every anchor passes the
//! tolerance band. Anchors that fail are listed by feature name in the
//! report.
//!
//! Strict-open posture: V4 accepts **distribution targets and
//! citations only**. The simulator's output histograms are compared
//! against published distribution shapes (Nature 2026 multi-sensor
//! drone dataset, Rahman-Robertson K/W-band drone+bird, Karlsson 77 GHz
//! FMCW Zenodo dataset, VTT 15/25 GHz fixed-wing UAV RCS). No measured
//! traces enter the repo. No measured-truth claims — that is V5,
//! reserved.
//!
//! Per the FUCKIT.md.done cross-tip resolution that separates
//! method-ceiling (fidelity F0–F5) from evidence-tier (validation
//! V0–V5), the V4 gate is purely an evidence-tier evaluator and does
//! NOT interact with the `fidelity_class` axis. The V0/V1/V2 gate in
//! [`crate::report`] and the V3 gate in [`crate::tier_benchmarked`] are both
//! left untouched — V4 is additive.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::distribution_metrics::{ks_distance_1d_sorted, wasserstein_1d_sorted};
use crate::error::ValidateError;
use crate::tier_benchmarked::GateStatus;

/// Distance metric used to compare an observed distribution against a
/// distribution anchor.
///
/// Both metrics are 1-D and operate on sorted samples. Wasserstein-1
/// (a.k.a. Earth Mover's Distance) is sensitive to *where* mass moves;
/// Kolmogorov-Smirnov reports the supremum absolute difference of the
/// empirical CDFs. KS is preferred when the comparison is "do these
/// two distributions plausibly share a parent"; Wasserstein when "are
/// the simulator outputs *quantitatively* close to the anchor".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnchorMetric {
    #[serde(rename = "wasserstein_1")]
    Wasserstein1,
    #[serde(rename = "kolmogorov_smirnov")]
    KolmogorovSmirnov,
}

impl AnchorMetric {
    pub fn as_str(&self) -> &'static str {
        match self {
            AnchorMetric::Wasserstein1 => "wasserstein_1",
            AnchorMetric::KolmogorovSmirnov => "kolmogorov_smirnov",
        }
    }
}

/// A distribution anchor: a citation-backed target distribution for a
/// named feature, plus the metric and tolerance the gate should use.
///
/// `target_samples` are the anchor's sample values (e.g. the SNR values
/// observed in a published drone-radar measurement campaign). The
/// gate computes the chosen metric between `target_samples` and the
/// caller's `observed_samples`. The simulator output never leaves the
/// caller's process; only the metric value is recorded in the gate
/// report.
///
/// `url` and `citation` are stored verbatim in the report so the
/// resulting `calibration_report.json` is fully self-describing for
/// downstream auditors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V4DistributionAnchor {
    /// Stable feature name (e.g. `"snr_db"`, `"micro_doppler_peak_hz"`).
    pub feature_name: String,
    /// Human-readable citation for the anchor source (e.g.
    /// `"Nature 2026 multi-sensor drone dataset (Sci Data)"`).
    pub citation: String,
    /// Optional URL or DOI for the citation. Strict-open posture:
    /// link only; do not vendor the dataset content.
    pub url: Option<String>,
    /// Sample values that define the anchor distribution. Sorted by
    /// the gate before metric computation; the caller need not
    /// pre-sort.
    pub target_samples: Vec<f64>,
    /// Distance metric to apply.
    pub metric: AnchorMetric,
    /// Tolerance band the observed-vs-target distance must satisfy
    /// (inclusive). For Wasserstein-1 this is in the feature's native
    /// unit (e.g. dB for `snr_db`); for KS the units are dimensionless
    /// in `[0, 1]`.
    pub tolerance: f64,
}

/// Pre-computed observation fed to the V4 gate: the simulator's
/// samples for the same feature the anchor describes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V4DistributionObservation {
    /// Must match the anchor's `feature_name`.
    pub feature_name: String,
    /// Simulator-produced samples for the feature.
    pub observed_samples: Vec<f64>,
}

/// Per-anchor evaluation result captured in the V4 gate report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V4AnchorOutcome {
    pub feature_name: String,
    pub citation: String,
    pub url: Option<String>,
    pub metric: AnchorMetric,
    pub tolerance: f64,
    pub distance: f64,
    pub anchor_sample_count: usize,
    pub observed_sample_count: usize,
    pub status: GateStatus,
}

/// Threshold envelope for the V4 gate.
///
/// `min_anchors` is the minimum number of distribution anchors that
/// must be supplied for the gate to even run; the historical roadmap
/// guidance is at least 3, so that a single bad anchor cannot dominate.
/// `min_samples_per_distribution` rejects under-powered distributions
/// (a 5-sample anchor will not survive bootstrap CIs in any case).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V4MetricThresholds {
    pub min_anchors: usize,
    pub min_samples_per_distribution: usize,
}

impl V4MetricThresholds {
    /// Conservative defaults: at least 3 anchors, at least 32 samples
    /// per distribution on either side. The per-anchor tolerance lives
    /// on the [`V4DistributionAnchor`] itself, intentionally — each
    /// anchor is its own physics and its own published reference, so a
    /// one-size-fits-all distance cap would be wrong.
    pub fn default_roadmap() -> Self {
        Self {
            min_anchors: 3,
            min_samples_per_distribution: 32,
        }
    }
}

/// Gate report — what the V4 gate produces.
///
/// `tier` is the symbolic label ("V4" or "measured_anchored") that
/// downstream consumers may surface. `status` is the boolean-ish gate
/// outcome. `failures` enumerates the feature names (one per anchor)
/// whose observed-vs-target distance exceeded the tolerance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V4GateReport {
    pub tier: &'static str,
    pub status: GateStatus,
    pub anchor_count: usize,
    pub failures: Vec<String>,
    pub outcomes: Vec<V4AnchorOutcome>,
    pub thresholds: V4MetricThresholds,
}

/// Convenience constant: the schema-alias string for V4.
pub const fn tier_alias_measured_anchored() -> &'static str {
    "measured_anchored"
}

fn compute_distance(anchor: &V4DistributionAnchor, observed: &[f64]) -> f64 {
    let mut a = anchor.target_samples.clone();
    let mut b = observed.to_vec();
    a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    match anchor.metric {
        AnchorMetric::Wasserstein1 => wasserstein_1d_sorted(&a, &b),
        AnchorMetric::KolmogorovSmirnov => ks_distance_1d_sorted(&a, &b),
    }
}

fn anchor_outcome(
    anchor: &V4DistributionAnchor,
    observed_sample_count: usize,
    distance: f64,
    status: GateStatus,
) -> V4AnchorOutcome {
    V4AnchorOutcome {
        feature_name: anchor.feature_name.clone(),
        citation: anchor.citation.clone(),
        url: anchor.url.clone(),
        metric: anchor.metric,
        tolerance: anchor.tolerance,
        distance,
        anchor_sample_count: anchor.target_samples.len(),
        observed_sample_count,
        status,
    }
}

/// Evaluate the V4 promotion gate.
///
/// `anchors` and `observations` are paired by `feature_name`. Every
/// anchor must have a matching observation; an unmatched anchor counts
/// as a failure with `distance = f64::INFINITY`.
///
/// The gate's status is `Pass` when (a) the anchor count meets
/// `thresholds.min_anchors`, (b) every anchor's distribution has at
/// least `thresholds.min_samples_per_distribution` samples on both
/// sides, and (c) every per-anchor distance is `<= tolerance`. Any
/// violation -> `Fail` and the violating feature names appear in
/// `failures`. The gate never returns `Warn`.
pub fn evaluate_v4_gate(
    thresholds: &V4MetricThresholds,
    anchors: &[V4DistributionAnchor],
    observations: &[V4DistributionObservation],
) -> V4GateReport {
    let mut outcomes = Vec::with_capacity(anchors.len());
    let mut failures = Vec::new();

    for anchor in anchors {
        let observed = observations
            .iter()
            .find(|o| o.feature_name == anchor.feature_name);

        let outcome = match observed {
            Some(obs) => {
                let underpowered_anchor =
                    anchor.target_samples.len() < thresholds.min_samples_per_distribution;
                let underpowered_obs =
                    obs.observed_samples.len() < thresholds.min_samples_per_distribution;
                if underpowered_anchor || underpowered_obs {
                    anchor_outcome(anchor, obs.observed_samples.len(), f64::INFINITY, GateStatus::Fail)
                } else {
                    let distance = compute_distance(anchor, &obs.observed_samples);
                    let status = if distance <= anchor.tolerance && distance.is_finite() {
                        GateStatus::Pass
                    } else {
                        GateStatus::Fail
                    };
                    anchor_outcome(anchor, obs.observed_samples.len(), distance, status)
                }
            }
            None => anchor_outcome(anchor, 0, f64::INFINITY, GateStatus::Fail),
        };
        if outcome.status == GateStatus::Fail {
            failures.push(anchor.feature_name.clone());
        }
        outcomes.push(outcome);
    }

    let too_few_anchors = anchors.len() < thresholds.min_anchors;
    let status = if too_few_anchors || !failures.is_empty() {
        GateStatus::Fail
    } else {
        GateStatus::Pass
    };
    if too_few_anchors && !failures.iter().any(|f| f == "__too_few_anchors__") {
        failures.insert(0, "__too_few_anchors__".to_string());
    }

    V4GateReport {
        tier: "V4",
        status,
        anchor_count: anchors.len(),
        failures,
        outcomes,
        thresholds: thresholds.clone(),
    }
}

/// On-disk schema for the V4 calibration report consumed by
/// [`evaluate_v4_from_calibration_report`].
#[derive(Debug, Deserialize, Serialize)]
pub struct CalibrationReportFile {
    pub anchors: Vec<V4DistributionAnchor>,
    pub observations: Vec<V4DistributionObservation>,
}

/// Read a calibration report from disk and evaluate the V4 gate
/// against the supplied thresholds.
///
/// The file is expected to be JSON of the shape:
/// `{ "anchors": [...], "observations": [...] }`. Returns
/// [`ValidateError::Schema`] for malformed JSON and
/// [`ValidateError::Io`] for filesystem errors.
pub fn evaluate_v4_from_calibration_report(
    thresholds: &V4MetricThresholds,
    calibration_path: &Path,
) -> Result<V4GateReport, ValidateError> {
    let raw = fs::read_to_string(calibration_path)?;
    let parsed: CalibrationReportFile = serde_json::from_str(&raw)
        .map_err(|e| ValidateError::Schema(format!("{}: {}", calibration_path.display(), e)))?;
    Ok(evaluate_v4_gate(
        thresholds,
        &parsed.anchors,
        &parsed.observations,
    ))
}

#[cfg(test)]
#[path = "tier_measured_anchored_tests_a.rs"]
mod tests_a;

#[cfg(test)]
#[path = "tier_measured_anchored_tests_b.rs"]
mod tests_b;
