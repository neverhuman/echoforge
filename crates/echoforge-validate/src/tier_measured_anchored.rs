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

/// 1-D Wasserstein-1 (Earth Mover's) distance between two empirical
/// distributions given as sorted sample vectors.
///
/// Standard rectangular-CDF formulation: for two ECDFs `F` and `G`,
/// `W_1(F, G) = ∫ |F(x) - G(x)| dx`. The implementation here uses the
/// equivalent merged-quantile form which is `O(n log n)` rather than
/// requiring an integration grid.
///
/// Both inputs MUST be sorted ascending. Empty inputs return `f64::INFINITY`
/// (treated as "no comparison possible"); the gate evaluator filters
/// these out via [`V4MetricThresholds::min_samples_per_distribution`].
pub fn wasserstein_1d_sorted(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }
    // Merge sorted quantile breakpoints across both samples, then
    // accumulate signed CDF differences times width.
    let na = a.len();
    let nb = b.len();
    let mut i = 0usize;
    let mut j = 0usize;
    let mut total = 0.0f64;
    let mut prev = if a[0] < b[0] { a[0] } else { b[0] };
    let mut cdf_a;
    let mut cdf_b;
    loop {
        let (next, advance_a, advance_b) = match (a.get(i), b.get(j)) {
            (Some(&av), Some(&bv)) => {
                if av < bv {
                    (av, true, false)
                } else if bv < av {
                    (bv, false, true)
                } else {
                    (av, true, true)
                }
            }
            (Some(&av), None) => (av, true, false),
            (None, Some(&bv)) => (bv, false, true),
            (None, None) => break,
        };
        let width = next - prev;
        cdf_a = i as f64 / na as f64;
        cdf_b = j as f64 / nb as f64;
        total += (cdf_a - cdf_b).abs() * width;
        if advance_a {
            i += 1;
        }
        if advance_b {
            j += 1;
        }
        prev = next;
    }
    total
}

/// 1-D Kolmogorov-Smirnov distance between two empirical
/// distributions given as sorted sample vectors. Returns the supremum
/// of `|F(x) - G(x)|` over all `x` in the merged sample set, which is
/// in `[0, 1]`.
///
/// Both inputs MUST be sorted ascending. Empty inputs return `1.0`
/// (maximum possible KS distance) so the gate's tolerance check fails
/// gracefully rather than silently accepting an empty observation.
pub fn ks_distance_1d_sorted(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }
    let na = a.len();
    let nb = b.len();
    let mut i = 0usize;
    let mut j = 0usize;
    let mut sup = 0.0f64;
    while i < na || j < nb {
        let (advance_a, advance_b) = match (a.get(i), b.get(j)) {
            (Some(&av), Some(&bv)) => {
                if av < bv {
                    (true, false)
                } else if bv < av {
                    (false, true)
                } else {
                    (true, true)
                }
            }
            (Some(_), None) => (true, false),
            (None, Some(_)) => (false, true),
            (None, None) => break,
        };
        if advance_a {
            i += 1;
        }
        if advance_b {
            j += 1;
        }
        let cdf_a = i as f64 / na as f64;
        let cdf_b = j as f64 / nb as f64;
        let diff = (cdf_a - cdf_b).abs();
        if diff > sup {
            sup = diff;
        }
    }
    sup
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
                    V4AnchorOutcome {
                        feature_name: anchor.feature_name.clone(),
                        citation: anchor.citation.clone(),
                        url: anchor.url.clone(),
                        metric: anchor.metric,
                        tolerance: anchor.tolerance,
                        distance: f64::INFINITY,
                        anchor_sample_count: anchor.target_samples.len(),
                        observed_sample_count: obs.observed_samples.len(),
                        status: GateStatus::Fail,
                    }
                } else {
                    let distance = compute_distance(anchor, &obs.observed_samples);
                    let pass = distance <= anchor.tolerance && distance.is_finite();
                    V4AnchorOutcome {
                        feature_name: anchor.feature_name.clone(),
                        citation: anchor.citation.clone(),
                        url: anchor.url.clone(),
                        metric: anchor.metric,
                        tolerance: anchor.tolerance,
                        distance,
                        anchor_sample_count: anchor.target_samples.len(),
                        observed_sample_count: obs.observed_samples.len(),
                        status: if pass { GateStatus::Pass } else { GateStatus::Fail },
                    }
                }
            }
            None => V4AnchorOutcome {
                feature_name: anchor.feature_name.clone(),
                citation: anchor.citation.clone(),
                url: anchor.url.clone(),
                metric: anchor.metric,
                tolerance: anchor.tolerance,
                distance: f64::INFINITY,
                anchor_sample_count: anchor.target_samples.len(),
                observed_sample_count: 0,
                status: GateStatus::Fail,
            },
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
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn baseline_thresholds() -> V4MetricThresholds {
        V4MetricThresholds {
            min_anchors: 3,
            min_samples_per_distribution: 4,
        }
    }

    fn anchor(
        feature: &str,
        metric: AnchorMetric,
        tolerance: f64,
        samples: Vec<f64>,
    ) -> V4DistributionAnchor {
        V4DistributionAnchor {
            feature_name: feature.to_string(),
            citation: format!("test-anchor::{feature}"),
            url: None,
            target_samples: samples,
            metric,
            tolerance,
        }
    }

    fn observation(feature: &str, samples: Vec<f64>) -> V4DistributionObservation {
        V4DistributionObservation {
            feature_name: feature.to_string(),
            observed_samples: samples,
        }
    }

    #[test]
    fn wasserstein_zero_for_identical_distributions() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let d = wasserstein_1d_sorted(&a, &b);
        assert!(d.abs() < 1e-12, "expected 0, got {}", d);
    }

    #[test]
    fn wasserstein_shift_equals_translation() {
        // Translating every sample by +1.0 should give W_1 = 1.0.
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = wasserstein_1d_sorted(&a, &b);
        assert!((d - 1.0).abs() < 1e-9, "expected ~1.0, got {}", d);
    }

    #[test]
    fn wasserstein_handles_empty() {
        let a: Vec<f64> = vec![];
        let b = vec![1.0, 2.0];
        assert_eq!(wasserstein_1d_sorted(&a, &b), f64::INFINITY);
    }

    #[test]
    fn ks_zero_for_identical_distributions() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        assert!(ks_distance_1d_sorted(&a, &b).abs() < 1e-12);
    }

    #[test]
    fn ks_one_for_disjoint_distributions() {
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![10.0, 10.0, 10.0, 10.0];
        let d = ks_distance_1d_sorted(&a, &b);
        assert!((d - 1.0).abs() < 1e-9, "expected ~1.0, got {}", d);
    }

    #[test]
    fn ks_handles_empty() {
        let a: Vec<f64> = vec![];
        let b = vec![1.0, 2.0];
        assert_eq!(ks_distance_1d_sorted(&a, &b), 1.0);
    }

    #[test]
    fn evaluate_passes_when_all_distances_within_tolerance() {
        let anchors = vec![
            anchor(
                "snr_db",
                AnchorMetric::Wasserstein1,
                2.0,
                vec![5.0, 6.0, 7.0, 8.0, 9.0],
            ),
            anchor(
                "micro_doppler_peak_hz",
                AnchorMetric::Wasserstein1,
                10.0,
                vec![60.0, 70.0, 80.0, 90.0, 100.0],
            ),
            anchor(
                "altitude_m",
                AnchorMetric::KolmogorovSmirnov,
                0.5,
                vec![100.0, 200.0, 300.0, 400.0, 500.0],
            ),
        ];
        let observations = vec![
            observation("snr_db", vec![5.5, 6.5, 7.5, 8.5, 9.5]),
            observation(
                "micro_doppler_peak_hz",
                vec![62.0, 72.0, 82.0, 92.0, 102.0],
            ),
            observation("altitude_m", vec![110.0, 210.0, 310.0, 410.0, 510.0]),
        ];
        let report = evaluate_v4_gate(&baseline_thresholds(), &anchors, &observations);
        assert_eq!(report.status, GateStatus::Pass);
        assert!(report.failures.is_empty(), "{:?}", report.failures);
        assert_eq!(report.tier, "V4");
        assert_eq!(report.anchor_count, 3);
    }

    #[test]
    fn evaluate_fails_when_any_anchor_exceeds_tolerance() {
        let anchors = vec![
            anchor(
                "snr_db",
                AnchorMetric::Wasserstein1,
                0.1, // tight tolerance, will fail
                vec![5.0, 6.0, 7.0, 8.0],
            ),
            anchor(
                "altitude_m",
                AnchorMetric::Wasserstein1,
                100.0,
                vec![100.0, 200.0, 300.0, 400.0],
            ),
            anchor(
                "range_m",
                AnchorMetric::KolmogorovSmirnov,
                0.5,
                vec![1000.0, 2000.0, 3000.0, 4000.0],
            ),
        ];
        let observations = vec![
            observation("snr_db", vec![10.0, 11.0, 12.0, 13.0]),
            observation("altitude_m", vec![110.0, 210.0, 310.0, 410.0]),
            observation("range_m", vec![1100.0, 2100.0, 3100.0, 4100.0]),
        ];
        let report = evaluate_v4_gate(&baseline_thresholds(), &anchors, &observations);
        assert_eq!(report.status, GateStatus::Fail);
        assert_eq!(report.failures, vec!["snr_db".to_string()]);
        // Other anchors still produce outcomes with Pass status.
        let altitude = report
            .outcomes
            .iter()
            .find(|o| o.feature_name == "altitude_m")
            .expect("altitude outcome present");
        assert_eq!(altitude.status, GateStatus::Pass);
    }

    #[test]
    fn evaluate_fails_when_too_few_anchors_supplied() {
        let anchors = vec![anchor(
            "snr_db",
            AnchorMetric::Wasserstein1,
            10.0,
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )];
        let observations = vec![observation("snr_db", vec![1.0, 2.0, 3.0, 4.0, 5.0])];
        let report = evaluate_v4_gate(&baseline_thresholds(), &anchors, &observations);
        assert_eq!(report.status, GateStatus::Fail);
        assert!(report.failures.contains(&"__too_few_anchors__".to_string()));
    }

    #[test]
    fn evaluate_fails_when_observation_is_missing() {
        let anchors = vec![
            anchor(
                "snr_db",
                AnchorMetric::Wasserstein1,
                10.0,
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
            ),
            anchor(
                "altitude_m",
                AnchorMetric::Wasserstein1,
                10.0,
                vec![100.0, 200.0, 300.0, 400.0, 500.0],
            ),
            anchor(
                "range_m",
                AnchorMetric::Wasserstein1,
                10.0,
                vec![1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            ),
        ];
        let observations = vec![
            observation("snr_db", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            // altitude_m and range_m intentionally missing
        ];
        let report = evaluate_v4_gate(&baseline_thresholds(), &anchors, &observations);
        assert_eq!(report.status, GateStatus::Fail);
        assert!(report.failures.contains(&"altitude_m".to_string()));
        assert!(report.failures.contains(&"range_m".to_string()));
        let missing = report
            .outcomes
            .iter()
            .find(|o| o.feature_name == "altitude_m")
            .expect("altitude outcome present");
        assert_eq!(missing.distance, f64::INFINITY);
        assert_eq!(missing.observed_sample_count, 0);
    }

    #[test]
    fn evaluate_fails_when_distribution_is_underpowered() {
        // min_samples_per_distribution = 4 in baseline_thresholds()
        let anchors = vec![
            anchor(
                "snr_db",
                AnchorMetric::Wasserstein1,
                10.0,
                vec![1.0, 2.0, 3.0, 4.0],
            ),
            anchor(
                "altitude_m",
                AnchorMetric::Wasserstein1,
                10.0,
                vec![100.0, 200.0], // underpowered: < min_samples
            ),
            anchor(
                "range_m",
                AnchorMetric::Wasserstein1,
                10.0,
                vec![1000.0, 2000.0, 3000.0, 4000.0],
            ),
        ];
        let observations = vec![
            observation("snr_db", vec![1.0, 2.0, 3.0, 4.0]),
            observation("altitude_m", vec![110.0, 210.0, 310.0, 410.0]),
            observation("range_m", vec![1000.0, 2000.0, 3000.0, 4000.0]),
        ];
        let report = evaluate_v4_gate(&baseline_thresholds(), &anchors, &observations);
        assert_eq!(report.status, GateStatus::Fail);
        assert!(report.failures.contains(&"altitude_m".to_string()));
        let underpowered = report
            .outcomes
            .iter()
            .find(|o| o.feature_name == "altitude_m")
            .expect("altitude outcome present");
        assert_eq!(underpowered.distance, f64::INFINITY);
        assert_eq!(underpowered.anchor_sample_count, 2);
    }

    #[test]
    fn tier_alias_string_is_measured_anchored() {
        assert_eq!(tier_alias_measured_anchored(), "measured_anchored");
    }

    #[test]
    fn anchor_metric_string_aliases_are_stable() {
        assert_eq!(AnchorMetric::Wasserstein1.as_str(), "wasserstein_1");
        assert_eq!(
            AnchorMetric::KolmogorovSmirnov.as_str(),
            "kolmogorov_smirnov"
        );
    }

    #[test]
    fn evaluate_from_calibration_report_round_trip() {
        let anchors = vec![
            anchor(
                "snr_db",
                AnchorMetric::Wasserstein1,
                2.0,
                vec![5.0, 6.0, 7.0, 8.0],
            ),
            anchor(
                "altitude_m",
                AnchorMetric::Wasserstein1,
                50.0,
                vec![100.0, 200.0, 300.0, 400.0],
            ),
            anchor(
                "range_m",
                AnchorMetric::KolmogorovSmirnov,
                0.5,
                vec![1000.0, 2000.0, 3000.0, 4000.0],
            ),
        ];
        let observations = vec![
            observation("snr_db", vec![5.5, 6.5, 7.5, 8.5]),
            observation("altitude_m", vec![110.0, 210.0, 310.0, 410.0]),
            observation("range_m", vec![1100.0, 2100.0, 3100.0, 4100.0]),
        ];
        let payload = CalibrationReportFile {
            anchors,
            observations,
        };
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", serde_json::to_string(&payload).unwrap()).unwrap();
        let report = evaluate_v4_from_calibration_report(&baseline_thresholds(), file.path())
            .expect("v4 evaluation succeeds");
        assert_eq!(report.status, GateStatus::Pass);
        assert_eq!(report.anchor_count, 3);
    }

    #[test]
    fn evaluate_from_calibration_report_surfaces_schema_errors() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{{ not valid json }}").unwrap();
        let err = evaluate_v4_from_calibration_report(&baseline_thresholds(), file.path())
            .expect_err("malformed json must fail");
        assert!(matches!(err, ValidateError::Schema(_)));
    }

    #[test]
    fn evaluate_from_calibration_report_surfaces_io_errors() {
        let err = evaluate_v4_from_calibration_report(
            &baseline_thresholds(),
            Path::new("/nonexistent/v4_calibration_report.json"),
        )
        .expect_err("missing file must fail");
        assert!(matches!(err, ValidateError::Io(_)));
    }

    #[test]
    fn v4_gate_report_serializes_with_stable_field_names() {
        // We don't round-trip through Deserialize for V4GateReport because
        // its `tier` field is `&'static str` (matching V3GateReport); serde
        // would need to borrow into a 'static lifetime. The serialized form
        // is what downstream consumers actually need — they read this JSON,
        // they don't deserialize it back into the Rust struct.
        let anchors = vec![
            anchor(
                "snr_db",
                AnchorMetric::Wasserstein1,
                2.0,
                vec![5.0, 6.0, 7.0, 8.0],
            ),
            anchor(
                "altitude_m",
                AnchorMetric::Wasserstein1,
                50.0,
                vec![100.0, 200.0, 300.0, 400.0],
            ),
            anchor(
                "range_m",
                AnchorMetric::KolmogorovSmirnov,
                0.5,
                vec![1000.0, 2000.0, 3000.0, 4000.0],
            ),
        ];
        let observations = vec![
            observation("snr_db", vec![5.5, 6.5, 7.5, 8.5]),
            observation("altitude_m", vec![110.0, 210.0, 310.0, 410.0]),
            observation("range_m", vec![1100.0, 2100.0, 3100.0, 4100.0]),
        ];
        let report = evaluate_v4_gate(&baseline_thresholds(), &anchors, &observations);
        let json: serde_json::Value =
            serde_json::to_value(&report).expect("serialize to value");
        assert_eq!(json["tier"], "V4");
        assert_eq!(json["status"], "pass");
        assert_eq!(json["anchor_count"], 3);
        assert!(json["outcomes"].as_array().unwrap().len() == 3);
        let first = &json["outcomes"][0];
        assert_eq!(first["feature_name"], "snr_db");
        assert_eq!(first["metric"], "wasserstein_1");
        assert_eq!(first["status"], "pass");
    }
}
