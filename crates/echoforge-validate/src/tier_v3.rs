//! V3 (benchmarked) validation tier gate.
//!
//! Per `docs/validation-tiers.md`, V3 = "benchmark suite with stable
//! metrics (Pd/Pfa, range/Doppler error, OSPA on tracks)". The
//! string-alias for V3 in `schemas/common.schema.json#/$defs/validation`
//! is `benchmarked`.
//!
//! This module ships the GATE LOGIC: given a set of thresholds and a
//! pre-computed metric observation, decide whether the artifact passes
//! the V3 promotion gate. The upstream metrics producer (e.g. a live
//! benchmark campaign against a sagittasbr-computed RCS truth) is a
//! separate concern; this gate accepts any observation that conforms
//! to the [`V3MetricObservation`] shape, regardless of how it was
//! computed.
//!
//! The V0/V1/V2 promotion gate in [`crate::report`] is intentionally
//! left untouched — V3 is an additive evaluator that can be wired into
//! the CLI in a follow-up packet once the upstream metrics producer is
//! online.
//!
//! Per the FUCKIT.md.done cross-tip resolution, the V3 gate does NOT
//! interact with the `fidelity_class` (F0..F5) method-ceiling axis;
//! see `docs/validation-tiers.md` "Why the two axes are independent".

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ValidateError;

/// Outcome of a single V3 gate evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GateStatus {
    Pass,
    Warn,
    Fail,
}

impl GateStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            GateStatus::Pass => "pass",
            GateStatus::Warn => "warn",
            GateStatus::Fail => "fail",
        }
    }
}

/// Threshold envelope for V3 benchmark metrics.
///
/// Every observed metric must beat its threshold for the V3 gate to
/// pass. Reasonable starting values per the historical roadmap entry:
///
/// * `pd_min = 0.85` — minimum acceptable probability of detection.
/// * `pfa_max = 0.01` — maximum acceptable false-alarm rate.
/// * `range_error_max_m`, `doppler_error_max_mps`, `ospa_max` —
///   campaign-specific tolerances.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V3MetricThresholds {
    /// Minimum probability of detection (dimensionless, in `[0, 1]`).
    pub pd_min: f64,
    /// Maximum probability of false alarm (dimensionless, in `[0, 1]`).
    pub pfa_max: f64,
    /// Maximum tolerated range error in metres.
    pub range_error_max_m: f64,
    /// Maximum tolerated Doppler error in metres per second.
    pub doppler_error_max_mps: f64,
    /// Maximum tolerated OSPA distance for tracks.
    pub ospa_max: f64,
}

impl V3MetricThresholds {
    /// Conservative defaults aligned with the roadmap entry.
    pub fn default_roadmap() -> Self {
        Self {
            pd_min: 0.85,
            pfa_max: 0.01,
            range_error_max_m: 5.0,
            doppler_error_max_mps: 1.0,
            ospa_max: 10.0,
        }
    }
}

/// Pre-computed metric observation fed to the V3 gate.
///
/// All fields are in the same units as their corresponding threshold
/// in [`V3MetricThresholds`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V3MetricObservation {
    /// Observed probability of detection.
    pub pd: f64,
    /// Observed probability of false alarm.
    pub pfa: f64,
    /// Observed range error in metres.
    pub range_error_m: f64,
    /// Observed Doppler error in metres per second.
    pub doppler_error_mps: f64,
    /// Observed OSPA distance.
    pub ospa_distance: f64,
    /// Number of samples that backed the observation. Reported
    /// alongside the gate result so downstream consumers can reason
    /// about confidence.
    pub sample_count: usize,
}

/// Gate report — what the V3 gate produces.
///
/// `tier` is the symbolic label ("V3" or "benchmarked") that downstream
/// consumers may surface. `status` is the boolean-ish gate outcome.
/// `failures` enumerates the metric names that violated their
/// threshold (empty on pass).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V3GateReport {
    /// Symbolic tier label. Set to `"V3"` for the ladder vocabulary;
    /// callers that want the schema-alias form may map this to
    /// `"benchmarked"` (see [`tier_alias_benchmarked`]).
    pub tier: &'static str,
    pub status: GateStatus,
    /// List of metric names that violated their threshold. Empty on
    /// pass. The strings are stable identifiers
    /// (`"pd"`, `"pfa"`, `"range_error"`, `"doppler_error"`, `"ospa"`)
    /// so machine readers can match on them.
    pub failures: Vec<String>,
    pub observation: V3MetricObservation,
    pub thresholds: V3MetricThresholds,
}

/// Convenience constant: the schema-alias string for V3.
pub const fn tier_alias_benchmarked() -> &'static str {
    "benchmarked"
}

/// Evaluate the V3 promotion gate.
///
/// Returns a [`V3GateReport`] with `status = pass` when every observed
/// metric beats its threshold; otherwise `status = fail` and `failures`
/// lists every violated metric name. This evaluator never returns
/// `warn` — V3 is a strict pass/fail gate by construction.
pub fn evaluate_v3_gate(
    thresholds: &V3MetricThresholds,
    observation: &V3MetricObservation,
) -> V3GateReport {
    let mut failures = Vec::new();
    if !(observation.pd >= thresholds.pd_min) {
        failures.push("pd".to_string());
    }
    if !(observation.pfa <= thresholds.pfa_max) {
        failures.push("pfa".to_string());
    }
    if !(observation.range_error_m <= thresholds.range_error_max_m) {
        failures.push("range_error".to_string());
    }
    if !(observation.doppler_error_mps <= thresholds.doppler_error_max_mps) {
        failures.push("doppler_error".to_string());
    }
    if !(observation.ospa_distance <= thresholds.ospa_max) {
        failures.push("ospa".to_string());
    }

    let status = if failures.is_empty() {
        GateStatus::Pass
    } else {
        GateStatus::Fail
    };

    V3GateReport {
        tier: "V3",
        status,
        failures,
        observation: observation.clone(),
        thresholds: thresholds.clone(),
    }
}

/// Loose deserialization of a `benchmark_report.json` for the V3 gate.
///
/// The schema (`schemas/benchmark_report.schema.json`) keeps
/// `additionalProperties: true` and does not yet require the
/// metric fields; this struct picks up the optional metric block when
/// it is present and emits a friendly [`ValidateError::Schema`] when
/// it is not.
#[derive(Debug, Deserialize)]
struct BenchmarkReportSlim {
    #[serde(default)]
    metrics: Option<BenchmarkMetricsSlim>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkMetricsSlim {
    #[serde(default)]
    pd: Option<f64>,
    #[serde(default)]
    pfa: Option<f64>,
    #[serde(default)]
    range_error_m: Option<f64>,
    #[serde(default)]
    doppler_error_mps: Option<f64>,
    #[serde(default)]
    ospa_distance: Option<f64>,
    #[serde(default)]
    sample_count: Option<usize>,
}

/// Read a `benchmark_report.json`, extract the expected metric fields,
/// and run the V3 gate against the supplied thresholds.
///
/// Returns [`ValidateError::Schema`] when the file is present but does
/// not contain the expected `metrics.{pd, pfa, range_error_m,
/// doppler_error_mps, ospa_distance, sample_count}` block. Returns
/// [`ValidateError::Io`] / [`ValidateError::Json`] for file-system or
/// JSON-parse failures.
pub fn evaluate_v3_from_benchmark_json(
    thresholds: &V3MetricThresholds,
    benchmark_path: &Path,
) -> Result<V3GateReport, ValidateError> {
    let raw = fs::read_to_string(benchmark_path)?;
    let parsed: BenchmarkReportSlim = serde_json::from_str(&raw)
        .map_err(|e| ValidateError::Schema(format!("{}: {}", benchmark_path.display(), e)))?;
    let metrics = parsed.metrics.ok_or_else(|| {
        ValidateError::Schema(format!(
            "{}: missing required `metrics` block for V3 gate (expected fields: pd, pfa, range_error_m, doppler_error_mps, ospa_distance, sample_count)",
            benchmark_path.display()
        ))
    })?;
    let observation = V3MetricObservation {
        pd: metrics
            .pd
            .ok_or_else(|| missing_field(benchmark_path, "metrics.pd"))?,
        pfa: metrics
            .pfa
            .ok_or_else(|| missing_field(benchmark_path, "metrics.pfa"))?,
        range_error_m: metrics
            .range_error_m
            .ok_or_else(|| missing_field(benchmark_path, "metrics.range_error_m"))?,
        doppler_error_mps: metrics
            .doppler_error_mps
            .ok_or_else(|| missing_field(benchmark_path, "metrics.doppler_error_mps"))?,
        ospa_distance: metrics
            .ospa_distance
            .ok_or_else(|| missing_field(benchmark_path, "metrics.ospa_distance"))?,
        sample_count: metrics
            .sample_count
            .ok_or_else(|| missing_field(benchmark_path, "metrics.sample_count"))?,
    };
    Ok(evaluate_v3_gate(thresholds, &observation))
}

fn missing_field(path: &Path, field: &str) -> ValidateError {
    ValidateError::Schema(format!(
        "{}: missing required field `{}` for V3 gate",
        path.display(),
        field
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn baseline_thresholds() -> V3MetricThresholds {
        V3MetricThresholds::default_roadmap()
    }

    fn passing_observation() -> V3MetricObservation {
        V3MetricObservation {
            pd: 0.95,
            pfa: 0.005,
            range_error_m: 2.0,
            doppler_error_mps: 0.5,
            ospa_distance: 4.0,
            sample_count: 10_000,
        }
    }

    #[test]
    fn evaluate_v3_passes_when_every_metric_beats_threshold() {
        let report = evaluate_v3_gate(&baseline_thresholds(), &passing_observation());
        assert_eq!(report.status, GateStatus::Pass);
        assert!(report.failures.is_empty());
        assert_eq!(report.tier, "V3");
        assert_eq!(tier_alias_benchmarked(), "benchmarked");
        assert_eq!(report.observation.sample_count, 10_000);
    }

    #[test]
    fn evaluate_v3_fails_on_low_pd() {
        let mut obs = passing_observation();
        obs.pd = 0.5; // below 0.85
        let report = evaluate_v3_gate(&baseline_thresholds(), &obs);
        assert_eq!(report.status, GateStatus::Fail);
        assert_eq!(report.failures, vec!["pd".to_string()]);
    }

    #[test]
    fn evaluate_v3_fails_on_multiple_violations() {
        let mut obs = passing_observation();
        obs.pd = 0.3; // pd violation
        obs.pfa = 0.5; // pfa violation
        obs.range_error_m = 100.0; // range violation
        obs.ospa_distance = 999.0; // ospa violation
        let report = evaluate_v3_gate(&baseline_thresholds(), &obs);
        assert_eq!(report.status, GateStatus::Fail);
        assert_eq!(report.failures.len(), 4);
        assert!(report.failures.contains(&"pd".to_string()));
        assert!(report.failures.contains(&"pfa".to_string()));
        assert!(report.failures.contains(&"range_error".to_string()));
        assert!(report.failures.contains(&"ospa".to_string()));
        // doppler is fine in this fixture
        assert!(!report.failures.contains(&"doppler_error".to_string()));
    }

    #[test]
    fn evaluate_v3_fails_on_each_axis_individually() {
        // Confirms every threshold is independently wired into the gate.
        let cases: Vec<(&str, Box<dyn Fn(&mut V3MetricObservation)>)> = vec![
            ("pd", Box::new(|o: &mut V3MetricObservation| o.pd = 0.0)),
            ("pfa", Box::new(|o: &mut V3MetricObservation| o.pfa = 1.0)),
            (
                "range_error",
                Box::new(|o: &mut V3MetricObservation| o.range_error_m = 1e6),
            ),
            (
                "doppler_error",
                Box::new(|o: &mut V3MetricObservation| o.doppler_error_mps = 1e6),
            ),
            (
                "ospa",
                Box::new(|o: &mut V3MetricObservation| o.ospa_distance = 1e6),
            ),
        ];
        for (name, mutate) in cases {
            let mut obs = passing_observation();
            mutate(&mut obs);
            let report = evaluate_v3_gate(&baseline_thresholds(), &obs);
            assert_eq!(report.status, GateStatus::Fail, "axis {name}");
            assert_eq!(report.failures, vec![name.to_string()], "axis {name}");
        }
    }

    #[test]
    fn evaluate_v3_pd_equal_to_threshold_passes() {
        // Boundary: equal-to threshold must pass (>=, <=).
        let thresholds = baseline_thresholds();
        let obs = V3MetricObservation {
            pd: thresholds.pd_min,
            pfa: thresholds.pfa_max,
            range_error_m: thresholds.range_error_max_m,
            doppler_error_mps: thresholds.doppler_error_max_mps,
            ospa_distance: thresholds.ospa_max,
            sample_count: 1,
        };
        let report = evaluate_v3_gate(&thresholds, &obs);
        assert_eq!(report.status, GateStatus::Pass);
        assert!(report.failures.is_empty());
    }

    #[test]
    fn evaluate_v3_from_benchmark_json_passes_on_valid_file() {
        let body = r#"{
            "schema_ref": "schemas/benchmark_report.schema.json",
            "dataset_card_schema_ref": "schemas/dataset_card.schema.json",
            "benchmark_id": "ef:benchmark:demo:0000000000000000:1",
            "dataset_id": "ef:dataset_card:demo:0000000000000000:1",
            "status": "draft",
            "required_sections": [],
            "required_artifacts": [],
            "known_limitations": [],
            "metrics": {
                "pd": 0.95,
                "pfa": 0.005,
                "range_error_m": 2.0,
                "doppler_error_mps": 0.5,
                "ospa_distance": 4.0,
                "sample_count": 12345
            }
        }"#;
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(body.as_bytes()).unwrap();
        let report = evaluate_v3_from_benchmark_json(&baseline_thresholds(), tf.path()).unwrap();
        assert_eq!(report.status, GateStatus::Pass);
        assert_eq!(report.observation.sample_count, 12345);
    }

    #[test]
    fn evaluate_v3_from_benchmark_json_reports_failure() {
        let body = r#"{
            "metrics": {
                "pd": 0.10,
                "pfa": 0.50,
                "range_error_m": 99.0,
                "doppler_error_mps": 9.0,
                "ospa_distance": 999.0,
                "sample_count": 1
            }
        }"#;
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(body.as_bytes()).unwrap();
        let report = evaluate_v3_from_benchmark_json(&baseline_thresholds(), tf.path()).unwrap();
        assert_eq!(report.status, GateStatus::Fail);
        assert_eq!(report.failures.len(), 5);
    }

    #[test]
    fn evaluate_v3_from_benchmark_json_friendly_error_when_metrics_missing() {
        let body = r#"{
            "schema_ref": "schemas/benchmark_report.schema.json",
            "dataset_card_schema_ref": "schemas/dataset_card.schema.json",
            "benchmark_id": "ef:benchmark:demo:0000000000000000:1",
            "dataset_id": "ef:dataset_card:demo:0000000000000000:1",
            "status": "draft",
            "required_sections": [],
            "required_artifacts": [],
            "known_limitations": []
        }"#;
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(body.as_bytes()).unwrap();
        let err = evaluate_v3_from_benchmark_json(&baseline_thresholds(), tf.path())
            .err()
            .unwrap();
        match err {
            ValidateError::Schema(msg) => {
                assert!(
                    msg.contains("missing required `metrics` block"),
                    "msg={msg}"
                );
            }
            other => panic!("expected ValidateError::Schema, got {other:?}"),
        }
    }

    #[test]
    fn evaluate_v3_from_benchmark_json_friendly_error_when_a_field_missing() {
        // Metrics block present but missing `ospa_distance`.
        let body = r#"{
            "metrics": {
                "pd": 0.95,
                "pfa": 0.005,
                "range_error_m": 2.0,
                "doppler_error_mps": 0.5,
                "sample_count": 100
            }
        }"#;
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(body.as_bytes()).unwrap();
        let err = evaluate_v3_from_benchmark_json(&baseline_thresholds(), tf.path())
            .err()
            .unwrap();
        match err {
            ValidateError::Schema(msg) => {
                assert!(
                    msg.contains("metrics.ospa_distance"),
                    "expected message to mention metrics.ospa_distance, got {msg}"
                );
            }
            other => panic!("expected ValidateError::Schema, got {other:?}"),
        }
    }

    #[test]
    fn evaluate_v3_from_benchmark_json_io_error_when_file_missing() {
        let path = Path::new("/nonexistent/echoforge/__no_such_benchmark_report__.json");
        let err = evaluate_v3_from_benchmark_json(&baseline_thresholds(), path)
            .err()
            .unwrap();
        assert!(matches!(err, ValidateError::Io(_)), "got {err:?}");
    }

    #[test]
    fn nan_pd_is_treated_as_failure() {
        // NaN comparisons via `>=` are false; the gate must therefore
        // treat NaN as a failure rather than silently passing.
        let mut obs = passing_observation();
        obs.pd = f64::NAN;
        let report = evaluate_v3_gate(&baseline_thresholds(), &obs);
        assert_eq!(report.status, GateStatus::Fail);
        assert!(report.failures.contains(&"pd".to_string()));
    }

    #[test]
    fn v3_gate_does_not_mention_v0_v1_v2_strings() {
        // Per packet boundary: the V3 evaluator must not silently
        // downgrade or upgrade tier strings. The tier label is fixed
        // to "V3"; downstream callers map to the schema alias
        // "benchmarked" via `tier_alias_benchmarked()`.
        let report = evaluate_v3_gate(&baseline_thresholds(), &passing_observation());
        assert_eq!(report.tier, "V3");
        assert_ne!(report.tier, "V0");
        assert_ne!(report.tier, "V1");
        assert_ne!(report.tier, "V2");
    }
}
