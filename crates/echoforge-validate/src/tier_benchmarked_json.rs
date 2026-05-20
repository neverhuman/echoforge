//! JSON-backed benchmark report loader for the V3 gate.
//!
//! Reads a `benchmark_report.json`, extracts the `metrics` block, and
//! delegates to [`crate::tier_benchmarked::evaluate_v3_gate`].

use std::fs;
use std::path::Path;

use serde_json::{Map, Value};

use crate::error::ValidateError;
use crate::tier_benchmarked::{evaluate_v3_gate, V3GateReport, V3MetricObservation, V3MetricThresholds};

/// Read a `benchmark_report.json`, extract the metrics block, and run
/// the V3 gate against the supplied thresholds.
///
/// Returns [`ValidateError::Schema`] when the `metrics` block is absent
/// or malformed. Returns [`ValidateError::Io`] for filesystem failures.
pub fn evaluate_v3_from_benchmark_json(
    thresholds: &V3MetricThresholds,
    benchmark_path: &Path,
) -> Result<V3GateReport, ValidateError> {
    let raw = fs::read_to_string(benchmark_path)?;
    let mut root: Map<String, Value> = serde_json::from_str(&raw)
        .map_err(|e| ValidateError::Schema(format!("{}: {}", benchmark_path.display(), e)))?;
    let Some(metrics_val) = root.remove("metrics") else {
        return Err(ValidateError::Schema(format!(
            "{}: missing required `metrics` block for V3 gate",
            benchmark_path.display()
        )));
    };
    let obs: V3MetricObservation = serde_json::from_value(metrics_val)
        .map_err(|e| ValidateError::Schema(format!("{}: {}", benchmark_path.display(), e)))?;
    Ok(evaluate_v3_gate(thresholds, &obs))
}

#[cfg(test)]
mod tests {
    use super::evaluate_v3_from_benchmark_json;
    use crate::error::ValidateError;
    use crate::tier_benchmarked::{GateStatus, V3GateReport, V3MetricThresholds};
    use std::io::Write;
    use std::path::Path;
    use tempfile::NamedTempFile;

    fn run(json: &str) -> Result<V3GateReport, ValidateError> {
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(json.as_bytes()).unwrap();
        evaluate_v3_from_benchmark_json(&V3MetricThresholds::default_roadmap(), tf.path())
    }

    #[test]
    fn json_pass() {
        let r = run(r#"{"metrics":{"pd":0.95,"pfa":0.005,"range_error_m":2.0,"doppler_error_mps":0.5,"ospa_distance":4.0,"sample_count":12345}}"#).unwrap();
        assert_eq!(r.status, GateStatus::Pass);
        assert_eq!(r.observation.sample_count, 12345);
    }

    #[test]
    fn json_fail_all_axes() {
        let r = run(r#"{"metrics":{"pd":0.10,"pfa":0.50,"range_error_m":99.0,"doppler_error_mps":9.0,"ospa_distance":999.0,"sample_count":1}}"#).unwrap();
        assert_eq!(r.status, GateStatus::Fail);
        assert_eq!(r.failures.len(), 5);
    }

    #[test]
    fn json_error_no_metrics_block() {
        // let-else to assert the specific error variant
        let ValidateError::Schema(msg) = run(r#"{"benchmark_id":"x","status":"draft"}"#).unwrap_err() else {
            panic!("expected ValidateError::Schema");
        };
        assert!(msg.contains("missing required `metrics` block"), "msg={msg}");
    }

    #[test]
    fn json_error_missing_field() {
        // match to inspect the serde field name in the error message
        let body = r#"{"metrics":{"pd":0.95,"pfa":0.005,"range_error_m":2.0,"doppler_error_mps":0.5,"sample_count":100}}"#;
        match run(body).unwrap_err() {
            ValidateError::Schema(msg) => assert!(msg.contains("ospa_distance"), "msg={msg}"),
            other => panic!("expected Schema, got {other:?}"),
        }
    }

    #[test]
    fn json_io_error() {
        let path = Path::new("/nonexistent/echoforge/__no_such_benchmark_report__.json");
        let err = evaluate_v3_from_benchmark_json(&V3MetricThresholds::default_roadmap(), path).unwrap_err();
        assert!(matches!(err, ValidateError::Io(_)), "got {err:?}");
    }
}
