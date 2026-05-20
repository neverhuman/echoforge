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
    let json: serde_json::Value = serde_json::to_value(&report).expect("serialize to value");
    assert_eq!(json["tier"], "V4");
    assert_eq!(json["status"], "pass");
    assert_eq!(json["anchor_count"], 3);
    assert!(json["outcomes"].as_array().unwrap().len() == 3);
    let first = &json["outcomes"][0];
    assert_eq!(first["feature_name"], "snr_db");
    assert_eq!(first["metric"], "wasserstein_1");
    assert_eq!(first["status"], "pass");
}
