use super::*;
use crate::distribution_metrics::{ks_distance_1d_sorted, wasserstein_1d_sorted};
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
