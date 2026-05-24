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
//! Per the local coordination archive resolution, the V3 gate does NOT
//! interact with the `fidelity_class` (F0..F5) method-ceiling axis;
//! see `docs/validation-tiers.md` "Why the two axes are independent".
//!
//! JSON loading lives in [`crate::tier_benchmarked_json`] to keep the
//! two public entry points in separate compilation units.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

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

/// Threshold envelope for V3 benchmark metrics. Fields: `pd_min` (≥ 0.85
/// default), `pfa_max` (≤ 0.01), `range_error_max_m`, `doppler_error_max_mps`,
/// `ospa_max` — all campaign-specific except pd/pfa which have roadmap defaults.
/// Every observed metric must beat its threshold for the V3 gate to pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V3MetricThresholds {
    pub pd_min: f64,
    pub pfa_max: f64,
    pub range_error_max_m: f64,
    pub doppler_error_max_mps: f64,
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
    fn meets_min(value: f64, minimum: f64) -> bool {
        matches!(
            value.partial_cmp(&minimum),
            Some(Ordering::Equal | Ordering::Greater)
        )
    }

    fn meets_max(value: f64, maximum: f64) -> bool {
        matches!(
            value.partial_cmp(&maximum),
            Some(Ordering::Equal | Ordering::Less)
        )
    }

    let failures: Vec<String> = [
        (!meets_min(observation.pd, thresholds.pd_min)).then_some("pd"),
        (!meets_max(observation.pfa, thresholds.pfa_max)).then_some("pfa"),
        (!meets_max(observation.range_error_m, thresholds.range_error_max_m))
            .then_some("range_error"),
        (!meets_max(
            observation.doppler_error_mps,
            thresholds.doppler_error_max_mps,
        ))
        .then_some("doppler_error"),
        (!meets_max(observation.ospa_distance, thresholds.ospa_max)).then_some("ospa"),
    ]
    .into_iter()
    .flatten()
    .map(str::to_owned)
    .collect();
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

#[cfg(test)]
mod tests {
    use super::*;

    type ObservationMutator = Box<dyn Fn(&mut V3MetricObservation)>;

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
        obs.pd = 0.5;
        let report = evaluate_v3_gate(&baseline_thresholds(), &obs);
        assert_eq!(report.status, GateStatus::Fail);
        assert_eq!(report.failures, vec!["pd".to_string()]);
    }

    #[test]
    fn evaluate_v3_fails_on_multiple_violations() {
        let mut obs = passing_observation();
        obs.pd = 0.3;
        obs.pfa = 0.5;
        obs.range_error_m = 100.0;
        obs.ospa_distance = 999.0;
        let report = evaluate_v3_gate(&baseline_thresholds(), &obs);
        assert_eq!(report.status, GateStatus::Fail);
        assert_eq!(report.failures.len(), 4);
        assert!(report.failures.contains(&"pd".to_string()));
        assert!(report.failures.contains(&"pfa".to_string()));
        assert!(report.failures.contains(&"range_error".to_string()));
        assert!(report.failures.contains(&"ospa".to_string()));
        assert!(!report.failures.contains(&"doppler_error".to_string()));
    }

    #[test]
    fn evaluate_v3_fails_on_each_axis_individually() {
        let cases: Vec<(&str, ObservationMutator)> = vec![
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
    fn nan_pd_is_treated_as_failure() {
        let mut obs = passing_observation();
        obs.pd = f64::NAN;
        let report = evaluate_v3_gate(&baseline_thresholds(), &obs);
        assert_eq!(report.status, GateStatus::Fail);
        assert!(report.failures.contains(&"pd".to_string()));
    }

    #[test]
    fn v3_gate_does_not_mention_v0_v1_v2_strings() {
        let report = evaluate_v3_gate(&baseline_thresholds(), &passing_observation());
        assert_eq!(report.tier, "V3");
        assert_ne!(report.tier, "V0");
        assert_ne!(report.tier, "V1");
        assert_ne!(report.tier, "V2");
    }
}
