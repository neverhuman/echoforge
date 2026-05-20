//! Variant synthesizer.
//!
//! Given a `FailureCluster` and a base `ObjectClass`, produces a perturbed
//! variant entry that nudges the failing-class envelopes toward the failure
//! region. The strategy is intentionally conservative — we widen, never
//! shrink, and we only touch envelopes the curator has agreed are safe to
//! probabilistically extend (kinematics, micro-motion, sensor-observables,
//! and the upper RCS bound).
//!
//! The output is an `ObjectClassDelta`: a fully realized new `ObjectClass`
//! with a derived id (`<base_id>-mined-variant-<N>`), the `mined_variant` tag, and a
//! `mined_provenance` extra block describing the cluster that motivated the
//! perturbation. The delta is curator-reviewable before any merge into
//! `airspace-objects.json`.
//!
//! ### Perturbation policy
//!
//! For each envelope `[min, max]`, the widening factor is
//! `1.0 + (fp_rate - fp_threshold) * WIDEN_GAIN`, clamped to
//! `[WIDEN_MIN, WIDEN_MAX]`. The upper bound moves out by
//! `(max - min) * (widen - 1.0)` and the lower bound by an equal amount
//! downward (clamped to zero for fields that must be non-negative).
//!
//! For RCS the *upper* bound is allowed to move out by a small dB increment
//! because RCS dynamic range is already large; the lower bound is left alone
//! to avoid pushing the proxy below the radar's noise floor.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::cluster::FailureCluster;
use crate::types::{KinematicsBounds, MicroMotionBounds, ObjectClass, SensorObservableBounds};

/// Default gain applied to the (fp_rate - fp_threshold) headroom when
/// converting a cluster's intensity into an envelope-widening factor.
pub const DEFAULT_WIDEN_GAIN: f64 = 1.5;
/// Minimum widening factor (never shrink an envelope).
pub const WIDEN_MIN: f64 = 1.05;
/// Maximum widening factor per iteration (cap so a single hot cluster can't
/// blow out a proxy).
pub const WIDEN_MAX: f64 = 1.40;
/// RCS upper-bound bump in dB (additive, not multiplicative).
pub const RCS_UPPER_BUMP_DB: f64 = 2.5;

/// A proposed variant — a new `ObjectClass` plus the cluster metadata that
/// motivated it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectClassDelta {
    /// The new (derived) object class. Suitable for appending to
    /// `airspace-objects.json` once a curator approves.
    pub variant: ObjectClass,
    /// The cluster that triggered the synthesis.
    pub source_cluster: FailureCluster,
    /// id of the base class the variant was derived from.
    pub base_class_id: String,
    /// Index used when minting the derived id (`-mined-variant-<N>`).
    pub variant_index: u32,
    /// Threshold the synthesizer was operating under (echoed for audit).
    pub fp_threshold: f64,
}

/// Synthesize a variant from a cluster + base class.
///
/// `variant_index` is the integer used in the derived id; callers (e.g. the
/// loop driver) maintain this counter per base class so that successive
/// iterations don't collide.
pub fn synthesize_variant(
    cluster: &FailureCluster,
    base: &ObjectClass,
    fp_threshold: f64,
    variant_index: u32,
) -> ObjectClassDelta {
    let widen = compute_widen_factor(cluster.fp_rate, fp_threshold, DEFAULT_WIDEN_GAIN);
    let mut variant = base.clone();

    variant.id = format!("{}-mined-variant-{}", base.id, variant_index);
    let original_display = if base.display_name.is_empty() {
        base.id.clone()
    } else {
        base.display_name.clone()
    };
    variant.display_name = format!("{} (mined variant {})", original_display, variant_index);

    if !variant.role_tags.iter().any(|t| t == "mined_variant") {
        variant.role_tags.push("mined_variant".to_string());
    }

    variant.kinematics = widen_kinematics(&base.kinematics, widen);
    variant.micro_motion = widen_micro_motion(&base.micro_motion, widen);
    variant.sensor_observables = widen_sensor_observables(&base.sensor_observables, widen);

    if let Some(rcs) = base.rcs_dbsm {
        // RCS upper bound moves out by a fixed dB bump (not multiplicative —
        // dB scaling is already log).
        let new_max = rcs[1] + RCS_UPPER_BUMP_DB;
        variant.rcs_dbsm = Some([rcs[0], new_max]);
    }

    // Record the provenance of this synthesis in `extra` so curator review
    // can trace why the envelope moved. We never overwrite an existing
    // `mined_provenance` block — that would erase prior iteration history.
    let mut provenance = serde_json::Map::new();
    provenance.insert(
        "base_class_id".to_string(),
        serde_json::Value::String(base.id.clone()),
    );
    provenance.insert(
        "source_cluster_class_id".to_string(),
        serde_json::Value::String(cluster.class_id.clone()),
    );
    provenance.insert(
        "source_cluster_fp_rate".to_string(),
        serde_json::json!(cluster.fp_rate),
    );
    provenance.insert("fp_threshold".to_string(), serde_json::json!(fp_threshold));
    provenance.insert("widen_factor".to_string(), serde_json::json!(widen));
    provenance.insert(
        "variant_index".to_string(),
        serde_json::json!(variant_index),
    );
    variant.extra.insert(
        "mined_provenance".to_string(),
        serde_json::Value::Object(provenance),
    );

    ObjectClassDelta {
        variant,
        source_cluster: cluster.clone(),
        base_class_id: base.id.clone(),
        variant_index,
        fp_threshold,
    }
}

/// Compute the widening factor given a cluster's fp_rate, the
/// threshold, and a gain. Clamped to `[WIDEN_MIN, WIDEN_MAX]`.
pub fn compute_widen_factor(fp_rate: f64, fp_threshold: f64, gain: f64) -> f64 {
    let headroom = (fp_rate - fp_threshold).max(0.0);
    let raw = 1.0 + headroom * gain;
    raw.clamp(WIDEN_MIN, WIDEN_MAX)
}

fn widen_kinematics(base: &KinematicsBounds, widen: f64) -> KinematicsBounds {
    let mut out = base.clone();
    out.ground_speed_mps = base.ground_speed_mps.map(|b| widen_pos(b, widen));
    out.radial_velocity_mps = base.radial_velocity_mps.map(|b| widen_signed(b, widen));
    out.climb_rate_mps = base.climb_rate_mps.map(|b| widen_signed(b, widen));
    out.max_altitude_m = base.max_altitude_m.map(|b| widen_pos(b, widen));
    out.turn_rate_deg_s = base.turn_rate_deg_s.map(|b| widen_pos(b, widen));
    out.acceleration_mps2 = base.acceleration_mps2.map(|b| widen_pos(b, widen));
    out.altitude_agl_m = base.altitude_agl_m.map(|b| widen_pos(b, widen));
    out
}

fn widen_micro_motion(base: &MicroMotionBounds, widen: f64) -> MicroMotionBounds {
    let mut out = base.clone();
    out.propulsor_hz = base.propulsor_hz.map(|b| widen_pos(b, widen));
    out.micro_doppler_hz = base.micro_doppler_hz.map(|b| widen_pos(b, widen));
    out.amplitude_modulation = base.amplitude_modulation.map(|b| widen_pos(b, widen));
    out.attitude_jitter_deg = base.attitude_jitter_deg.map(|b| widen_pos(b, widen));
    out
}

fn widen_sensor_observables(base: &SensorObservableBounds, widen: f64) -> SensorObservableBounds {
    let mut out = base.clone();
    if let Some(b) = base.doppler_spread_bins {
        let lo = b[0];
        let hi_f = (b[1] as f64 * widen).ceil();
        let hi = hi_f.min(u32::MAX as f64) as u32;
        out.doppler_spread_bins = Some([lo, hi]);
    }
    out.scintillation_sigma = base.scintillation_sigma.map(|b| widen_pos(b, widen));
    // classification_prior is a probability; leave it alone — perturbing it
    // here would silently reshape the dataset balance, which the curator
    // should control.
    out
}

/// Widen a non-negative `[min, max]` envelope by `widen` factor.
///
/// The lower bound is reduced toward zero by the same delta the upper bound
/// is increased by; both are clamped to `>= 0.0`.
fn widen_pos(b: [f64; 2], widen: f64) -> [f64; 2] {
    let (lo, hi) = (b[0].min(b[1]), b[0].max(b[1]));
    let span = (hi - lo).max(0.0);
    let delta = span * (widen - 1.0) * 0.5;
    let new_lo = (lo - delta).max(0.0);
    let new_hi = hi + delta;
    [new_lo, new_hi]
}

/// Widen a signed `[min, max]` envelope (no zero clamp).
fn widen_signed(b: [f64; 2], widen: f64) -> [f64; 2] {
    let (lo, hi) = (b[0].min(b[1]), b[0].max(b[1]));
    let span = (hi - lo).max(0.0);
    let delta = span * (widen - 1.0) * 0.5;
    [lo - delta, hi + delta]
}

/// Materialize a synthesis batch — one delta per cluster, walking the
/// supplied base classes. Clusters whose `class_id` does not match any base
/// class are skipped (the caller can pick up the slack with a custom matcher
/// in `loop_driver` if desired).
pub fn synthesize_batch(
    clusters: &[FailureCluster],
    base_classes: &[ObjectClass],
    fp_threshold: f64,
    max_variants: usize,
    starting_index: u32,
) -> Vec<ObjectClassDelta> {
    let by_family = index_by_object_family(base_classes);
    let by_id: BTreeMap<&str, &ObjectClass> =
        base_classes.iter().map(|c| (c.id.as_str(), c)).collect();

    let mut out = Vec::new();
    let mut next_index = starting_index;
    for cluster in clusters {
        if out.len() >= max_variants {
            break;
        }
        // Match by object_family first; then by direct id if the family lookup missed.
        let base = match by_family.get(cluster.class_id.as_str()).copied() {
            Some(b) => Some(b),
            None => by_id.get(cluster.class_id.as_str()).copied(),
        };
        let Some(base) = base else {
            continue;
        };
        out.push(synthesize_variant(cluster, base, fp_threshold, next_index));
        next_index = next_index.saturating_add(1);
    }
    out
}

fn index_by_object_family(classes: &[ObjectClass]) -> BTreeMap<&str, &ObjectClass> {
    // Multiple classes may share an object_family; we keep the first.
    let mut out = BTreeMap::new();
    for class in classes {
        out.entry(class.object_family.as_str()).or_insert(class);
    }
    out
}
