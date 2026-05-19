//! Minimal JSON-shaped types used across the mining loop.
//!
//! These intentionally mirror the on-disk shapes of:
//!
//! * `outputs/campaigns/<id>/qa/model_eval_<model>.json` (per-detector
//!   per-frame outcome rollup written by the streaming campaign runner)
//! * `configs/monte-carlo/airspace-objects.json` (Monte-Carlo object-class
//!   library)
//!
//! They do NOT depend on `echoforge-dataset` or `echoforge-radar` — that
//! decoupling lets the mining loop be developed and tested without coupling
//! to the in-flight streaming and radar-clutter packets.
//!
//! Where the upstream JSON has fields we don't need (e.g. ROC point sets,
//! confidence calibration bins, sensor archetypes), we use a `serde_json::Value`
//! container so the round-trip stays faithful even when the upstream schema
//! grows new optional fields.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A single model-evaluation rollup (one detector, one campaign run).
///
/// Field shape matches `echoforge_dataset::campaign::ModelEvaluationReport`
/// as of 2026-05-18; we re-declare it here so this crate can be built and
/// tested without pulling in the dataset crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEvalRollup {
    pub model_id: String,
    pub positive_records: usize,
    pub negative_records: usize,
    pub pd: f64,
    pub pfa: f64,
    #[serde(default)]
    pub missed_positive_records: Vec<String>,
    /// Per-hard-negative-family false-alarm counts. The keys are the family
    /// strings that the streaming runner emits (e.g.
    /// `"commercial_aircraft_corridor"`); empty family means an unclassified
    /// negative scene.
    #[serde(default)]
    pub false_alarm_by_hard_negative_family: BTreeMap<String, usize>,
    /// Optional latency stats — reserved by the current cluster store but kept
    /// in the round-trip so downstream tools can correlate.
    #[serde(default)]
    pub mean_first_detection_latency_frames: Option<f64>,
    /// Catch-all for ROC, PR, and calibration arrays we don't read. Storing
    /// as `Value` keeps the upstream JSON growable without bumping this crate.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Top-level object-class library, mirroring
/// `configs/monte-carlo/airspace-objects.json`.
///
/// We keep only the fields the synthesizer reads (`object_classes`) and use
/// `extra` for the rest so a future caller could round-trip the file without
/// data loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirspaceObjectsConfig {
    #[serde(default)]
    pub config_id: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub purpose: String,
    #[serde(default)]
    pub guardrails: Vec<String>,
    pub object_classes: Vec<ObjectClass>,
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// One object-class entry from the Monte-Carlo library.
///
/// The shape mirrors the entries in `airspace-objects.json`. Numeric
/// envelope fields are `[min, max]` arrays. We pin the fields the synthesizer
/// perturbs (micro_motion, kinematics, sensor_observables) and leave the rest
/// as `Value`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectClass {
    pub id: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub object_family: String,
    #[serde(default)]
    pub role_tags: Vec<String>,
    /// Free-form dimensions block (length/wingspan/height). Kept opaque so the
    /// synthesizer doesn't have to commit to one schema version.
    #[serde(default)]
    pub dimensions_m: serde_json::Value,
    /// `[min, max]` RCS envelope in dBsm; perturbed when the failure cluster
    /// indicates RCS-mismatch confusion.
    #[serde(default)]
    pub rcs_dbsm: Option<[f64; 2]>,
    /// Mixture composition; opaque (the synthesizer does not touch material
    /// chemistry without curator approval).
    #[serde(default)]
    pub material_mix: serde_json::Value,
    /// Kinematics envelope — perturbed by the synthesizer.
    #[serde(default)]
    pub kinematics: KinematicsBounds,
    /// Micro-motion envelope — perturbed by the synthesizer.
    #[serde(default)]
    pub micro_motion: MicroMotionBounds,
    /// Behavior block — passed through.
    #[serde(default)]
    pub behavior: serde_json::Value,
    /// Sensor observable envelopes — perturbed by the synthesizer.
    #[serde(default)]
    pub sensor_observables: SensorObservableBounds,
    /// Any extra fields are preserved verbatim.
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Subset of the kinematics block the synthesizer actually perturbs.
///
/// All fields are optional so older fixtures without one or another field
/// still deserialize. The on-disk JSON shape is preserved exactly.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KinematicsBounds {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ground_speed_mps: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radial_velocity_mps: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub climb_rate_mps: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_altitude_m: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_rate_deg_s: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceleration_mps2: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub altitude_agl_m: Option<[f64; 2]>,
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Subset of the micro-motion block the synthesizer actually perturbs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MicroMotionBounds {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub propulsor_hz: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub micro_doppler_hz: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amplitude_modulation: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attitude_jitter_deg: Option<[f64; 2]>,
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Subset of the sensor-observables block the synthesizer actually perturbs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SensorObservableBounds {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doppler_spread_bins: Option<[u32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scintillation_sigma: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classification_prior: Option<f64>,
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl AirspaceObjectsConfig {
    /// Look up an object class by id. Returns `None` if no class with this id
    /// is present in the library.
    pub fn class_by_id(&self, id: &str) -> Option<&ObjectClass> {
        self.object_classes.iter().find(|c| c.id == id)
    }
}
