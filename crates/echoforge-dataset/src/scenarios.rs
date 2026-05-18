//! Surveillance-scenario archetype loader.
//!
//! This module owns the *site / sensor / target / environment* archetype
//! that downstream phase-timeline slicers and the unified scene generator
//! consume. It is a strict-open public-proxy artifact: every numeric value
//! is sourced from open-source surveillance-radar literature (see the
//! `citations` field of the embedded JSON), and **no parameter is calibrated
//! against a measured operational system**.
//!
//! The canonical scenario JSON lives at
//! `configs/scenarios/uae-coastal-surveillance-v1.json` and is embedded at
//! compile time via `include_str!` for the
//! [`SurveillanceScenario::embedded_uae_coastal`] convenience constructor.
//! This mirrors the pattern established by
//! [`crate::monte_carlo::embedded_airspace_config`].
//!
//! Re-exports from `lib.rs` are owned by the parent agent; do not edit
//! `lib.rs` from this module.
//!
//! # Strict-open posture
//!
//! - The radar coordinates in the embedded scenario are illustrative,
//!   not an operational deployment location.
//! - Sensor parameters are representative of a "long-range air
//!   surveillance" archetype consistent with published S-band surveillance
//!   literature; no specific deployed system is modeled or claimed.
//! - The `strict_open_posture` field on [`SurveillanceScenario`] is the
//!   stable string downstream consumers should surface to end users when
//!   citing this scenario.

use std::path::Path;

use serde::{Deserialize, Serialize};

const SCENARIO_JSON: &str =
    include_str!("../../../configs/scenarios/uae-coastal-surveillance-v1.json");

/// Geometry, height, and refractivity descriptor of the surveillance radar
/// site. All numeric fields are SI-units; angles are in degrees.
///
/// `k_factor_earth` is the effective-Earth radius multiplier used by the
/// 4/3-Earth approximation common in surveillance-radar horizon
/// calculations (Skolnik 2001, chap. 8).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RadarSite {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub ground_altitude_msl_m: f64,
    pub antenna_height_agl_m: f64,
    pub k_factor_earth: f64,
    pub terrain_class: String,
}

/// Pulse-Doppler surveillance-sensor archetype. All numeric fields are
/// SI-units; angles are in degrees; `dwell_ms_on_target` is in
/// milliseconds (per its name) for consumer convenience.
///
/// `archetype_id` is the stable lookup key downstream slicers should
/// pattern-match on rather than the raw display name.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SensorConfig {
    pub archetype_id: String,
    pub center_frequency_hz: f64,
    pub bandwidth_hz: f64,
    pub pulse_width_s: f64,
    pub pri_s: f64,
    pub pulse_count_per_cpi: usize,
    pub cpi_duration_s: f64,
    pub transmit_power_w: f64,
    pub antenna_gain_dbi: f64,
    pub beamwidth_az_deg: f64,
    pub beamwidth_el_deg: f64,
    pub sidelobe_max_db: f64,
    pub noise_figure_db: f64,
    pub system_loss_db: f64,
    pub polarization: String,
    pub scan_period_s: f64,
    pub dwell_ms_on_target: f64,
}

/// Target launch-site descriptor relative to the surveillance radar.
///
/// `bearing_deg_from_radar` is measured clockwise from true north as
/// conventional in surveillance-radar plot coordinates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TargetLaunchSite {
    pub site_id: String,
    pub range_km: f64,
    pub bearing_deg_from_radar: f64,
    pub elevation_msl_m: f64,
}

/// Coarse environmental state used by the propagation / attenuation
/// stages downstream of this archetype.
///
/// `terrain_class_at_target_sites` is parallel to
/// [`SurveillanceScenario::target_launch_sites`] and must have the same
/// length; consumers should treat a length mismatch as a producer bug.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentState {
    pub weather_regime: String,
    pub rain_rate_mm_per_hr: f64,
    pub temperature_k: f64,
    pub pressure_kpa: f64,
    pub water_vapor_g_per_m3: f64,
    pub terrain_class_at_target_sites: Vec<String>,
    pub sea_state: String,
}

/// Top-level surveillance-scenario archetype.
///
/// Construct via [`SurveillanceScenario::embedded_uae_coastal`] for the
/// in-tree default or [`SurveillanceScenario::load`] to read an external
/// JSON file matching the same shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurveillanceScenario {
    pub schema_version: String,
    pub scenario_id: String,
    pub display_name: String,
    pub strict_open_posture: String,
    pub radar_site: RadarSite,
    pub sensor: SensorConfig,
    pub target_launch_sites: Vec<TargetLaunchSite>,
    pub environment: EnvironmentState,
    pub citations: Vec<String>,
    pub notes: String,
}

impl SurveillanceScenario {
    /// Load a scenario archetype from a JSON file on disk.
    ///
    /// Errors are surfaced as [`ScenarioLoadError`] with a discriminator
    /// for I/O vs. JSON-parse vs. semantic-schema failures.
    pub fn load(path: &Path) -> Result<Self, ScenarioLoadError> {
        let raw = std::fs::read_to_string(path)?;
        let scenario: Self = serde_json::from_str(&raw)?;
        scenario.validate_schema()?;
        Ok(scenario)
    }

    /// Return the in-tree UAE-coastal v1 archetype, parsed from the
    /// JSON embedded at compile time.
    ///
    /// Panics only if the embedded JSON has been corrupted in-source
    /// (a compile-time-asserted invariant); the
    /// `embedded_uae_coastal_parses` unit test pins this contract.
    pub fn embedded_uae_coastal() -> Self {
        let scenario: Self = serde_json::from_str(SCENARIO_JSON)
            .expect("embedded UAE-coastal scenario JSON must parse; corrupted in-source");
        scenario
            .validate_schema()
            .expect("embedded UAE-coastal scenario must satisfy schema invariants");
        scenario
    }

    fn validate_schema(&self) -> Result<(), ScenarioLoadError> {
        if self.schema_version.is_empty() {
            return Err(ScenarioLoadError::Schema(
                "schema_version must not be empty".to_string(),
            ));
        }
        if self.scenario_id.is_empty() {
            return Err(ScenarioLoadError::Schema(
                "scenario_id must not be empty".to_string(),
            ));
        }
        if self.target_launch_sites.is_empty() {
            return Err(ScenarioLoadError::Schema(
                "target_launch_sites must list at least one site".to_string(),
            ));
        }
        if self.environment.terrain_class_at_target_sites.len()
            != self.target_launch_sites.len()
        {
            return Err(ScenarioLoadError::Schema(format!(
                "environment.terrain_class_at_target_sites length ({}) must equal target_launch_sites length ({})",
                self.environment.terrain_class_at_target_sites.len(),
                self.target_launch_sites.len()
            )));
        }
        if self.sensor.center_frequency_hz <= 0.0 {
            return Err(ScenarioLoadError::Schema(
                "sensor.center_frequency_hz must be strictly positive".to_string(),
            ));
        }
        if self.radar_site.antenna_height_agl_m < 0.0 {
            return Err(ScenarioLoadError::Schema(
                "radar_site.antenna_height_agl_m must be non-negative".to_string(),
            ));
        }
        if self.citations.is_empty() {
            return Err(ScenarioLoadError::Schema(
                "citations must list at least one open-source reference".to_string(),
            ));
        }
        Ok(())
    }
}

/// Error class surfaced by [`SurveillanceScenario::load`].
///
/// The variant set mirrors the spec-stated `thiserror::Error` enum on the
/// `gulf-site-propagation-archetypes-v1` packet so the public surface is
/// stable regardless of whether downstream code uses an external derive
/// macro; the `Display` strings are byte-stable to the spec.
#[derive(Debug)]
pub enum ScenarioLoadError {
    /// Filesystem I/O failure while reading the scenario JSON.
    Io(std::io::Error),
    /// `serde_json` failed to parse the scenario JSON.
    Json(serde_json::Error),
    /// JSON parsed but failed a semantic-schema invariant
    /// (e.g. mismatched parallel arrays, empty required field).
    Schema(String),
}

impl std::fmt::Display for ScenarioLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Json(err) => write!(f, "json parse error: {err}"),
            Self::Schema(msg) => write!(f, "schema validation error: {msg}"),
        }
    }
}

impl std::error::Error for ScenarioLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Json(err) => Some(err),
            Self::Schema(_) => None,
        }
    }
}

impl From<std::io::Error> for ScenarioLoadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for ScenarioLoadError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_uae_coastal_parses() {
        let scenario = SurveillanceScenario::embedded_uae_coastal();
        assert_eq!(scenario.scenario_id, "uae-coastal-surveillance-v1");
        assert_eq!(scenario.schema_version, "1.0.0");
        assert!(!scenario.strict_open_posture.is_empty());
        assert!(!scenario.display_name.is_empty());
        // Smoke-check that schema validation passed inside the constructor.
        scenario
            .validate_schema()
            .expect("embedded scenario must satisfy schema invariants");
    }

    #[test]
    fn embedded_scenario_has_exactly_three_launch_sites() {
        let scenario = SurveillanceScenario::embedded_uae_coastal();
        assert_eq!(
            scenario.target_launch_sites.len(),
            3,
            "v1 archetype pins three baseline ranges (50, 100, 150 km)"
        );
        // The parallel array invariant is the contract downstream slicers
        // rely on; pin it here in addition to validate_schema().
        assert_eq!(
            scenario.environment.terrain_class_at_target_sites.len(),
            scenario.target_launch_sites.len(),
        );
    }

    #[test]
    fn embedded_scenario_ranges_are_50_100_150_km() {
        let scenario = SurveillanceScenario::embedded_uae_coastal();
        let ranges: Vec<f64> = scenario
            .target_launch_sites
            .iter()
            .map(|site| site.range_km)
            .collect();
        // Order-stable per the JSON authoring convention.
        let expected = [50.0_f64, 100.0_f64, 150.0_f64];
        assert_eq!(ranges.len(), expected.len());
        for (got, want) in ranges.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() <= 1.0,
                "range {got} km outside 1 km tolerance of expected {want} km"
            );
        }
    }

    #[test]
    fn embedded_scenario_sensor_frequency_is_s_band() {
        let scenario = SurveillanceScenario::embedded_uae_coastal();
        // S-band is bounded 2.0-4.0 GHz by IEEE 521-2002; the v3 plan
        // pins the archetype to the 2.7-3.1 GHz long-range air-surveillance
        // band.
        let freq_hz = scenario.sensor.center_frequency_hz;
        assert!(
            (2.7e9..=3.1e9).contains(&freq_hz),
            "sensor center frequency {freq_hz} Hz must fall in 2.7-3.1 GHz S-band window"
        );
    }

    #[test]
    fn embedded_scenario_antenna_height_is_twenty_m_agl() {
        let scenario = SurveillanceScenario::embedded_uae_coastal();
        // Exact-equality on a literal authored value; intentional rather
        // than a tolerance-window because the v1 archetype pins this.
        assert_eq!(scenario.radar_site.antenna_height_agl_m, 20.0_f64);
    }

    #[test]
    fn load_round_trips_through_tempfile() {
        // Round-trip the embedded scenario through the disk-load path to
        // confirm load() is structurally compatible with the same shape
        // serde_json emits for the in-tree archetype. tempfile is a
        // dev-dependency on this crate (see Cargo.toml).
        let scenario = SurveillanceScenario::embedded_uae_coastal();
        let json = serde_json::to_string_pretty(&scenario)
            .expect("scenario must serialize back to JSON");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("uae-coastal-roundtrip.json");
        std::fs::write(&path, &json).expect("write temp scenario");
        let loaded = SurveillanceScenario::load(&path).expect("load back");
        assert_eq!(loaded, scenario);
    }

    #[test]
    fn load_surfaces_io_error_when_file_missing() {
        // Pin the discriminator on the error enum so downstream readers
        // can pattern-match. Also covers the From<std::io::Error> impl.
        let path = std::path::Path::new("/dev/null/does-not-exist/uae.json");
        let err = SurveillanceScenario::load(path).expect_err("missing file must fail");
        assert!(
            matches!(err, ScenarioLoadError::Io(_)),
            "expected ScenarioLoadError::Io, got {err:?}"
        );
    }

    #[test]
    fn load_surfaces_schema_error_when_launch_sites_empty() {
        // Pin the semantic-schema invariant: an empty target_launch_sites
        // list must NOT load silently.
        let mut scenario = SurveillanceScenario::embedded_uae_coastal();
        scenario.target_launch_sites.clear();
        scenario.environment.terrain_class_at_target_sites.clear();
        let json = serde_json::to_string(&scenario).expect("serialize");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("empty-sites.json");
        std::fs::write(&path, &json).expect("write");
        let err = SurveillanceScenario::load(&path).expect_err("empty sites must fail");
        assert!(matches!(err, ScenarioLoadError::Schema(_)));
    }
}
