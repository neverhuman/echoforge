use super::*;

#[test]
fn embedded_uae_coastal_parses() {
    let scenario = SurveillanceScenario::embedded_uae_coastal();
    assert_eq!(scenario.scenario_id, "uae-coastal-surveillance-v1");
    assert_eq!(scenario.schema_version, "1.0.0");
    assert!(!scenario.strict_open_posture.is_empty());
    assert!(!scenario.display_name.is_empty());
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
    let freq_hz = scenario.sensor.center_frequency_hz;
    assert!(
        (2.7e9..=3.1e9).contains(&freq_hz),
        "sensor center frequency {freq_hz} Hz must fall in 2.7-3.1 GHz S-band window"
    );
}

#[test]
fn embedded_scenario_antenna_height_is_twenty_m_agl() {
    let scenario = SurveillanceScenario::embedded_uae_coastal();
    assert_eq!(scenario.radar_site.antenna_height_agl_m, 20.0_f64);
}

#[test]
fn load_round_trips_through_tempfile() {
    let scenario = SurveillanceScenario::embedded_uae_coastal();
    let json = serde_json::to_string_pretty(&scenario)
        .expect("scenario must serialize back to JSON");
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("uae-coastal-roundtrip.json");
    std::fs::write(&path, &json).expect("write scenario to disk");
    let loaded = SurveillanceScenario::load(&path).expect("load back");
    assert_eq!(loaded, scenario);
}

#[test]
fn load_surfaces_io_error_when_file_missing() {
    let path = std::path::Path::new("/dev/null/does-not-exist/uae.json");
    let err = SurveillanceScenario::load(path).expect_err("missing file must fail");
    assert!(
        matches!(err, ScenarioLoadError::Io(_)),
        "expected ScenarioLoadError::Io, got {err:?}"
    );
}

#[test]
fn load_surfaces_schema_error_when_launch_sites_empty() {
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
