use super::*;

fn typical_cooled_mwir() -> EoIrSensorParams {
    EoIrSensorParams {
        band: EoIrBand::MwirCooled,
        focal_length_mm: 200.0,
        aperture_diameter_mm: 100.0,
        fov_h_deg: 2.7,
        fov_v_deg: 2.2,
        array_h_pixels: 640,
        array_v_pixels: 512,
        pitch_um: 15.0,
        netd_mk: 18.0,
        integration_time_ms: 2.0,
        optics_transmission: 0.85,
    }
}

#[test]
fn ifov_200mm_15um_is_075_mrad() {
    let params = typical_cooled_mwir();
    let ifov = instantaneous_fov_mrad(&params);
    let expected = 0.075;
    assert!(
        (ifov - expected).abs() < 0.005,
        "IFOV for 200 mm / 15 μm was {ifov:.4} mrad, expected {expected} ± 0.005"
    );
}

#[test]
fn johnson_detect_range_shahed_class_geometry_only() {
    let params = typical_cooled_mwir();
    let range_m = declared_range_for_task(&params, 2.5, JohnsonTask::Detect);
    let expected_m = 33_333.0;
    assert!(
        (range_m - expected_m).abs() < 2_000.0,
        "Johnson detect range was {range_m:.0} m, expected {expected_m} m ± 2000"
    );
}

#[test]
fn recognize_range_is_quarter_of_detect_range() {
    let params = typical_cooled_mwir();
    let detect = declared_range_for_task(&params, 2.5, JohnsonTask::Detect);
    let recognize = declared_range_for_task(&params, 2.5, JohnsonTask::Recognize);
    assert!(
        (recognize - detect / 4.0).abs() < 1.0,
        "recognize range {recognize:.1} m ≠ detect/4 ({:.1} m)",
        detect / 4.0
    );
}

#[test]
fn johnson_required_pixels_canonical_values() {
    assert_eq!(johnson_required_pixels(JohnsonTask::Detect), 1);
    assert_eq!(johnson_required_pixels(JohnsonTask::Recognize), 4);
    assert_eq!(johnson_required_pixels(JohnsonTask::Identify), 8);
}

#[test]
fn pixels_on_target_halves_when_range_doubles() {
    let params = typical_cooled_mwir();
    let near = pixels_on_target(&params, 2.5, 5_000.0);
    let far = pixels_on_target(&params, 2.5, 10_000.0);
    assert!(
        (near - 2.0 * far).abs() < 0.01,
        "pixels-on-target should halve when range doubles: near={near:.4}, far={far:.4}"
    );
}

#[test]
fn snr_with_weather_composes_with_optics() {
    let params = typical_cooled_mwir();
    let snr = eo_ir_snr_with_weather(&params, 5_000.0, 0.60, 23.0, 0.15);
    assert!(
        snr > 1e-3 && snr < params.optics_transmission,
        "expected weather-composed SNR strictly below optics τ ({}) and well above clamp; got {snr:.4}",
        params.optics_transmission
    );
    assert!(
        (0.40..0.55).contains(&snr),
        "weather-composed SNR {snr:.3} outside 0.40-0.55 window for MWIR"
    );
}
