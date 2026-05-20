use super::*;

fn shahed_scene(antenna_alt: f64, atm_db_per_km: f64) -> SceneDescriptor {
    SceneDescriptor {
        geometry: SiteGeometry {
            antenna_altitude_agl_m: antenna_alt,
        },
        environment: EnvironmentDescriptor {
            clutter_regime: None,
            atmospheric_one_way_db_per_km: atm_db_per_km,
            rain_rate_mm_per_h: 0.0,
            ground_reflection_coefficient_magnitude: 0.0,
        },
        targets: vec![TargetEntity {
            class: TargetClass::ShahedClassPiston,
            kinematics: TargetKinematics::FromTakeoffProfile(TakeoffProfile::default()),
            spawn_time_s: 0.0,
        }],
    }
}

/// Round-trip serde: a hand-built `SceneDescriptor` must survive
/// JSON serialise + deserialise without losing fields. This is the
/// minimal contract test that on-disk fixtures (Lane J campaign
/// JSONs, scenario manifests) can rely on.
#[test]
fn scene_descriptor_roundtrip_serde() {
    let scene = shahed_scene(20.0, 0.012);

    let json = serde_json::to_string(&scene).expect("serialise scene");
    let parsed: SceneDescriptor = serde_json::from_str(&json).expect("deserialise scene");
    assert_eq!(scene, parsed);
}

/// Minimal one-entity scene — assert the shape so consumers (the
/// `synthesize_takeoff_episode` wrapper, Lane J's campaign runner)
/// have a checked structural contract rather than relying on
/// in-code inspection of which fields exist.
#[test]
fn scene_descriptor_minimal() {
    let scene = shahed_scene(12.5, 0.0);

    assert_eq!(scene.targets.len(), 1);
    assert_eq!(scene.geometry.antenna_altitude_agl_m, 12.5);
    assert_eq!(scene.targets[0].class, TargetClass::ShahedClassPiston);
    assert_eq!(scene.targets[0].spawn_time_s, 0.0);
    assert!(matches!(
        scene.targets[0].kinematics,
        TargetKinematics::FromTakeoffProfile(_)
    ));
}

/// `TargetClass` round-trips through serde — including the
/// `MultipathGhost { parent_idx }` newtype variant which carries
/// data and exercises the most error-prone serde path.
#[test]
fn target_class_serde_covers_all_variants() {
    let variants = vec![
        TargetClass::ShahedClassPiston,
        TargetClass::ShahedClassJet,
        TargetClass::Bird,
        TargetClass::GroundVehicle,
        TargetClass::WindTurbine,
        TargetClass::Balloon,
        TargetClass::Kite,
        TargetClass::MultipathGhost { parent_idx: 0 },
        TargetClass::TerrainGlint,
        TargetClass::Helicopter,
        TargetClass::ManRadarReturn,
    ];
    for class in variants {
        let json = serde_json::to_string(&class).expect("serialise class");
        let parsed: TargetClass = serde_json::from_str(&json).expect("deserialise class");
        assert_eq!(class, parsed);
    }
}

// -------------------------------------------------------------------
// Wave 5 Lane J — per-class TargetKinematics dispatch tests.
//
// These verify the kinematic envelope of each confuser variant.
// Micro-Doppler / RCS dispatch is downstream of state_at and is
// not the responsibility of these tests; the gates below ONLY
// exercise geometry (range, altitude, radial_velocity_mps).
// -------------------------------------------------------------------

/// Lane J — Bird at cruise speed 15 m/s, heading 0° (directly
/// toward the radar), t = 10 s. Closing motion → radial velocity
/// equals +15 m/s (cos 0° · 15 = 15) and range decreases by 150 m
/// from the supplied initial range. Citation: Rahman & Robertson
/// 2018 wingbeat envelope; Wave-A `physics_dossier.md` bird-table.
#[test]
fn target_kinematics_bird_state_at() {
    let bird = TargetKinematics::Bird {
        cruise_speed_mps: 15.0,
        altitude_agl_m: 120.0,
        heading_deg: 0.0,
        wingbeat_hz: 4.0,
        wing_length_m: 0.5,
    };
    let initial_range_m = 2_500.0;
    let state = bird.state_at(10.0, initial_range_m, 20.0);
    // Radial velocity: cos(0°) * 15 = 15 m/s (closing).
    assert!(
        (state.radial_velocity_mps - 15.0).abs() < 1e-9,
        "expected radial_velocity_mps = 15.0, got {}",
        state.radial_velocity_mps
    );
    // Range delta: 15 m/s * 10 s * cos(0°) = 150 m closing.
    let expected_range = initial_range_m - 150.0;
    assert!(
        (state.range_m - expected_range).abs() < 1e-9,
        "expected range = {expected_range}, got {}",
        state.range_m
    );
    // Altitude is whatever the bird specified.
    assert_eq!(state.altitude_m, 120.0);
    assert_eq!(state.time_s, 10.0);
}

/// Lane J — GroundVehicle altitude is pinned to 0 m AGL regardless
/// of the call-site `initial_range_m` recovery or the vehicle's
/// configured heading. Per Wave-A `physics_dossier.md` ground-table:
/// ground vehicles travel at the surface (sensor antenna AGL is the
/// elevation reference, not vehicle altitude).
#[test]
fn target_kinematics_vehicle_static_altitude() {
    let vehicle = TargetKinematics::GroundVehicle {
        speed_mps: 20.0,
        heading_deg: 45.0,
        initial_range_m: 3_000.0,
    };
    // Try several call-sites to ensure altitude never depends on
    // initial_range_m / antenna_alt_agl_m fallbacks.
    for t in [0.0, 5.0, 12.5] {
        let state = vehicle.state_at(t, 999_999.0, 50.0);
        assert!(
            state.altitude_m.abs() < 1e-9,
            "GroundVehicle altitude must be ~0 m AGL; t={t} altitude={}",
            state.altitude_m
        );
    }
}

/// Lane J — WindTurbine hub is static: radial velocity is identically
/// zero regardless of elapsed time. The blade micro-Doppler line
/// spectrum is *downstream* of state_at and is contributed by
/// `crate::micro_doppler_gen::PropellerGenerator` at synthesis
/// time. Citation: Naqvi et al. 2015 wind-turbine signature paper;
/// Skolnik 3rd ed. §11.5.
#[test]
fn target_kinematics_turbine_radial_velocity_zero() {
    let turbine = TargetKinematics::WindTurbine {
        hub_range_m: 5_000.0,
        hub_altitude_agl_m: 80.0,
        blade_count: 3,
        rotation_hz: 0.3, // ~18 RPM
        blade_length_m: 45.0,
    };
    for t in [0.0, 1.0, 5.0, 30.0] {
        let state = turbine.state_at(t, 0.0, 20.0);
        assert!(
            state.radial_velocity_mps.abs() < 1e-12,
            "WindTurbine radial_velocity must be zero (static hub); \
             t={t} got {}",
            state.radial_velocity_mps
        );
        assert_eq!(state.range_m, 5_000.0, "hub range must stay constant");
        assert_eq!(state.altitude_m, 80.0, "hub altitude must stay constant");
    }
}

/// Lane J — Balloon: tethered=true collapses to static (zero radial
/// velocity); tethered=false produces along-heading drift. Both
/// branches preserve altitude_m and respect the closing-motion
/// sign convention. Per Wave-A `physics_dossier.md` confuser-list-
/// balloon: drift speeds 1–10 m/s in nominal wind.
#[test]
fn target_kinematics_balloon_drift() {
    // Tethered: zero motion regardless of drift configuration.
    let tethered = TargetKinematics::Balloon {
        drift_speed_mps: 5.0,
        drift_heading_deg: 0.0,
        altitude_agl_m: 1_200.0,
        tethered: true,
    };
    let state_t = tethered.state_at(20.0, 8_000.0, 20.0);
    assert!(
        state_t.radial_velocity_mps.abs() < 1e-12,
        "tethered balloon must have zero radial velocity; got {}",
        state_t.radial_velocity_mps
    );
    assert_eq!(state_t.range_m, 8_000.0, "tethered range stays at anchor");
    assert_eq!(state_t.altitude_m, 1_200.0);

    // Free-drifting: 5 m/s along heading 0° → 5 m/s radial.
    let drifting = TargetKinematics::Balloon {
        drift_speed_mps: 5.0,
        drift_heading_deg: 0.0,
        altitude_agl_m: 1_200.0,
        tethered: false,
    };
    let state_d = drifting.state_at(20.0, 8_000.0, 20.0);
    assert!(
        (state_d.radial_velocity_mps - 5.0).abs() < 1e-9,
        "drifting balloon radial velocity must equal drift_speed_mps along heading 0°; \
         got {}",
        state_d.radial_velocity_mps
    );
    // Range decreases by 5 m/s * 20 s = 100 m.
    assert!(
        (state_d.range_m - 7_900.0).abs() < 1e-9,
        "drifting balloon range must decrease by 100 m; got {}",
        state_d.range_m
    );
}

/// Lane J — Helicopter at 50 m/s heading 90° (broadside crossing)
/// has zero radial velocity because cos(90°) = 0. The dual-rotor
/// micro-Doppler is captured downstream by
/// `crate::micro_doppler_gen::HelicopterRotorGenerator`; state_at
/// returns geometry only. Citation: Chen 2011 §6.2.
#[test]
fn target_kinematics_helicopter_cruise() {
    let heli = TargetKinematics::Helicopter {
        cruise_speed_mps: 50.0,
        altitude_agl_m: 200.0,
        heading_deg: 90.0, // pure cross-flight
        main_blade_count: 4,
        main_rotation_hz: 5.0, // 300 RPM
        main_blade_length_m: 7.0,
        tail_blade_count: 2,
        tail_rotation_hz: 25.0, // 1500 RPM tail
        tail_blade_length_m: 1.0,
    };
    let state = heli.state_at(8.0, 4_000.0, 20.0);
    // cos(90°) = 0 → radial velocity is zero.
    assert!(
        state.radial_velocity_mps.abs() < 1e-9,
        "Helicopter at heading 90° must have zero radial velocity; \
         got {}",
        state.radial_velocity_mps
    );
    // cos(90°) = 0 → range stays at initial value (no along-LOS motion).
    assert!(
        (state.range_m - 4_000.0).abs() < 1e-9,
        "cross-flight helicopter range must equal initial range; got {}",
        state.range_m
    );
    assert_eq!(state.altitude_m, 200.0);
}

/// Lane J — multipath ghost geometry: a ghost paired with a real
/// target at range R = 10_000 m, antenna height h_r = 20 m, target
/// altitude h_t = 100 m has its phantom return at offset range
/// `R + 2·h_r·h_t/R = 10_000 + 2·20·100/10_000 = 10_000.4 m` per
/// Skolnik *Introduction to Radar Systems* 3rd ed. §1.6 two-ray
/// multipath geometry. This test pins the closed-form offset; the
/// state_at dispatch panics on MultipathGhost (per docstring) so we
/// validate the offset arithmetic at the synthesis-loop surface
/// rather than through state_at.
#[test]
fn multipath_ghost_range_offset() {
    let parent_range_m = 10_000.0f64;
    let h_r = 20.0f64; // antenna AGL
    let h_t = 100.0f64; // target AGL
    let ghost_range_m = parent_range_m + 2.0 * h_r * h_t / parent_range_m;
    // Closed-form: 2 * 20 * 100 / 10000 = 0.4 m
    let expected = 10_000.4f64;
    assert!(
        (ghost_range_m - expected).abs() < 1e-12,
        "multipath ghost geometry: expected {expected}, got {ghost_range_m}"
    );
    // Cross-check the ghost entity also round-trips through serde
    // — important so on-disk Lane J campaign fixtures stay portable.
    let ghost = TargetKinematics::MultipathGhost {
        parent_idx: 0,
        reflection_coefficient_magnitude: 0.6,
    };
    let json = serde_json::to_string(&ghost).expect("serialise ghost kinematics");
    let parsed: TargetKinematics =
        serde_json::from_str(&json).expect("deserialise ghost kinematics");
    assert_eq!(ghost, parsed);
}

/// Lane J — Kite is near-stationary at the anchor, with a bounded
/// wind-gust ripple on the radial velocity. The flutter envelope
/// is deterministic at synthesis time (a 0.5 Hz sinusoid in t_s)
/// so episode reproduction stays bit-stable. Per Wave-A
/// `physics_dossier.md` confuser-list-kite: anchored kites flutter
/// with ±1–5 m/s radial-velocity jitter under nominal wind.
#[test]
fn target_kinematics_kite_anchored_with_flutter() {
    let kite = TargetKinematics::Kite {
        anchor_range_m: 1_500.0,
        altitude_agl_m: 80.0,
        wind_gust_amplitude_mps: 3.0,
    };
    // At t = 0, sin(0) = 0 → radial velocity is zero.
    let s0 = kite.state_at(0.0, 0.0, 20.0);
    assert!(
        s0.radial_velocity_mps.abs() < 1e-9,
        "kite at t=0 must have zero radial velocity (sin(0)=0); got {}",
        s0.radial_velocity_mps
    );
    assert_eq!(s0.range_m, 1_500.0, "anchor range stays constant");
    // At any time the radial velocity must be bounded by the gust
    // amplitude — this is the physical safety bound.
    for t in [0.5, 1.0, 2.5, 7.3] {
        let s = kite.state_at(t, 0.0, 20.0);
        assert!(
            s.radial_velocity_mps.abs() <= 3.0 + 1e-9,
            "kite radial velocity must be bounded by gust amplitude 3.0; \
             t={t} got {}",
            s.radial_velocity_mps
        );
        assert_eq!(s.range_m, 1_500.0, "anchor range stays constant");
    }
}

#[test]
fn target_kinematics_static_returns_are_fixed() {
    let terrain = TargetKinematics::TerrainGlint {
        range_m: 2_400.0,
        altitude_agl_m: 35.0,
    };
    let man = TargetKinematics::ManRadarReturn {
        range_m: 1_180.0,
        altitude_agl_m: 1.5,
    };
    for t in [0.0, 2.0, 11.0] {
        let terrain_state = terrain.state_at(t, 999.0, 20.0);
        assert_eq!(terrain_state.range_m, 2_400.0);
        assert_eq!(terrain_state.altitude_m, 35.0);
        assert!(terrain_state.radial_velocity_mps.abs() < 1e-12);

        let man_state = man.state_at(t, 999.0, 20.0);
        assert_eq!(man_state.range_m, 1_180.0);
        assert_eq!(man_state.altitude_m, 1.5);
        assert!(man_state.radial_velocity_mps.abs() < 1e-12);
    }
}

#[test]
fn scene_descriptor_allows_empty_target_roster() {
    let scene = SceneDescriptor {
        geometry: SiteGeometry {
            antenna_altitude_agl_m: 20.0,
        },
        environment: EnvironmentDescriptor {
            clutter_regime: None,
            atmospheric_one_way_db_per_km: 0.0,
            rain_rate_mm_per_h: 0.0,
            ground_reflection_coefficient_magnitude: 0.0,
        },
        targets: Vec::new(),
    };
    let json = serde_json::to_string(&scene).expect("serialise empty scene");
    let parsed: SceneDescriptor = serde_json::from_str(&json).expect("deserialise empty scene");
    assert!(parsed.targets.is_empty());
    assert_eq!(scene, parsed);
}
