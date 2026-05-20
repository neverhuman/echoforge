//! Scene descriptor construction and confuser class mapping for the unified
//! physics path (Wave 5 Lane K_rust).

use echoforge_radar::{
    NoiseProfile, RadarSimConfig, SceneDescriptor, TakeoffProfile, TargetClass, TargetEntity,
    TargetKinematics,
};

use super::types::{MlClass, MlEnvelope, SplitMix64};

/// Adapt an envelope sample to a [`TakeoffProfile`]. This is the
/// per-class kinematics bridge Lane K_rust uses until Lane J lands
/// native confuser kinematics (constant-velocity birds, parked
/// vehicles, stationary turbines, etc.). It carries the envelope's
/// range / speed / micro-Doppler / RCS into the unified physics path
/// regardless of class so the radar chain produces a single coherent
/// episode per record.
pub(super) fn adapt_envelope_to_takeoff_profile(
    envelope: &MlEnvelope,
    rng: &mut SplitMix64,
) -> TakeoffProfile {
    TakeoffProfile {
        initial_range_m: envelope.initial_range_m,
        runway_heading_deg: rng.range_f64(-18.0, 18.0),
        ground_speed_mps: envelope.speed_mps,
        acceleration_mps2: rng.range_f64(0.0, 0.9),
        climb_rate_mps: rng.range_f64(0.0, 3.5),
        max_altitude_m: envelope.altitude_m.max(1.0),
        radial_velocity_bias_mps: envelope.radial_velocity_mps,
        pitch_jitter_deg: rng.range_f64(0.2, 4.0),
        yaw_jitter_deg: rng.range_f64(0.2, 4.0),
        propulsor_hz: envelope.micro_peak_hz as f64,
        micro_doppler_hz: envelope.micro_peak_hz as f64,
        rcs_scalar: 10f64.powf(envelope.rcs_dbsm / 10.0).max(0.01),
        blade_count: None,
        blade_length_m: None,
    }
}

fn bird_entity(
    envelope: &MlEnvelope,
    heading_deg: f64,
    wingbeat_scale: f64,
    range_scale: f64,
    wing_length_m: f64,
) -> TargetEntity {
    TargetEntity {
        class: TargetClass::Bird,
        kinematics: TargetKinematics::Bird {
            cruise_speed_mps: (0.65 * envelope.speed_mps * range_scale).clamp(5.0, 42.0),
            altitude_agl_m: envelope.altitude_m.max(1.0),
            heading_deg,
            wingbeat_hz: (envelope.micro_peak_hz as f64 * wingbeat_scale).clamp(2.0, 16.0),
            wing_length_m,
        },
        spawn_time_s: 0.0,
    }
}

fn static_entity(class: TargetClass, kinematics: TargetKinematics) -> TargetEntity {
    TargetEntity {
        class,
        kinematics,
        spawn_time_s: 0.0,
    }
}

fn build_scene_entities(
    class: &MlClass,
    envelope: &MlEnvelope,
    rng: &mut SplitMix64,
) -> Vec<TargetEntity> {
    match class.hard_negative_family.as_str() {
        "positive_public_proxy" => vec![static_entity(
            TargetClass::ShahedClassPiston,
            TargetKinematics::FromTakeoffProfile(adapt_envelope_to_takeoff_profile(envelope, rng)),
        )],
        "single_bird" => {
            let heading = rng.range_f64(-20.0, 20.0);
            let wing_length = 0.18 + 0.22 * rng.unit_f64();
            vec![bird_entity(
                envelope,
                heading,
                0.55,
                1.0,
                wing_length.max(0.05),
            )]
        }
        "bird_flock" => {
            let heading = rng.range_f64(-28.0, 28.0);
            vec![
                bird_entity(
                    envelope,
                    heading - 6.0,
                    0.75,
                    0.92,
                    (0.16 + 0.14 * rng.unit_f64()).max(0.05),
                ),
                bird_entity(
                    envelope,
                    heading + 4.0,
                    0.82,
                    1.00,
                    (0.16 + 0.14 * rng.unit_f64()).max(0.05),
                ),
                bird_entity(
                    envelope,
                    heading + 12.0,
                    0.68,
                    1.06,
                    (0.16 + 0.14 * rng.unit_f64()).max(0.05),
                ),
            ]
        }
        "bat_insect_cloud" => {
            let heading = rng.range_f64(-34.0, 34.0);
            vec![
                bird_entity(
                    envelope,
                    heading - 8.0,
                    1.35,
                    0.80,
                    (0.06 + 0.08 * rng.unit_f64()).max(0.03),
                ),
                bird_entity(
                    envelope,
                    heading + 3.0,
                    1.55,
                    0.88,
                    (0.06 + 0.08 * rng.unit_f64()).max(0.03),
                ),
                bird_entity(
                    envelope,
                    heading + 11.0,
                    1.80,
                    0.96,
                    (0.06 + 0.08 * rng.unit_f64()).max(0.03),
                ),
                bird_entity(
                    envelope,
                    heading + 18.0,
                    1.95,
                    1.04,
                    (0.06 + 0.08 * rng.unit_f64()).max(0.03),
                ),
            ]
        }
        "balloon_weather" => {
            let drift_heading_deg = rng.range_f64(-180.0, 180.0);
            vec![static_entity(
                TargetClass::Balloon,
                TargetKinematics::Balloon {
                    drift_speed_mps: envelope.speed_mps.clamp(0.0, 8.0),
                    drift_heading_deg,
                    altitude_agl_m: envelope.altitude_m.max(10.0),
                    tethered: false,
                },
            )]
        }
        "kite" => vec![static_entity(
            TargetClass::Kite,
            TargetKinematics::Kite {
                anchor_range_m: envelope.initial_range_m,
                altitude_agl_m: envelope.altitude_m.max(5.0),
                wind_gust_amplitude_mps: envelope.speed_mps.clamp(0.5, 8.0),
            },
        )],
        "windborne_debris" => {
            let drift_heading_deg = rng.range_f64(-180.0, 180.0);
            vec![static_entity(
                TargetClass::Balloon,
                TargetKinematics::Balloon {
                    drift_speed_mps: envelope.speed_mps.clamp(0.0, 18.0),
                    drift_heading_deg,
                    altitude_agl_m: envelope.altitude_m.max(10.0),
                    tethered: false,
                },
            )]
        }
        "ground_vehicle" => {
            let heading_deg = rng.range_f64(-35.0, 35.0);
            vec![static_entity(
                TargetClass::GroundVehicle,
                TargetKinematics::GroundVehicle {
                    speed_mps: envelope.speed_mps.clamp(0.0, 35.0),
                    heading_deg,
                    initial_range_m: envelope.initial_range_m,
                },
            )]
        }
        "power_line_pylon" => vec![static_entity(
            TargetClass::TerrainGlint,
            TargetKinematics::TerrainGlint {
                range_m: envelope.initial_range_m,
                altitude_agl_m: envelope.altitude_m.max(0.0),
            },
        )],
        "wind_turbine" => {
            let rotation_hz = (0.18 + 0.22 * rng.unit_f64()).clamp(0.12, 0.45);
            vec![static_entity(
                TargetClass::WindTurbine,
                TargetKinematics::WindTurbine {
                    hub_range_m: envelope.initial_range_m,
                    hub_altitude_agl_m: envelope.altitude_m.max(20.0),
                    blade_count: 3,
                    rotation_hz,
                    blade_length_m: envelope.dimensions_m.wingspan.max(15.0),
                },
            )]
        }
        "rain_cell" | "dust_haze" | "rfi_burst" | "terrain_only" => Vec::new(),
        "multipath_ghost" => {
            let parent_profile = adapt_envelope_to_takeoff_profile(envelope, rng);
            vec![
                static_entity(
                    TargetClass::ShahedClassPiston,
                    TargetKinematics::FromTakeoffProfile(parent_profile),
                ),
                static_entity(
                    TargetClass::MultipathGhost { parent_idx: 0 },
                    TargetKinematics::MultipathGhost {
                        parent_idx: 0,
                        reflection_coefficient_magnitude: (0.45 + 0.35 * rng.unit_f64())
                            .clamp(0.1, 0.95),
                    },
                ),
            ]
        }
        _ => vec![static_entity(
            TargetClass::TerrainGlint,
            TargetKinematics::TerrainGlint {
                range_m: envelope.initial_range_m,
                altitude_agl_m: envelope.altitude_m.max(0.0),
            },
        )],
    }
}

/// Map a confuser family to the appropriate Lane I
/// [`TargetClass`] variant. When a family naturally maps to multiple
/// entities (e.g. `multipath_ghost` wants a parent + ghost pair) we
/// document the gap as a Lane J pending item and recover to a single-entity
/// scene with the closest static-confuser variant. Lane J reconciles.
pub(super) fn confuser_class_for_family(family: &str, is_positive: bool) -> TargetClass {
    if is_positive {
        return TargetClass::ShahedClassPiston;
    }
    match family {
        "single_bird" | "bird_flock" => TargetClass::Bird,
        // Lane J: bats and insect clouds want a `Bird`-like
        // variant with a faster wingbeat micro-Doppler envelope. Lane
        // J adds a dedicated class; for now we reuse `Bird` because
        // the kinematics envelope is similar enough for the
        // single-entity bridge.
        "bat_insect_cloud" => TargetClass::Bird,
        "balloon_weather" => TargetClass::Balloon,
        "kite" => TargetClass::Kite,
        // Lane J: windborne debris (Mylar reflectors, plastic
        // bag debris) deserves its own class. For now use Balloon as
        // the closest slow-windborne archetype.
        "windborne_debris" => TargetClass::Balloon,
        "ground_vehicle" => TargetClass::GroundVehicle,
        // Static infrastructure mapped to `TerrainGlint`; wind
        // turbines get the dedicated variant.
        "power_line_pylon" => TargetClass::TerrainGlint,
        "wind_turbine" => TargetClass::WindTurbine,
        // Lane J: weather (rain, dust, RFI) and pure-terrain
        // scenes don't have a target entity at all in the strict
        // sense; they are clutter/noise stressors. The Lane I
        // single-entity gate forces us to put SOMETHING here, so we
        // use `TerrainGlint` as a static pending entry. Lane J should
        // allow zero-target scenes with environment-only physics for
        // these families.
        "rain_cell" | "dust_haze" | "rfi_burst" | "terrain_only" => TargetClass::TerrainGlint,
        // Lane J: multipath ghosts want a paired entity with
        // `TargetClass::MultipathGhost { parent_idx: 0 }`. Lane I's
        // single-entity assertion (debug_assert in synthesize_scene)
        // blocks this today. Recover to a static-ground proxy so
        // the unified path still runs; Lane J will land the paired
        // entity wiring.
        "multipath_ghost" => TargetClass::GroundVehicle,
        _ => TargetClass::TerrainGlint,
    }
}

/// Build a single-entity [`SceneDescriptor`] for the unified physics
/// path. Lane I (`synthesize_scene`) only honours a single entity with
/// `TargetKinematics::FromTakeoffProfile`; Lane J extends this. The
/// `geometry` / `environment` fields are kept in sync with `config` /
/// `noise` so byte-stable backward-compatible with pre-Lane-I fixtures is
/// preserved.
pub(super) fn build_scene_descriptor(
    class: &MlClass,
    envelope: &MlEnvelope,
    rng: &mut SplitMix64,
    config: &RadarSimConfig,
    noise: &NoiseProfile,
) -> SceneDescriptor {
    SceneDescriptor::from_radar_config(config, noise, build_scene_entities(class, envelope, rng))
}

/// Build the noise profile for an envelope — shared by the production worker
/// and the test helper so both use identical physics parameters.
pub(super) fn build_noise_profile(envelope: &MlEnvelope) -> NoiseProfile {
    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.awgn_sigma = (0.035 + 0.05 * envelope.clutter_pressure) as f32;
    noise.clutter_sigma = (0.02 + 0.09 * envelope.clutter_pressure) as f32;
    noise.rfi_probability = (0.004 + 0.045 * envelope.rfi_pressure).min(0.12);
    noise.rfi_amplitude = 0.55 + 1.35 * envelope.rfi_pressure;
    noise.amplitude_scintillation_sigma = envelope.amplitude_impairment.max(0.01);
    noise.phase_noise_std_rad = envelope.phase_impairment_rad.max(0.002);
    noise.ground_glint_count = (2.0 + 10.0 * envelope.clutter_pressure).round() as usize;
    noise
}
