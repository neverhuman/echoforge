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
        rcs_scalar: 10f64.powf(envelope.rcs_dbsm / 20.0).max(0.01),
        blade_count: None,
        blade_length_m: None,
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
    profile: TakeoffProfile,
    config: &RadarSimConfig,
    noise: &NoiseProfile,
) -> SceneDescriptor {
    let target_class = confuser_class_for_family(
        class.hard_negative_family.as_str(),
        class.is_public_proxy_positive,
    );
    SceneDescriptor::from_radar_config(
        config,
        noise,
        vec![TargetEntity {
            class: target_class,
            kinematics: TargetKinematics::FromTakeoffProfile(profile),
            spawn_time_s: 0.0,
        }],
    )
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
