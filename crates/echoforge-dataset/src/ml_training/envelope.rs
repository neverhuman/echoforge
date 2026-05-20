//! Envelope sampling: maps per-class family names to MlEnvelope instances.

use super::types::{DimensionsSample, MlClass, MlEnvelope, SplitMix64};

/// All `[min, max]` range pairs needed to sample one `MlEnvelope`.
/// Use `[v, v]` for a deterministic (non-random) field.
#[derive(Clone, Copy)]
struct EnvelopeSpec {
    dim_length: [f64; 2],
    dim_wingspan: [f64; 2],
    dim_height: [f64; 2],
    rcs_dbsm: [f64; 2],
    speed_mps: [f64; 2],
    initial_range_m: [f64; 2],
    radial_velocity_mps: [f64; 2],
    altitude_m: [f64; 2],
    base_snr_db: [f32; 2],
    micro_peak_hz: [f32; 2],
    micro_bandwidth_hz: [f32; 2],
    clutter_pressure: [f32; 2],
    rfi_pressure: [f32; 2],
    dropout_probability: [f32; 2],
    phase_impairment_rad: [f32; 2],
    amplitude_impairment: [f32; 2],
}

fn build_envelope(spec: &EnvelopeSpec, rng: &mut SplitMix64) -> MlEnvelope {
    MlEnvelope {
        dimensions_m: DimensionsSample {
            length: rng.range_f64(spec.dim_length[0], spec.dim_length[1]),
            wingspan: rng.range_f64(spec.dim_wingspan[0], spec.dim_wingspan[1]),
            height: rng.range_f64(spec.dim_height[0], spec.dim_height[1]),
        },
        rcs_dbsm: rng.range_f64(spec.rcs_dbsm[0], spec.rcs_dbsm[1]),
        speed_mps: rng.range_f64(spec.speed_mps[0], spec.speed_mps[1].max(spec.speed_mps[0] + 1.0)),
        initial_range_m: rng.range_f64(spec.initial_range_m[0], spec.initial_range_m[1]),
        radial_velocity_mps: rng.range_f64(spec.radial_velocity_mps[0], spec.radial_velocity_mps[1]),
        altitude_m: rng.range_f64(spec.altitude_m[0], spec.altitude_m[1]),
        base_snr_db: rng.range_f32(spec.base_snr_db[0], spec.base_snr_db[1]),
        micro_peak_hz: rng.range_f32(spec.micro_peak_hz[0], spec.micro_peak_hz[1]),
        micro_bandwidth_hz: rng.range_f32(spec.micro_bandwidth_hz[0], spec.micro_bandwidth_hz[1]),
        clutter_pressure: rng.range_f32(spec.clutter_pressure[0], spec.clutter_pressure[1]),
        rfi_pressure: rng.range_f32(spec.rfi_pressure[0], spec.rfi_pressure[1]),
        dropout_probability: rng.range_f32(spec.dropout_probability[0], spec.dropout_probability[1]),
        phase_impairment_rad: rng.range_f32(spec.phase_impairment_rad[0], spec.phase_impairment_rad[1]),
        amplitude_impairment: rng.range_f32(spec.amplitude_impairment[0], spec.amplitude_impairment[1]),
    }
}

// ── Per-class base specs (constant fields; variable fields overridden at call site) ──

// Phase-1.5 envelope harmonization: see sample_envelope for the full rationale.

const POSITIVE_SPEC: EnvelopeSpec = EnvelopeSpec {
    dim_length: [2.4, 3.9], dim_wingspan: [1.8, 2.9], dim_height: [0.30, 0.85],
    rcs_dbsm: [-30.0, -2.0], speed_mps: [8.0, 70.0],
    initial_range_m: [600.0, 8_500.0], radial_velocity_mps: [-55.0, 15.0], altitude_m: [10.0, 800.0],
    base_snr_db: [-3.0, 14.0], micro_peak_hz: [35.0, 130.0], micro_bandwidth_hz: [30.0, 150.0],
    clutter_pressure: [0.10, 0.55], rfi_pressure: [0.02, 0.28],
    dropout_probability: [0.0, 0.06], phase_impairment_rad: [0.006, 0.045], amplitude_impairment: [0.04, 0.18],
};

// Phase-1.5: widened micro-Doppler — raptors at 60-80 Hz, bats at 80-120 Hz, insect clouds at 150-400 Hz.
const BIOLOGICAL_BASE: EnvelopeSpec = EnvelopeSpec {
    dim_length: [0.03, 1.2], dim_wingspan: [0.04, 2.4], dim_height: [0.01, 0.45],
    rcs_dbsm: [-48.0, -10.0], speed_mps: [0.0, 0.0],  // overridden per call
    initial_range_m: [300.0, 7_500.0], radial_velocity_mps: [-32.0, 32.0], altitude_m: [5.0, 1_100.0],
    base_snr_db: [-4.0, 13.0], micro_peak_hz: [0.0, 0.0],  // overridden per call
    micro_bandwidth_hz: [8.0, 140.0], clutter_pressure: [0.12, 0.55], rfi_pressure: [0.0, 0.22],
    dropout_probability: [0.0, 0.07], phase_impairment_rad: [0.008, 0.050], amplitude_impairment: [0.05, 0.20],
};

// Phase-1.5: balloon/kite/debris altitudes expanded to overlap positive UAS distribution.
const SLOW_WINDBORNE_BASE: EnvelopeSpec = EnvelopeSpec {
    dim_length: [0.2, 5.0], dim_wingspan: [0.2, 8.0], dim_height: [0.2, 5.0],
    rcs_dbsm: [0.0, 0.0], speed_mps: [0.0, 0.0],  // overridden per call
    initial_range_m: [300.0, 7_000.0], radial_velocity_mps: [-18.0, 18.0], altitude_m: [5.0, 2_000.0],
    base_snr_db: [-5.0, 12.0],
    // Phase-1.5: windborne objects flutter — kite tails, Mylar, debris show Doppler smearing.
    micro_peak_hz: [0.0, 40.0], micro_bandwidth_hz: [2.0, 35.0],
    clutter_pressure: [0.18, 0.58], rfi_pressure: [0.0, 0.25],
    dropout_probability: [0.0, 0.08], phase_impairment_rad: [0.01, 0.055], amplitude_impairment: [0.05, 0.22],
};

// Phase-1.5: trucks can have higher SNR than UAS; multipath_ghost altitude tail reaches UAS regime.
const GROUND_BASE: EnvelopeSpec = EnvelopeSpec {
    dim_length: [2.0, 14.0], dim_wingspan: [1.5, 3.5], dim_height: [1.0, 4.2],
    rcs_dbsm: [0.0, 0.0], speed_mps: [0.0, 0.0],  // overridden per call
    initial_range_m: [200.0, 7_500.0], radial_velocity_mps: [-50.0, 50.0], altitude_m: [0.0, 600.0],
    base_snr_db: [-2.0, 18.0],
    // Phase-1.5: vehicle drivetrain produces aliased micro-Doppler in UAS prop regime.
    micro_peak_hz: [2.0, 110.0], micro_bandwidth_hz: [8.0, 110.0],
    clutter_pressure: [0.30, 0.85], rfi_pressure: [0.02, 0.32],
    dropout_probability: [0.0, 0.10], phase_impairment_rad: [0.012, 0.070], amplitude_impairment: [0.08, 0.26],
};

// Phase-1.5: wind-turbine blade-tip Doppler aliases into ±50 m/s at X-band; range narrowed accordingly.
const INFRASTRUCTURE_BASE: EnvelopeSpec = EnvelopeSpec {
    dim_length: [5.0, 80.0], dim_wingspan: [2.0, 80.0], dim_height: [5.0, 160.0],
    rcs_dbsm: [0.0, 0.0], speed_mps: [0.0, 0.0],  // overridden per call
    initial_range_m: [600.0, 9_000.0], radial_velocity_mps: [-60.0, 60.0], altitude_m: [20.0, 1_500.0],
    base_snr_db: [-4.0, 16.0],
    // Phase-1.5: blade-tip Doppler aliases above UAS prop regime under PRF folding for X-band.
    micro_peak_hz: [0.0, 130.0], micro_bandwidth_hz: [4.0, 140.0],
    clutter_pressure: [0.45, 0.90], rfi_pressure: [0.02, 0.30],
    dropout_probability: [0.0, 0.09], phase_impairment_rad: [0.010, 0.060], amplitude_impairment: [0.08, 0.24],
};

// Phase-1.5: rain/dust/RFI cells extended to overlap UAS observable range.
const WEATHER_BASE: EnvelopeSpec = EnvelopeSpec {
    dim_length: [0.0, 0.0], dim_wingspan: [0.0, 0.0], dim_height: [0.0, 0.0],
    rcs_dbsm: [-45.0, -10.0], speed_mps: [0.0, 0.0],  // overridden per call
    initial_range_m: [100.0, 8_000.0], radial_velocity_mps: [-18.0, 18.0], altitude_m: [0.0, 2_500.0],
    base_snr_db: [-8.0, 12.0],
    // Phase-1.5: heavy-rain cores and RFI bursts smear arbitrary Doppler frequencies.
    micro_peak_hz: [0.0, 35.0], micro_bandwidth_hz: [0.5, 40.0],
    clutter_pressure: [0.0, 0.0], rfi_pressure: [0.0, 0.0],  // overridden per call (jittered base)
    dropout_probability: [0.02, 0.14], phase_impairment_rad: [0.018, 0.090], amplitude_impairment: [0.10, 0.32],
};

// ── Public entry point ────────────────────────────────────────────────────────

pub(super) fn sample_envelope(class: &MlClass, rng: &mut SplitMix64) -> MlEnvelope {
    match class.hard_negative_family.as_str() {
        "positive_public_proxy" => build_envelope(&POSITIVE_SPEC, rng),
        "single_bird" => build_envelope(&EnvelopeSpec { speed_mps: [1.0, 24.0], micro_peak_hz: [3.0, 90.0], ..BIOLOGICAL_BASE }, rng),
        "bird_flock" => build_envelope(&EnvelopeSpec { speed_mps: [4.0, 31.0], micro_peak_hz: [5.0, 130.0], ..BIOLOGICAL_BASE }, rng),
        "bat_insect_cloud" => build_envelope(&EnvelopeSpec { speed_mps: [0.5, 18.0], micro_peak_hz: [12.0, 180.0], ..BIOLOGICAL_BASE }, rng),
        "balloon_weather" => build_envelope(&EnvelopeSpec { speed_mps: [0.0, 7.0], rcs_dbsm: [-34.0, -8.0], ..SLOW_WINDBORNE_BASE }, rng),
        "kite" => build_envelope(&EnvelopeSpec { speed_mps: [0.2, 12.0], rcs_dbsm: [-30.0, -5.0], ..SLOW_WINDBORNE_BASE }, rng),
        "windborne_debris" => build_envelope(&EnvelopeSpec { speed_mps: [0.0, 18.0], rcs_dbsm: [-36.0, -6.0], ..SLOW_WINDBORNE_BASE }, rng),
        "ground_vehicle" => build_envelope(&EnvelopeSpec { speed_mps: [0.0, 32.0], rcs_dbsm: [-5.0, 18.0], ..GROUND_BASE }, rng),
        "power_line_pylon" => build_envelope(&EnvelopeSpec { speed_mps: [0.0, 8.0], rcs_dbsm: [8.0, 34.0], ..INFRASTRUCTURE_BASE }, rng),
        "wind_turbine" => build_envelope(&EnvelopeSpec { speed_mps: [0.0, 80.0], rcs_dbsm: [4.0, 28.0], ..INFRASTRUCTURE_BASE }, rng),
        "rain_cell" => weather_variant(rng, 0.0, 12.0, 0.62, 0.20),
        "dust_haze" => weather_variant(rng, 0.0, 10.0, 0.58, 0.12),
        "rfi_burst" => weather_variant(rng, 0.0, 8.0, 0.25, 0.82),
        "terrain_only" => weather_variant(rng, 0.0, 4.0, 0.72, 0.08),
        "multipath_ghost" => build_envelope(&EnvelopeSpec { speed_mps: [0.0, 25.0], rcs_dbsm: [-20.0, 12.0], ..GROUND_BASE }, rng),
        _ => build_envelope(&EnvelopeSpec { speed_mps: [0.0, 12.0], rcs_dbsm: [-35.0, -8.0], ..SLOW_WINDBORNE_BASE }, rng),
    }
}

/// Weather classes use a jittered-base approach for clutter/rfi pressure.
///
/// Pre-expanding [base−δ, base+δ] (with the actual parameter values all landing
/// within (0.02, 0.90)) is equivalent to `(base + range_f32(−δ, +δ)).clamp(0, 1)`
/// without probability-mass truncation at the bounds.
fn weather_variant(rng: &mut SplitMix64, speed_min: f64, speed_max: f64, clutter: f32, rfi: f32) -> MlEnvelope {
    build_envelope(&EnvelopeSpec {
        speed_mps: [speed_min, speed_max],
        clutter_pressure: [(clutter - 0.08).max(0.0), (clutter + 0.08).min(1.0)],
        rfi_pressure: [(rfi - 0.06).max(0.0), (rfi + 0.08).min(1.0)],
        ..WEATHER_BASE
    }, rng)
}
