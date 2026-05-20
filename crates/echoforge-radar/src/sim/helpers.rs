use crate::micro_doppler_gen::{MicroDopplerGenerator, PropellerGenerator};
use crate::scene::{TargetEntity, TargetKinematics};

use super::config::{NoiseProfile, TakeoffProfile};
use super::episode::{SplitMix64, TargetState};

pub(super) const C_M_PER_S: f64 = 299_792_458.0;

/// Wave 5 Lane J helper — extract a sensible recovery initial range
/// for an entity. Variants that carry their own range (GroundVehicle,
/// WindTurbine, Kite) return that range; variants without one
/// (Bird, Helicopter, Balloon, FromTakeoffProfile) fall back to the
/// TakeoffProfile default `initial_range_m`. Used by the per-entity
/// initial-state pass so multipath ghost parents can be re-evaluated
/// before the synthesis loop runs.
pub(super) fn entity_initial_range_recovery(entity: &TargetEntity) -> f64 {
    match &entity.kinematics {
        TargetKinematics::FromTakeoffProfile(p) => p.initial_range_m,
        TargetKinematics::GroundVehicle {
            initial_range_m, ..
        } => *initial_range_m,
        TargetKinematics::WindTurbine { hub_range_m, .. } => *hub_range_m,
        TargetKinematics::Kite { anchor_range_m, .. } => *anchor_range_m,
        // Bird / Helicopter / Balloon / MultipathGhost don't carry a
        // range; use the TakeoffProfile default so first-entity geometry
        // is well-defined when the scene mixes types. Downstream Lane K
        // adds per-entity initial range plumbing on these variants.
        TargetKinematics::Bird { .. }
        | TargetKinematics::Helicopter { .. }
        | TargetKinematics::Balloon { .. }
        | TargetKinematics::MultipathGhost { .. } => TakeoffProfile::default().initial_range_m,
    }
}

/// Wave 5 Lane J helper — resolve a per-pulse `(state, range_offset_m)`
/// pair for entity `idx` at time `t_s`. Multipath ghost entities return
/// the parent's state and a non-zero range offset computed from the
/// two-ray geometry `2·h_r·h_t/R` (Skolnik 3rd ed. §1.6); other
/// entities return their own state and zero offset.
pub(super) fn resolve_entity_state(
    targets: &[TargetEntity],
    idx: usize,
    t_s: f64,
    antenna_alt_agl_m: f64,
) -> (TargetState, f64) {
    let entity = &targets[idx];
    match &entity.kinematics {
        TargetKinematics::MultipathGhost { parent_idx, .. } => {
            let parent = targets.get(*parent_idx).unwrap_or(entity);
            let parent_initial_range = entity_initial_range_recovery(parent);
            let parent_state = parent.kinematics.state_at(
                t_s,
                parent_initial_range,
                antenna_alt_agl_m,
            );
            // Two-ray multipath offset: 2·h_r·h_t/R. Guard against
            // R = 0 by clamping the range to a small positive value.
            let r = parent_state.range_m.max(1e-3);
            let offset = 2.0 * antenna_alt_agl_m * parent_state.altitude_m / r;
            (parent_state, offset)
        }
        _ => {
            let initial_range = entity_initial_range_recovery(entity);
            let state = entity
                .kinematics
                .state_at(t_s, initial_range, antenna_alt_agl_m);
            (state, 0.0)
        }
    }
}

/// Wave 5 Lane J helper — per-entity micro-Doppler envelope at time
/// `t_s`. Returns the multiplicative amplitude modulation that goes
/// onto the entity's per-pulse return.
///
/// For entities carrying a [`TakeoffProfile`] (the bridged Shahed
/// path), this routes through the prior multi-blade / single-sinusoid
/// dispatch from [`crate::micro_doppler_gen::PropellerGenerator`] so
/// pre-Lane-J reproduction fixtures replay byte-identically. For
/// confuser variants, the envelope returns 1.0 — the native
/// micro-Doppler line spectra are the responsibility of the downstream
/// [`crate::micro_doppler_gen`] generators (BirdWingbeatGenerator,
/// HelicopterRotorGenerator, PropellerGenerator) at the per-class
/// feature-extraction surface, not the per-pulse amplitude
/// modulation. Returning 1.0 at the synthesis surface is the
/// conservative first-order proxy that does not over-claim micro-
/// Doppler fidelity for confusers; full per-class AM modelling lands
/// in Lane K (per-class RCS-aspect + propulsion dispatch).
pub(super) fn micro_doppler_envelope(
    entity: &TargetEntity,
    t_s: f64,
    state: &TargetState,
    recovery_profile: &TakeoffProfile,
) -> f64 {
    let profile = match &entity.kinematics {
        TargetKinematics::FromTakeoffProfile(p) => p,
        _ => return 1.0,
    };
    // Defensive: when state and profile point to different entities
    // (cross-entity dispatch in mixed scenes), the profile must still
    // come from the entity itself. The recovery_profile is only
    // consulted to keep the type signature uniform; never used here.
    let _ = recovery_profile;
    match (profile.blade_count, profile.blade_length_m) {
        (Some(n_blades), Some(length_m)) => {
            let prop = PropellerGenerator::new(
                n_blades,
                profile.propulsor_hz,
                length_m,
                state.propulsor_phase_rad,
            );
            let v_micro = prop.radial_velocity_at(t_s);
            let v_tip = prop.tip_speed_mps();
            if v_tip > 1e-6 {
                1.0 + 0.15 * (v_micro / v_tip)
            } else {
                1.0
            }
        }
        _ => {
            1.0 + 0.15
                * (2.0 * std::f64::consts::PI * profile.micro_doppler_hz * t_s
                    + state.propulsor_phase_rad)
                    .sin()
        }
    }
}

/// Wave 5 Lane J helper — per-class first-order RCS scalar (linear m²).
///
/// This is a **first-order proxy** keyed on Wave-A `physics_dossier.md`
/// confuser median RCS values; full per-class aspect-dependent RCS
/// lookup against `crate::rcs::Rcs::seeded_public_proxy_v1` lands in
/// Lane K. The proxy is sufficient for the Lane J multi-class
/// dispatch gate: each entity gets a non-zero contribution whose
/// magnitude tracks the class's typical RCS envelope, and the
/// emergent SNR list distinguishes positives from confusers.
///
/// Reference RCS values (dBsm → linear m² via 10^(dBsm/10)):
///   - ShahedClassPiston / ShahedClassJet → use the entity's
///     TakeoffProfile.rcs_scalar (already in linear m²).
///   - Bird → -25 dBsm (single large bird; Rahman & Robertson 2018).
///   - GroundVehicle → +5 dBsm (car / SUV at broadside aspect).
///   - WindTurbine → +25 dBsm (large utility-scale tower; Naqvi 2015).
///   - Balloon → -20 dBsm (Mylar reflector envelope).
///   - Kite → -25 dBsm (typical tethered kite).
///   - Helicopter → +5 dBsm (rotary-wing aircraft at typical aspect).
///   - MultipathGhost / TerrainGlint / ManRadarReturn → first-order
///     recovery at -30 dBsm (these are highly geometry-dependent).
pub(super) fn class_default_rcs_scalar(
    class: &crate::scene::TargetClass,
    kinematics: &TargetKinematics,
    recovery_profile: &TakeoffProfile,
) -> f64 {
    let _ = recovery_profile;
    // First, give FromTakeoffProfile entities their explicit RCS so
    // pre-Lane-J fixtures byte-match (their rcs_scalar is the
    // load-bearing input to the link budget).
    if let TargetKinematics::FromTakeoffProfile(profile) = kinematics {
        return profile.rcs_scalar;
    }
    use crate::scene::TargetClass as TC;
    let dbsm: f64 =
        if matches!(class, TC::ShahedClassPiston | TC::ShahedClassJet) { -10.0 }
        else if matches!(class, TC::Bird | TC::Kite) { -25.0 }
        else if matches!(class, TC::GroundVehicle | TC::Helicopter) { 5.0 }
        else if matches!(class, TC::WindTurbine) { 25.0 }
        else if matches!(class, TC::Balloon) { -20.0 }
        else { -30.0 }; // MultipathGhost, TerrainGlint, ManRadarReturn
    10f64.powf(dbsm / 10.0)
}

pub(super) fn build_ground_glints(
    sample_count: usize,
    noise: &NoiseProfile,
    rng: &mut SplitMix64,
) -> Vec<(usize, f32)> {
    if sample_count == 0 {
        return Vec::new();
    }

    (0..noise.ground_glint_count)
        .map(|_| {
            let bin = (rng.unit_f32() * sample_count as f32) as usize;
            let amp = noise.ground_glint_amplitude * (0.5 + rng.unit_f32());
            (bin.min(sample_count - 1), amp)
        })
        .collect()
}

pub(super) fn integrate_profiles(profiles: &[Vec<f32>], len: usize) -> Vec<f32> {
    if profiles.is_empty() {
        return Vec::new();
    }

    let mut integrated = vec![0.0f32; len];
    for profile in profiles {
        for (index, value) in profile.iter().enumerate().take(len) {
            integrated[index] += *value;
        }
    }
    let scale = 1.0 / profiles.len() as f32;
    for value in &mut integrated {
        *value *= scale;
    }
    integrated
}

pub fn range_bin_to_m(index: usize, zero_delay_bin: usize, sample_rate_hz: f64) -> f64 {
    let delay = index as isize - zero_delay_bin as isize;
    if delay <= 0 {
        0.0
    } else {
        (delay as f64) * C_M_PER_S / (2.0 * sample_rate_hz)
    }
}
