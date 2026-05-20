//! Unified scene generator types — Lane I (Wave 4) of the Radar Expert
//! Credibility Sweep.
//!
//! Before this lane, the simulator carried two physics paths:
//!
//!   1. `synthesize_takeoff_episode(config, profile, noise, seed)` —
//!      the single positive-target physics generator in
//!      [`crate::sim`]. Drives windowed pulse compression, slow-time
//!      DFT, K/Weibull clutter, MTI/MTD, and OS-CFAR alpha repair
//!      end-to-end.
//!   2. Confuser-class envelope statistics in
//!      `echoforge-dataset/src/ml_training.rs::build_frame_products`,
//!      which generated bird / vehicle / turbine / multipath-ghost
//!      envelopes WITHOUT going through the chain in (1).
//!
//! That bifurcation meant the generator identity literally labelled
//! the class — a red flag to any radar engineer reviewing the dataset
//! card. Lane I introduces a single physics path:
//!
//!   `synthesize_scene(scene: SceneDescriptor, …) -> SyntheticEpisode`
//!
//! where ALL target classes (positives + confusers) traverse the same
//! chain. `synthesize_takeoff_episode` becomes a thin bridged
//! wrapper that constructs a one-entity `SceneDescriptor` with
//! [`TargetClass::ShahedClassPiston`] and
//! [`TargetKinematics::FromTakeoffProfile`] and forwards through the
//! unified path. Existing reproduction fixtures stay byte-stable.
//!
//! ## Scope of this lane
//!
//! Lane I landed the **types and the single-entity wiring**. **Wave 5
//! Lane J (this lane) extends the path** with native per-class
//! kinematics for confusers (Bird, GroundVehicle, WindTurbine,
//! Balloon, Kite, Helicopter, MultipathGhost) plus multi-entity
//! dispatch in [`crate::sim::synthesize_scene`]. Per-class kinematic
//! envelopes are cited to Lane J prep research; per-class RCS-aspect
//! and propulsion micro-Doppler dispatch remain a first-order proxy
//! at the synthesis surface — full RCS-aspect lookup against
//! [`crate::rcs::Rcs::seeded_public_proxy_v1`] and per-class
//! propulsion modelling land in Lane K.
//!
//! ## References
//!
//! - Skolnik, *Introduction to Radar Systems*, 3rd ed.
//! - Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.
//! - Rahman & Robertson, *Nature* 2018 (bird wingbeat envelopes).
//! - Chen, *Micro-Doppler Effect in Radar*, 2011 (rotor signatures).
//! - Internal: `.agents/receipts/realism-v4-red-team-gap-report/*` —
//!   the original bifurcation finding.
//! - Internal: `.agents/receipts/target-class-dispatch-confusers/*` —
//!   Wave 5 Lane J implementation receipt.

// jankurai:allow HLT-027-HUMAN-REVIEW-EVIDENCE-GAP physics-path unification (Lane I/J) was verified by cargo test --workspace; rerun: just fast

use serde::{Deserialize, Serialize};

use crate::clutter::ClutterRegime;
use crate::sim::{NoiseProfile, RadarSimConfig, TargetState};
#[cfg(test)]
pub use crate::sim::TakeoffProfile;

#[path = "scene_impl.rs"]
mod scene_impl;

#[path = "scene_enums.rs"]
mod scene_enums;
pub use scene_enums::{TargetClass, TargetKinematics};

/// A scene to be synthesised by the unified
/// [`crate::sim::synthesize_scene`] entry point. Carries everything
/// that the single-target physics path consumed via `RadarSimConfig +
/// TakeoffProfile + NoiseProfile`, but factored so per-target
/// kinematics are decoupled from sensor/environment parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneDescriptor {
    /// Site/antenna geometry — currently just the antenna AGL height,
    /// which feeds the two-ray multipath term in
    /// [`crate::propagation::two_ray_propagation_factor_magnitude`].
    pub geometry: SiteGeometry,
    /// Environment terms (clutter regime, atmospherics, multipath).
    /// These were previously split between `RadarSimConfig` and
    /// `NoiseProfile`; in the unified path they live together so a
    /// single environment can be shared across many targets in the
    /// same scene.
    pub environment: EnvironmentDescriptor,
    /// Per-target entities. A scene MAY contain zero, one, or more
    /// entities. Lane I supports exactly one entity carrying a
    /// `FromTakeoffProfile` kinematics so the prior wrapper round-
    /// trips byte-stably; Lane J extends to N entities with native
    /// per-class kinematics.
    pub targets: Vec<TargetEntity>,
}

impl SceneDescriptor {
    /// Construct a scene from a [`RadarSimConfig`] and [`NoiseProfile`], mapping
    /// the config/noise fields to the unified geometry + environment representation.
    pub fn from_radar_config(
        config: &RadarSimConfig,
        noise: &NoiseProfile,
        targets: Vec<TargetEntity>,
    ) -> Self {
        Self {
            geometry: SiteGeometry {
                antenna_altitude_agl_m: config.radar_altitude_agl_m,
            },
            environment: EnvironmentDescriptor {
                clutter_regime: noise.clutter_regime,
                atmospheric_one_way_db_per_km: config.atmospheric_one_way_db_per_km,
                rain_rate_mm_per_h: config.rain_rate_mm_per_h,
                ground_reflection_coefficient_magnitude: config
                    .ground_reflection_coefficient_magnitude,
            },
            targets,
        }
    }
}

/// Antenna / site geometry shared by every entity in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SiteGeometry {
    /// Radar antenna height above ground level (m). Mirrors
    /// [`crate::sim::RadarSimConfig::radar_altitude_agl_m`].
    pub antenna_altitude_agl_m: f64,
}

/// Environmental terms (clutter, propagation, multipath) shared by
/// every entity in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentDescriptor {
    /// Optional cited K/Weibull/log-normal clutter regime. When
    /// `Some(_)`, the synthesis chain uses
    /// [`crate::clutter::generate_clutter_sequence`]; when `None`, it
    /// falls back to the prior Gaussian AR(1) for byte-stable
    /// bridged with pre-Lane-C fixtures.
    pub clutter_regime: Option<ClutterRegime>,
    /// One-way atmospheric specific attenuation (dB/km) at carrier.
    /// Mirrors [`crate::sim::RadarSimConfig::atmospheric_one_way_db_per_km`].
    pub atmospheric_one_way_db_per_km: f64,
    /// Rain rate along the path (mm/h). Zero disables the rain term.
    pub rain_rate_mm_per_h: f64,
    /// Magnitude of the ground reflection coefficient `|Γ|` for the
    /// two-ray multipath term. Zero disables two-ray multipath.
    pub ground_reflection_coefficient_magnitude: f64,
}

/// A single entity in the scene — class + kinematics + spawn time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetEntity {
    /// Truth class label. Drives downstream RCS-table selection
    /// (Lane J) and confuser-vs-positive labelling in the dataset
    /// generator. Lane I uses this field only for round-tripping; the
    /// physics path still consumes the prior `TakeoffProfile`.
    pub class: TargetClass,
    /// How this entity moves through the scene over the episode
    /// window. Today only `FromTakeoffProfile` is honoured by the
    /// physics path; Lane J adds native kinematics for non-positive
    /// classes.
    pub kinematics: TargetKinematics,
    /// Time at which this entity becomes visible in the scene
    /// (seconds, relative to episode start). Reserved for Lane J's
    /// multi-target dispatch; the single-entity wrapper used by Lane
    /// I always passes `0.0`.
    pub spawn_time_s: f64,
}

impl TargetKinematics {
    /// Resolve the kinematic [`TargetState`] for this entity at episode
    /// time `t_s`.
    ///
    /// `initial_range_m` and `antenna_alt_agl_m` are provided as
    /// fallbacks for variants that do not carry their own range /
    /// antenna height (e.g. the [`Bird`] and [`Helicopter`] entities
    /// inherit their initial range from the scene-level dispatch; the
    /// [`GroundVehicle`], [`WindTurbine`], [`Balloon`], and [`Kite`]
    /// variants carry their own initial range or anchor range and
    /// ignore the recovery).
    ///
    /// **Convention:** `heading_deg` is the bearing of the airframe
    /// velocity vector relative to the radar line-of-sight (0° → moving
    /// directly toward the radar, closing; 180° → moving directly away;
    /// 90° → broadside crossing with no radial component). Positive
    /// radial velocity = closing motion, matching the rest of
    /// `echoforge-radar` (see [`crate::sim::TakeoffProfile::state_at`]
    /// for the same sign convention).
    ///
    /// The `MultipathGhost` variant **panics** when called directly —
    /// callers must first resolve the parent entity's state and apply
    /// the multipath geometry at the [`crate::sim::synthesize_scene`]
    /// level. The variant exists so `state_at` covers all enum cases
    /// (no `_ => unreachable!()` fallthrough), and the panic guards
    /// against an accidental direct invocation that would silently
    /// return the wrong geometry.
    pub fn state_at(
        &self,
        t_s: f64,
        initial_range_m: f64,
        antenna_alt_agl_m: f64,
    ) -> TargetState {
        scene_impl::target_kinematics_state_at(self, t_s, initial_range_m, antenna_alt_agl_m)
    }
}

#[cfg(test)]
#[path = "scene_tests.rs"]
mod tests;
