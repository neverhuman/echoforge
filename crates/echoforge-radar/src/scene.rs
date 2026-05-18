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
//!      which fabricated bird / vehicle / turbine / multipath-ghost
//!      envelopes WITHOUT going through the chain in (1).
//!
//! That bifurcation meant the generator identity literally labelled
//! the class — a red flag to any radar engineer reviewing the dataset
//! card. Lane I introduces a single physics path:
//!
//!   `synthesize_scene(scene: SceneDescriptor, …) -> SyntheticEpisode`
//!
//! where ALL target classes (positives + confusers) traverse the same
//! chain. `synthesize_takeoff_episode` becomes a thin back-compat
//! wrapper that constructs a one-entity `SceneDescriptor` with
//! [`TargetClass::ShahedClassPiston`] and
//! [`TargetKinematics::FromTakeoffProfile`] and forwards through the
//! unified path. Existing reproduction fixtures stay byte-stable.
//!
//! ## Scope of this lane
//!
//! Lane I lands the **types and the single-entity wiring**. Multi-
//! entity scenes (multiple targets summed into shared IQ) and native
//! per-class kinematics for confusers (Bird, GroundVehicle, etc.) are
//! Lane J's responsibility. Wiring those today without the
//! corresponding RCS-aspect-lookup / propulsion-model dispatch would
//! constitute a measured-truth claim about confuser physics that we
//! cannot honestly defend at this point in the credibility sweep.
//!
//! ## References
//!
//! - Skolnik, *Introduction to Radar Systems*, 3rd ed.
//! - Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.
//! - Internal: `.agents/receipts/realism-v4-red-team-gap-report/*` —
//!   the original bifurcation finding.

use serde::{Deserialize, Serialize};

use crate::clutter::ClutterRegime;
use crate::sim::TakeoffProfile;

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
    /// `FromTakeoffProfile` kinematics so the legacy wrapper round-
    /// trips byte-stably; Lane J extends to N entities with native
    /// per-class kinematics.
    pub targets: Vec<TargetEntity>,
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
    /// falls back to the legacy Gaussian AR(1) for byte-stable
    /// back-compat with pre-Lane-C fixtures.
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
    /// physics path still consumes the legacy `TakeoffProfile`.
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

/// Truth class label for a scene entity. The enum spans both positive
/// (Shahed-class) and confuser classes so the unified synthesis path
/// can label EVERY emitted detection by its source class rather than
/// inferring it from which generator produced the envelope.
///
/// The list deliberately enumerates the confuser families that
/// downstream hard-negative-mining cares about (birds, vehicles,
/// turbines, multipath ghosts, terrain glint, helicopters, balloons,
/// kites, MANPADS ground returns). Lane J fills in native per-class
/// kinematics + RCS dispatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetClass {
    /// Piston-engined Shahed-class public-proxy archetype. Today this
    /// is the only class that drives full physics through the
    /// wrapper — see `synthesize_takeoff_episode`.
    ShahedClassPiston,
    /// Jet-engined Shahed-class proxy (e.g. -136 with jet conversion).
    ShahedClassJet,
    /// Bird (small or large). Confuser class; envelope statistics
    /// drive most current dataset entries.
    Bird,
    /// Ground vehicle (truck, SUV, etc.). Confuser class.
    GroundVehicle,
    /// Wind turbine. Confuser with strong micro-Doppler signature.
    WindTurbine,
    /// Tethered or free balloon. Confuser.
    Balloon,
    /// Kite. Confuser.
    Kite,
    /// Specular multipath ghost of another entity. `parent_idx`
    /// indexes back into `SceneDescriptor::targets`; the ghost
    /// inherits its parent's kinematics (offset by the multipath
    /// geometry) per Skolnik §1.6 and Lane J's dispatch.
    MultipathGhost { parent_idx: usize },
    /// Sparkle off terrain feature (rock face, building corner).
    /// Static confuser.
    TerrainGlint,
    /// Rotary-wing aircraft. Confuser with very different micro-
    /// Doppler signature from the Shahed prop.
    Helicopter,
    /// Stationary ground return resembling a MANPADS / man with
    /// radar-visible kit. Confuser used in hard-negative mining for
    /// the ground-launch decoy lane.
    ManRadarReturn,
}

/// How a [`TargetEntity`] moves through the scene.
///
/// Lane I exposes one populated variant, [`TargetKinematics::FromTakeoffProfile`],
/// which carries the legacy [`TakeoffProfile`] so the wrapper round-
/// trips byte-stably through the unified path. The `_UnusedFuture`
/// variant reserves the discriminant space for Lane J's native
/// kinematics without breaking serialized fixtures on the way.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TargetKinematics {
    /// Wraps a legacy [`TakeoffProfile`] so existing callers stay
    /// byte-stable through the unified path. This is the migration
    /// bridge between the bifurcated old physics and the single
    /// unified path Lane J will exercise.
    FromTakeoffProfile(TakeoffProfile),
    /// Reserved discriminant for Lane J — native scene kinematics
    /// (constant-velocity birds, parked vehicles, stationary
    /// turbines, multipath ghosts derived from a parent track,
    /// etc.). Constructing this variant today is a no-op for the
    /// synthesis path; it exists so on-disk serialised fixtures from
    /// Lane J's dev work can be parsed by Lane I binaries.
    #[doc(hidden)]
    #[serde(rename = "_unused_future")]
    _UnusedFuture,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip serde: a hand-built `SceneDescriptor` must survive
    /// JSON serialise + deserialise without losing fields. This is the
    /// minimal contract test that on-disk fixtures (Lane J campaign
    /// JSONs, scenario manifests) can rely on.
    #[test]
    fn scene_descriptor_roundtrip_serde() {
        let scene = SceneDescriptor {
            geometry: SiteGeometry {
                antenna_altitude_agl_m: 20.0,
            },
            environment: EnvironmentDescriptor {
                clutter_regime: None,
                atmospheric_one_way_db_per_km: 0.012,
                rain_rate_mm_per_h: 0.0,
                ground_reflection_coefficient_magnitude: 0.0,
            },
            targets: vec![TargetEntity {
                class: TargetClass::ShahedClassPiston,
                kinematics: TargetKinematics::FromTakeoffProfile(TakeoffProfile::default()),
                spawn_time_s: 0.0,
            }],
        };

        let json = serde_json::to_string(&scene).expect("serialise scene");
        let parsed: SceneDescriptor =
            serde_json::from_str(&json).expect("deserialise scene");
        assert_eq!(scene, parsed);
    }

    /// Minimal one-entity scene — assert the shape so consumers (the
    /// `synthesize_takeoff_episode` wrapper, Lane J's campaign runner)
    /// have a checked structural contract rather than relying on
    /// in-code inspection of which fields exist.
    #[test]
    fn scene_descriptor_minimal() {
        let scene = SceneDescriptor {
            geometry: SiteGeometry {
                antenna_altitude_agl_m: 12.5,
            },
            environment: EnvironmentDescriptor {
                clutter_regime: None,
                atmospheric_one_way_db_per_km: 0.0,
                rain_rate_mm_per_h: 0.0,
                ground_reflection_coefficient_magnitude: 0.0,
            },
            targets: vec![TargetEntity {
                class: TargetClass::ShahedClassPiston,
                kinematics: TargetKinematics::FromTakeoffProfile(TakeoffProfile::default()),
                spawn_time_s: 0.0,
            }],
        };

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
}
