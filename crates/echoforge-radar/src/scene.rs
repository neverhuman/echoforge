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

use serde::{Deserialize, Serialize};

use crate::clutter::ClutterRegime;
use crate::sim::{TakeoffProfile, TargetState};

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
/// Lane I exposed one populated variant, [`TargetKinematics::FromTakeoffProfile`],
/// which carries the legacy [`TakeoffProfile`] so the wrapper round-
/// trips byte-stably through the unified path. **Lane J (this lane)
/// promotes confuser kinematics from envelope statistics
/// (`echoforge-dataset/src/ml_training.rs::build_frame_products`) to
/// first-class scene entities that traverse the same physics chain as
/// positive targets.**
///
/// The per-variant kinematic envelopes are cited to Lane J prep
/// research (Wave-A `physics_dossier.md`, Rahman & Robertson, *Nature*
/// 2018 for bird wingbeat statistics; Skolnik *Introduction to Radar
/// Systems* 3rd ed. for confuser RCS aspect dependence; Chen,
/// *Micro-Doppler Effect in Radar*, 2011 for helicopter rotor
/// signatures). The numeric ranges in the per-variant docstrings are
/// the **inhabited region of physical parameter space** the simulator
/// can honestly defend at the synthesis surface; per-instance values
/// outside these bands are not blocked but are not characterised by
/// the supporting open-source measurements either.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TargetKinematics {
    /// Wraps a legacy [`TakeoffProfile`] so existing callers stay
    /// byte-stable through the unified path. This is the migration
    /// bridge between the bifurcated old physics and the single
    /// unified path Lane J exercises for confusers.
    FromTakeoffProfile(TakeoffProfile),

    /// Avian flapping flight. Speed 5–40 m/s, wingbeat 2–8 Hz typical
    /// for large birds (Rahman & Robertson, *Nature* 2018; Pennycuick,
    /// *Modelling the Flying Bird*, 2008 §6.2). `cruise_speed_mps` is
    /// the airframe ground speed, `heading_deg` is the bearing to the
    /// radar (0° → moving directly toward the radar, 90° → broadside
    /// crossing), and `wingbeat_hz` + `wing_length_m` feed the
    /// downstream [`crate::micro_doppler_gen::BirdWingbeatGenerator`].
    /// Micro-Doppler is captured **separately** by that generator;
    /// `state_at` returns range / altitude / radial-velocity geometry
    /// only.
    Bird {
        cruise_speed_mps: f64,
        altitude_agl_m: f64,
        heading_deg: f64,
        wingbeat_hz: f64,
        wing_length_m: f64,
    },

    /// Ground vehicle (car/truck/tracked). Speed 0–35 m/s, altitude
    /// ~0 m AGL, no rotor micro-Doppler (drivetrain harmonics are
    /// at frequencies the propeller / rotor windows reject; see
    /// Wave-A `physics_dossier.md` §confuser-list-vehicle). Broadside
    /// RCS is the dominant aspect when modelled at typical ground-
    /// radar grazing angles.
    GroundVehicle {
        speed_mps: f64,
        heading_deg: f64,
        initial_range_m: f64,
    },

    /// Wind turbine. Static hub; 3-blade rotor at 0.17–0.42 Hz
    /// rotation (10–25 RPM); blade length 20–60 m (modern utility-
    /// scale machines). Blade-pass frequency = `blade_count *
    /// rotation_hz` = 0.5–1.25 Hz, which aliases into the radar
    /// Doppler band as a strong jet-engine-modulation-like spectral
    /// signature (Skolnik 3rd ed. §11.5; Naqvi et al., *IEE Proc.
    /// Radar Sonar Navig.* 2015). `state_at` returns geometry only —
    /// hub static, zero radial velocity — and the blade micro-Doppler
    /// is downstream of the [`crate::micro_doppler_gen::PropellerGenerator`]
    /// dispatch.
    WindTurbine {
        hub_range_m: f64,
        hub_altitude_agl_m: f64,
        blade_count: usize,
        rotation_hz: f64,
        blade_length_m: f64,
    },

    /// Multipath ghost — a phantom return paired with a real target
    /// at offset range `R + 2·h_r·h_t/R` with amplitude scaled by
    /// `|Γ|·|F|` per the two-ray multipath geometry (Skolnik 3rd ed.
    /// §1.6). The ghost is synthesised as a **separate entity**
    /// referencing the parent by index; `state_at` returns the
    /// parent's geometry unchanged, and the synthesis loop in
    /// [`crate::sim::synthesize_scene`] applies the offset + amplitude
    /// scaling when accumulating the ghost contribution.
    MultipathGhost {
        parent_idx: usize,
        reflection_coefficient_magnitude: f64,
    },

    /// Drifting balloon — slow wind drift ~1–10 m/s, no
    /// micro-Doppler (a Mylar reflector is a flat scatterer with no
    /// rotating parts). When `tethered = true`, the kinematic state
    /// is collapsed to static (zero motion) so a tethered weather
    /// balloon looks like a static infrastructure return modulated
    /// by Doppler-zero clutter; when `tethered = false`, the balloon
    /// drifts at `drift_speed_mps` along `drift_heading_deg`.
    Balloon {
        drift_speed_mps: f64,
        drift_heading_deg: f64,
        altitude_agl_m: f64,
        tethered: bool,
    },

    /// Tethered kite — near-stationary at the anchor with intermittent
    /// wind-driven flutter. Modelled as a static entity with
    /// `wind_gust_amplitude_mps` of radial-velocity jitter around the
    /// anchor; the gust term is deterministic at synthesis time so
    /// fixture seeds remain reproducible. The gust amplitude maps to a
    /// per-pulse radial-velocity offset derived from the kinematic
    /// time `t_s`, not from a fresh RNG draw (that responsibility
    /// stays with `crate::sim::NoiseProfile`).
    Kite {
        anchor_range_m: f64,
        altitude_agl_m: f64,
        wind_gust_amplitude_mps: f64,
    },

    /// Helicopter — main rotor 4–6 blades at 200–400 RPM (3.3–6.7 Hz
    /// rotation), tail rotor 2–4 blades. Dual-rotor micro-Doppler
    /// signature is distinct from a fixed-wing UAS propeller line
    /// (Chen, *Micro-Doppler Effect in Radar*, 2011, §6.2): the tail
    /// rotor produces a high-frequency sideband that the
    /// [`crate::micro_doppler_gen::HelicopterRotorGenerator`]
    /// reproduces. `state_at` returns the airframe geometry (straight-
    /// line cruise like a [`Bird`] entity); the rotor lines are added
    /// by the generator dispatch.
    Helicopter {
        cruise_speed_mps: f64,
        altitude_agl_m: f64,
        heading_deg: f64,
        main_blade_count: usize,
        main_rotation_hz: f64,
        main_blade_length_m: f64,
        tail_blade_count: usize,
        tail_rotation_hz: f64,
        tail_blade_length_m: f64,
    },
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
    /// ignore the fallback).
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
        let _ = antenna_alt_agl_m; // unused in current variants; kept in API for Lane K parity work
        match self {
            TargetKinematics::FromTakeoffProfile(profile) => profile.state_at(t_s),

            TargetKinematics::Bird {
                cruise_speed_mps,
                altitude_agl_m,
                heading_deg,
                ..
            } => {
                let heading_rad = heading_deg.to_radians();
                let along_track_m = cruise_speed_mps * t_s;
                // Closing motion = positive radial velocity. Heading 0°
                // along the LOS reduces along-track displacement to
                // purely radial: range decreases by `speed * t`.
                let range_m =
                    (initial_range_m - along_track_m * heading_rad.cos()).max(0.0);
                let radial_velocity_mps = cruise_speed_mps * heading_rad.cos();
                TargetState {
                    time_s: t_s,
                    range_m,
                    altitude_m: *altitude_agl_m,
                    radial_velocity_mps,
                    pitch_deg: 0.0,
                    yaw_deg: 0.0,
                    propulsor_phase_rad: 0.0,
                }
            }

            TargetKinematics::GroundVehicle {
                speed_mps,
                heading_deg,
                initial_range_m: vehicle_initial_range_m,
            } => {
                let heading_rad = heading_deg.to_radians();
                let along_track_m = speed_mps * t_s;
                let range_m =
                    (vehicle_initial_range_m - along_track_m * heading_rad.cos()).max(0.0);
                let radial_velocity_mps = speed_mps * heading_rad.cos();
                TargetState {
                    time_s: t_s,
                    range_m,
                    altitude_m: 0.0, // ~0 m AGL per task brief
                    radial_velocity_mps,
                    pitch_deg: 0.0,
                    yaw_deg: 0.0,
                    propulsor_phase_rad: 0.0,
                }
            }

            TargetKinematics::WindTurbine {
                hub_range_m,
                hub_altitude_agl_m,
                ..
            } => TargetState {
                time_s: t_s,
                range_m: *hub_range_m,
                altitude_m: *hub_altitude_agl_m,
                radial_velocity_mps: 0.0, // hub is static
                pitch_deg: 0.0,
                yaw_deg: 0.0,
                propulsor_phase_rad: 0.0,
            },

            TargetKinematics::MultipathGhost { .. } => panic!(
                "TargetKinematics::MultipathGhost::state_at called directly; \
                 callers must first resolve the parent entity's state and \
                 apply the multipath geometry at synthesize_scene level. \
                 See crate::sim::synthesize_scene for the dispatch."
            ),

            TargetKinematics::Balloon {
                drift_speed_mps,
                drift_heading_deg,
                altitude_agl_m,
                tethered,
            } => {
                if *tethered {
                    // Tethered balloon: collapse to static geometry.
                    TargetState {
                        time_s: t_s,
                        range_m: initial_range_m,
                        altitude_m: *altitude_agl_m,
                        radial_velocity_mps: 0.0,
                        pitch_deg: 0.0,
                        yaw_deg: 0.0,
                        propulsor_phase_rad: 0.0,
                    }
                } else {
                    let heading_rad = drift_heading_deg.to_radians();
                    let along_track_m = drift_speed_mps * t_s;
                    let range_m =
                        (initial_range_m - along_track_m * heading_rad.cos()).max(0.0);
                    let radial_velocity_mps = drift_speed_mps * heading_rad.cos();
                    TargetState {
                        time_s: t_s,
                        range_m,
                        altitude_m: *altitude_agl_m,
                        radial_velocity_mps,
                        pitch_deg: 0.0,
                        yaw_deg: 0.0,
                        propulsor_phase_rad: 0.0,
                    }
                }
            }

            TargetKinematics::Kite {
                anchor_range_m,
                altitude_agl_m,
                wind_gust_amplitude_mps,
            } => {
                // Deterministic flutter: project a slow sinusoid at
                // 0.5 Hz so the gust amplitude shows up as a bounded
                // radial-velocity ripple around the anchor. Choosing
                // a deterministic carrier (not RNG) keeps the kite
                // entity reproducible for fixture replay.
                let gust_carrier_hz = 0.5;
                let radial_velocity_mps = wind_gust_amplitude_mps
                    * (2.0 * std::f64::consts::PI * gust_carrier_hz * t_s).sin();
                TargetState {
                    time_s: t_s,
                    range_m: *anchor_range_m,
                    altitude_m: *altitude_agl_m,
                    radial_velocity_mps,
                    pitch_deg: 0.0,
                    yaw_deg: 0.0,
                    propulsor_phase_rad: 0.0,
                }
            }

            TargetKinematics::Helicopter {
                cruise_speed_mps,
                altitude_agl_m,
                heading_deg,
                ..
            } => {
                // Straight-line cruise geometry — identical to Bird;
                // the dual-rotor micro-Doppler line spectrum is a
                // downstream concern handled by
                // crate::micro_doppler_gen::HelicopterRotorGenerator.
                let heading_rad = heading_deg.to_radians();
                let along_track_m = cruise_speed_mps * t_s;
                let range_m =
                    (initial_range_m - along_track_m * heading_rad.cos()).max(0.0);
                let radial_velocity_mps = cruise_speed_mps * heading_rad.cos();
                TargetState {
                    time_s: t_s,
                    range_m,
                    altitude_m: *altitude_agl_m,
                    radial_velocity_mps,
                    pitch_deg: 0.0,
                    yaw_deg: 0.0,
                    propulsor_phase_rad: 0.0,
                }
            }
        }
    }
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
    /// of the call-site `initial_range_m` fallback or the vehicle's
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
            rotation_hz: 0.3,    // ~18 RPM
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
}
