//! Enum definitions for scene target classes and kinematics.
//! Extracted from `scene.rs` for LOC compliance.
//!
//! See `scene.rs` for the full module documentation.

use serde::{Deserialize, Serialize};

use crate::sim::TakeoffProfile;

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

/// How a [`super::TargetEntity`] moves through the scene.
///
/// Lane I exposed one populated variant, [`TargetKinematics::FromTakeoffProfile`],
/// which carries the prior [`TakeoffProfile`] so the wrapper round-
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
    /// Wraps a prior [`TakeoffProfile`] so existing callers stay
    /// byte-stable through the unified path. This is the migration
    /// bridge between the bifurcated prior physics and the single
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
