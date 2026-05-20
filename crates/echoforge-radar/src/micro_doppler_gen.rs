//! Micro-Doppler kinematic generators.
//!
//! These generators produce time-series of radial-velocity contributions
//! (in metres per second) for typical micro-Doppler signal sources:
//! propellers, helicopter main+tail rotor pairs, bird wingbeats, and jet
//! compressor blades. They complement the micro-Doppler *detector* (see
//! [`crate::detectors::MicroDopplerDetector`]): the detector measures
//! periodic sideband structure on a range-Doppler grid; this module
//! supplies the *source* kinematics that the simulator can sum into the
//! IQ stream.
//!
//! The contract is intentionally simple: each generator exposes a
//! `radial_velocity_at(t)` function. The simulator (or any downstream
//! consumer) can convert that velocity into a phase increment via
//! `phi = 4π · v · f_c / c · dt` and inject it as a phasor multiply on
//! the target return.
//!
//! All math is pure-`f64` standard library; no external dependencies.

use std::f64::consts::TAU;

/// A source that produces a scalar radial velocity (m/s) at time `t_s`
/// seconds. Positive values indicate motion toward the radar (closing),
/// matching the convention used elsewhere in `echoforge-radar`.
pub trait MicroDopplerGenerator {
    /// Instantaneous radial velocity contribution at `t_s` seconds.
    fn radial_velocity_at(&self, t_s: f64) -> f64;
}

/// Propeller / rotor model: `blade_count` rigid blades of length
/// `blade_length_m` rotating at `rotation_hz`, viewed broadside.
///
/// Each blade tip traces a circle of radius `blade_length_m`. The radial
/// velocity contribution of blade `k` (0-indexed) is modeled as the
/// projection of its tip's tangential velocity onto the radar line-of-
/// sight, which for broadside-on viewing reduces to
///
/// ```text
///     v_k(t) = v_tip · sin(2π · f · t + 2π · k / N + φ0)
/// ```
///
/// where `v_tip = 2π · f · L` is the blade-tip speed. We return the
/// **dominant blade** at each instant — `max_k |v_k(t)|`, signed by
/// that blade — rather than the algebraic sum across blades. The
/// algebraic sum cancels identically to zero for any rotor with `N ≥ 2`
/// evenly-spaced blades (a well-known identity for sums of equally-
/// spaced unit phasors). The dominant-blade convention preserves the
/// physically observable "blade-flash" character at the blade-pass
/// frequency `N · f`, which is the line micro-Doppler detectors lock
/// onto. The peak amplitude reaches `v_tip` whenever any blade tip is
/// directly on the line-of-sight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropellerGenerator {
    pub blade_count: usize,
    pub rotation_hz: f64,
    pub blade_length_m: f64,
    pub phase_offset_rad: f64,
}

impl PropellerGenerator {
    pub fn new(
        blade_count: usize,
        rotation_hz: f64,
        blade_length_m: f64,
        phase_offset_rad: f64,
    ) -> Self {
        Self {
            blade_count,
            rotation_hz,
            blade_length_m,
            phase_offset_rad,
        }
    }

    /// Tip speed (m/s) of any blade: `2π · f · L`.
    pub fn tip_speed_mps(&self) -> f64 {
        TAU * self.rotation_hz * self.blade_length_m
    }
}

impl MicroDopplerGenerator for PropellerGenerator {
    fn radial_velocity_at(&self, t_s: f64) -> f64 {
        if self.blade_count == 0 {
            return 0.0;
        }
        let tip_speed = self.tip_speed_mps();
        let base_angle = TAU * self.rotation_hz * t_s + self.phase_offset_rad;
        // Dominant-blade selection: return the largest-magnitude
        // contribution at this instant, signed by its blade's
        // projection. See struct docs for the rationale.
        let mut best = 0.0_f64;
        for k in 0..self.blade_count {
            let blade_angle = base_angle + TAU * (k as f64) / (self.blade_count as f64);
            let v = tip_speed * blade_angle.sin();
            if v.abs() > best.abs() {
                best = v;
            }
        }
        best
    }
}

/// Combined helicopter main-rotor + tail-rotor generator. Both rotors
/// are modeled as [`PropellerGenerator`] instances summed together. The
/// tail rotor is offset from the main hub by `tail_offset_m`; that
/// offset enters as a small additional radial-velocity contribution
/// (the boom's motion is negligible at typical hover frequencies, so
/// we keep the offset purely geometric and let the rotor frequencies
/// dominate).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelicopterRotorGenerator {
    pub main_blade_count: usize,
    pub main_rotation_hz: f64,
    pub main_blade_length_m: f64,
    pub tail_blade_count: usize,
    pub tail_rotation_hz: f64,
    pub tail_blade_length_m: f64,
    pub tail_offset_m: f64,
}

impl HelicopterRotorGenerator {
    pub fn new(
        main_blade_count: usize,
        main_rotation_hz: f64,
        main_blade_length_m: f64,
        tail_blade_count: usize,
        tail_rotation_hz: f64,
        tail_blade_length_m: f64,
        tail_offset_m: f64,
    ) -> Self {
        Self {
            main_blade_count,
            main_rotation_hz,
            main_blade_length_m,
            tail_blade_count,
            tail_rotation_hz,
            tail_blade_length_m,
            tail_offset_m,
        }
    }

    fn main(&self) -> PropellerGenerator {
        PropellerGenerator::new(
            self.main_blade_count,
            self.main_rotation_hz,
            self.main_blade_length_m,
            0.0,
        )
    }

    fn tail(&self) -> PropellerGenerator {
        PropellerGenerator::new(
            self.tail_blade_count,
            self.tail_rotation_hz,
            self.tail_blade_length_m,
            0.0,
        )
    }
}

impl MicroDopplerGenerator for HelicopterRotorGenerator {
    fn radial_velocity_at(&self, t_s: f64) -> f64 {
        let main = self.main().radial_velocity_at(t_s);
        let tail = self.tail().radial_velocity_at(t_s);
        // tail_offset_m enters as a constant geometric bias on the boom
        // angular projection; for the purposes of micro-Doppler kinematics
        // we use a small `cos`-modulated correction so the offset has a
        // bounded effect on the radial-velocity envelope without changing
        // the dominant spectral lines from the two rotors.
        let boom_term = self.tail_offset_m
            * TAU
            * self.tail_rotation_hz
            * 0.005
            * (TAU * self.tail_rotation_hz * 0.5).cos();
        main + tail + boom_term
    }
}

/// Bird wingbeat generator. The wing-tip velocity along the radar line-
/// of-sight is well-approximated by a sinusoid at the wingbeat
/// frequency, with peak amplitude proportional to the wing length and
/// the wingbeat angular rate. We add a slow amplitude modulation of
/// depth `amplitude_modulation_depth` (0..1) to capture body-yaw and
/// glide-flap interplay observed at the species level.
///
/// ```text
///     v(t) = (v_peak · (1 + A · sin(2π · f_am · t))) · sin(2π · f_wb · t)
///     v_peak = 2π · f_wb · L
/// ```
///
/// Here `f_am = f_wb / 4` is the slow modulation frequency; this keeps
/// the wingbeat fundamental sharp and avoids polluting the principal
/// line spectrum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BirdWingbeatGenerator {
    pub wingbeat_hz: f64,
    pub wing_length_m: f64,
    pub amplitude_modulation_depth: f64,
}

impl BirdWingbeatGenerator {
    pub fn new(wingbeat_hz: f64, wing_length_m: f64, amplitude_modulation_depth: f64) -> Self {
        Self {
            wingbeat_hz,
            wing_length_m,
            amplitude_modulation_depth,
        }
    }

    /// Peak wing-tip radial velocity at full extension: `2π · f · L`.
    pub fn peak_velocity_mps(&self) -> f64 {
        TAU * self.wingbeat_hz * self.wing_length_m
    }
}

impl MicroDopplerGenerator for BirdWingbeatGenerator {
    fn radial_velocity_at(&self, t_s: f64) -> f64 {
        let v_peak = self.peak_velocity_mps();
        let am_depth = self.amplitude_modulation_depth.clamp(0.0, 1.0);
        let am_hz = self.wingbeat_hz * 0.25;
        let am = 1.0 + am_depth * (TAU * am_hz * t_s).sin();
        v_peak * am * (TAU * self.wingbeat_hz * t_s).sin()
    }
}

/// Jet engine compressor-blade generator. Treated as a high-rate
/// propeller with a configurable per-blade maximum velocity (the
/// physical blade length is implicit in `max_blade_velocity_mps` to
/// avoid having to specify the compressor radius). The radial velocity
/// contribution per blade is
///
/// ```text
///     v_k(t) = v_max · sin(2π · f · t + 2π · k / N)
/// ```
///
/// Like [`PropellerGenerator`], the returned value is the dominant
/// blade's signed velocity (`max_k |v_k(t)|`) rather than the algebraic
/// sum, so the spectral line lands at the blade-pass frequency
/// `N · f_compressor`. Compressor stages typically run in the tens of
/// kHz blade-pass regime; nothing in this implementation caps the
/// frequency, but downstream consumers should ensure the sample rate
/// is high enough to avoid aliasing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JetCompressorGenerator {
    pub compressor_hz: f64,
    pub blade_count: usize,
    pub max_blade_velocity_mps: f64,
}

impl JetCompressorGenerator {
    pub fn new(compressor_hz: f64, blade_count: usize, max_blade_velocity_mps: f64) -> Self {
        Self {
            compressor_hz,
            blade_count,
            max_blade_velocity_mps,
        }
    }
}

impl MicroDopplerGenerator for JetCompressorGenerator {
    fn radial_velocity_at(&self, t_s: f64) -> f64 {
        if self.blade_count == 0 {
            return 0.0;
        }
        let base_angle = TAU * self.compressor_hz * t_s;
        let mut best = 0.0_f64;
        for k in 0..self.blade_count {
            let blade_angle = base_angle + TAU * (k as f64) / (self.blade_count as f64);
            let v = self.max_blade_velocity_mps * blade_angle.sin();
            if v.abs() > best.abs() {
                best = v;
            }
        }
        best
    }
}

/// Sample a generator into a regularly-spaced time series of
/// radial-velocity values (m/s). The first sample lands at `t_start_s`;
/// successive samples are spaced `dt_s` apart. Returns exactly
/// `n_samples` values.
pub fn sample_velocity_series(
    gen: &dyn MicroDopplerGenerator,
    t_start_s: f64,
    dt_s: f64,
    n_samples: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = t_start_s + (i as f64) * dt_s;
        out.push(gen.radial_velocity_at(t));
    }
    out
}

#[cfg(test)]
#[path = "micro_doppler_gen_tests.rs"]
mod tests;
