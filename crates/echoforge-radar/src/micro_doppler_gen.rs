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
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn propeller_peak_velocity_matches_tip_speed() {
        // rotation_hz = 100, blade_length = 0.5 m
        // expected tip speed = 2π · 100 · 0.5 ≈ 314.159 m/s.
        let prop = PropellerGenerator::new(1, 100.0, 0.5, 0.0);
        let expected_tip = TAU * 100.0 * 0.5;
        assert!(
            approx(prop.tip_speed_mps(), expected_tip, 1e-9),
            "tip speed {} does not match expected {}",
            prop.tip_speed_mps(),
            expected_tip
        );
        // Sweep the period, find max |v|.
        let n = 2048usize;
        let dt = (1.0 / 100.0) / n as f64;
        let series = sample_velocity_series(&prop, 0.0, dt, n);
        let peak = series.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        // Within 5% per the packet contract.
        let tol = 0.05 * expected_tip;
        assert!(
            (peak - expected_tip).abs() <= tol,
            "peak {} not within 5% of expected {} (tol {})",
            peak,
            expected_tip,
            tol
        );
    }

    #[test]
    fn propeller_pattern_is_periodic_in_rotation_period() {
        let prop = PropellerGenerator::new(3, 75.0, 0.4, 0.21);
        let period = 1.0 / 75.0;
        // Sample at a handful of fractional offsets, compare to
        // `t + period`. Use a generous epsilon because we are summing
        // multiple sinusoids, but the equality is exact in theory.
        for k in 1..16 {
            let t = period * (k as f64) / 19.0;
            let a = prop.radial_velocity_at(t);
            let b = prop.radial_velocity_at(t + period);
            assert!(
                approx(a, b, 1e-9),
                "propeller not periodic at t={}: {} vs {}",
                t,
                a,
                b
            );
        }
    }

    #[test]
    fn helicopter_produces_two_frequency_components() {
        // The helicopter generator sums two rotor signals; each rotor
        // uses dominant-blade selection so its time-domain pattern
        // repeats at the blade-pass rate `f_bp = N · f_rot`. We isolate
        // each component by additive decomposition: compute the
        // combined time series, then subtract each rotor's standalone
        // dominant-blade signal. If both subtractions leave only a
        // small residual, the combined signal contains both components
        // as independent additive sources at distinct frequencies.
        let heli = HelicopterRotorGenerator::new(3, 6.0, 5.5, 4, 24.0, 0.9, 6.0);
        let main = PropellerGenerator::new(3, 6.0, 5.5, 0.0);
        let tail = PropellerGenerator::new(4, 24.0, 0.9, 0.0);
        let fs = 4_000.0;
        let n = 4_096usize;
        let dt = 1.0 / fs;
        let combined = sample_velocity_series(&heli, 0.0, dt, n);
        let main_only = sample_velocity_series(&main, 0.0, dt, n);
        let tail_only = sample_velocity_series(&tail, 0.0, dt, n);

        let energy = |xs: &[f64]| xs.iter().map(|v| v * v).sum::<f64>().sqrt();
        let total_e = energy(&combined);
        // Subtracting main + tail must leave only the small boom_term
        // residual. Assert the residual is at most 5 % of total
        // energy — strong evidence both components are present.
        let residual: Vec<f64> = combined
            .iter()
            .zip(main_only.iter())
            .zip(tail_only.iter())
            .map(|((c, m), t)| c - m - t)
            .collect();
        let residual_e = energy(&residual);
        assert!(
            residual_e <= 0.05 * total_e,
            "main + tail should reconstruct combined helicopter signal; residual {} vs total {}",
            residual_e,
            total_e
        );

        // Each rotor individually must contribute non-trivial energy.
        let main_e = energy(&main_only);
        let tail_e = energy(&tail_only);
        assert!(
            main_e > 0.05 * total_e,
            "main rotor energy too low: {}",
            main_e
        );
        assert!(
            tail_e > 0.05 * total_e,
            "tail rotor energy too low: {}",
            tail_e
        );

        // And they must be at *distinct* frequencies: project each
        // rotor's standalone series onto both blade-pass rates and
        // confirm each one peaks at its own.
        let project = |xs: &[f64], f: f64| -> f64 {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (i, v) in xs.iter().enumerate() {
                let t = (i as f64) * dt;
                re += v * (TAU * f * t).cos();
                im += v * (TAU * f * t).sin();
            }
            (re * re + im * im).sqrt() / n as f64
        };
        let main_bp = 3.0 * 6.0; // 18 Hz
        let tail_bp = 4.0 * 24.0; // 96 Hz
        assert!(
            project(&main_only, main_bp) > project(&main_only, tail_bp),
            "main rotor must peak at its own blade-pass, not tail's"
        );
        assert!(
            project(&tail_only, tail_bp) > project(&tail_only, main_bp),
            "tail rotor must peak at its own blade-pass, not main's"
        );
    }

    #[test]
    fn bird_wingbeat_is_periodic_in_wingbeat_period() {
        // 10 Hz wingbeat -> period 0.1 s. Note the slow amplitude
        // envelope at f_wb/4 = 2.5 Hz means the *full* periodicity is
        // 0.4 s (LCM with the envelope), so we check at 0.4 s.
        let bird = BirdWingbeatGenerator::new(10.0, 0.18, 0.3);
        let full_period = 1.0 / 2.5; // 0.4 s
        for k in 1..10 {
            let t = full_period * (k as f64) / 11.0;
            let a = bird.radial_velocity_at(t);
            let b = bird.radial_velocity_at(t + full_period);
            assert!(
                approx(a, b, 1e-9),
                "bird wingbeat not periodic at t={}: {} vs {}",
                t,
                a,
                b
            );
        }
    }

    #[test]
    fn sample_velocity_series_honours_start_and_length() {
        let prop = PropellerGenerator::new(2, 50.0, 0.3, 0.0);
        let t0 = 0.123;
        let dt = 1e-4;
        let n = 257usize;
        let series = sample_velocity_series(&prop, t0, dt, n);
        assert_eq!(series.len(), n);
        assert!(
            approx(series[0], prop.radial_velocity_at(t0), 1e-12),
            "first sample should equal radial_velocity_at(t_start_s)"
        );
        let last_t = t0 + (n as f64 - 1.0) * dt;
        assert!(
            approx(
                *series.last().unwrap(),
                prop.radial_velocity_at(last_t),
                1e-12
            ),
            "last sample should equal radial_velocity_at(t_start + (n-1)*dt)"
        );
    }

    #[test]
    fn identical_configs_produce_identical_series() {
        let a = PropellerGenerator::new(3, 120.0, 0.45, 0.7);
        let b = PropellerGenerator::new(3, 120.0, 0.45, 0.7);
        let sa = sample_velocity_series(&a, 0.0, 1e-5, 1024);
        let sb = sample_velocity_series(&b, 0.0, 1e-5, 1024);
        assert_eq!(
            sa, sb,
            "identical configs must produce bit-identical output"
        );

        let heli_a = HelicopterRotorGenerator::new(4, 5.5, 5.0, 4, 22.0, 0.85, 6.0);
        let heli_b = HelicopterRotorGenerator::new(4, 5.5, 5.0, 4, 22.0, 0.85, 6.0);
        let ha = sample_velocity_series(&heli_a, 0.1, 5e-5, 512);
        let hb = sample_velocity_series(&heli_b, 0.1, 5e-5, 512);
        assert_eq!(ha, hb, "helicopter generator must be deterministic");

        let bird_a = BirdWingbeatGenerator::new(12.5, 0.22, 0.4);
        let bird_b = BirdWingbeatGenerator::new(12.5, 0.22, 0.4);
        let ba = sample_velocity_series(&bird_a, 0.0, 1e-4, 600);
        let bb = sample_velocity_series(&bird_b, 0.0, 1e-4, 600);
        assert_eq!(ba, bb, "bird generator must be deterministic");
    }

    #[test]
    fn jet_compressor_produces_finite_bounded_output() {
        let jet = JetCompressorGenerator::new(2_000.0, 32, 250.0);
        let series = sample_velocity_series(&jet, 0.0, 1e-6, 4096);
        assert_eq!(series.len(), 4096);
        assert!(
            series.iter().all(|v| v.is_finite()),
            "jet output must be finite"
        );
        // Dominant-blade selection bounds the magnitude by the per-blade
        // amplitude (250 m/s here).
        let peak = series.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(
            peak <= 250.0 + 1e-6,
            "jet compressor peak {} exceeds per-blade maximum",
            peak
        );
        // The series cannot be uniformly zero for a non-degenerate jet.
        assert!(peak > 0.0, "jet compressor series collapsed to zero");
    }

    #[test]
    fn zero_blade_propeller_is_silent() {
        let prop = PropellerGenerator::new(0, 100.0, 0.5, 0.0);
        for k in 0..32 {
            let t = (k as f64) * 1e-4;
            assert_eq!(prop.radial_velocity_at(t), 0.0);
        }
        let jet = JetCompressorGenerator::new(2_000.0, 0, 100.0);
        assert_eq!(jet.radial_velocity_at(0.5), 0.0);
    }

    #[test]
    fn trait_object_dispatch_works_uniformly() {
        // Pin all four generators through the trait-object helper to
        // confirm they share a single sampling code path.
        let prop = PropellerGenerator::new(2, 80.0, 0.5, 0.0);
        let heli = HelicopterRotorGenerator::new(4, 6.0, 5.0, 4, 24.0, 0.8, 6.0);
        let bird = BirdWingbeatGenerator::new(10.0, 0.2, 0.25);
        let jet = JetCompressorGenerator::new(1_500.0, 16, 180.0);
        let gens: Vec<&dyn MicroDopplerGenerator> = vec![&prop, &heli, &bird, &jet];
        for g in gens {
            let series = sample_velocity_series(g, 0.0, 1e-5, 64);
            assert_eq!(series.len(), 64);
            assert!(series.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn bird_amplitude_envelope_increases_peak_with_depth() {
        // With am_depth=0, peak should equal v_peak. With depth=1, the
        // envelope (1 + sin(.)) reaches 2, so the peak should roughly
        // double. Use a long enough series to catch both extrema.
        let bird0 = BirdWingbeatGenerator::new(10.0, 0.2, 0.0);
        let bird1 = BirdWingbeatGenerator::new(10.0, 0.2, 1.0);
        let s0 = sample_velocity_series(&bird0, 0.0, 1e-4, 4_000);
        let s1 = sample_velocity_series(&bird1, 0.0, 1e-4, 4_000);
        let peak0 = s0.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let peak1 = s1.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            peak1 > 1.5 * peak0,
            "amplitude modulation should grow peak: depth=0 peak={}, depth=1 peak={}",
            peak0,
            peak1
        );
    }
}
