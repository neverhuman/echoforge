//! Micro-Doppler rotating-rod assertions.
//!
//! 1. Sideband spread (10 dB occupied bandwidth) grows linearly with ω:
//!    a 10× rotation rate produces ≈10× spread. The bound is loose
//!    (factor of 2 either side) because STFT bandwidth is also a
//!    function of window length and discretisation; the qualitative
//!    monotonic scaling is the physical assertion.
//! 2. A single-blade Propeller spectrogram matches the standalone
//!    RotatingRod spectrogram bin-for-bin.
//!
//! Reference: Chen, "The Micro-Doppler Effect in Radar", §4.2.

use echoforge_validate::micro_doppler::{
    propeller::Propeller,
    rotating_rod::{rcs_time_series, sideband_spread_hz, spectrogram, RotatingRod},
};

const FREQ_HZ: f64 = 10e9;
const LENGTH_M: f64 = 0.3;
const PRF_HZ: f64 = 20_000.0;
const DURATION_S: f64 = 0.2;
const WIN_S: f64 = 0.02;
const HOP_S: f64 = 0.005;

fn spread_for(omega: f64) -> f64 {
    let rod = RotatingRod::new(LENGTH_M, omega, FREQ_HZ);
    let ts = rcs_time_series(&rod, DURATION_S, PRF_HZ);
    let spec = spectrogram(&ts, WIN_S, HOP_S);
    sideband_spread_hz(&spec)
}

#[test]
fn sideband_spread_scales_with_omega() {
    let slow = spread_for(10.0 * std::f64::consts::PI); // 5 Hz rev rate
    let fast = spread_for(100.0 * std::f64::consts::PI); // 50 Hz rev rate
    assert!(slow > 0.0, "slow spread degenerate: {slow}");
    assert!(fast > 0.0, "fast spread degenerate: {fast}");
    let ratio = fast / slow;
    // Physical expectation: tip velocity scales with ω → spread ratio ≈ 10.
    // Allow a loose band around 10 to account for STFT windowing.
    assert!(
        ratio > 4.0 && ratio < 25.0,
        "spread ratio {ratio:.2} outside [4, 25] — expected ~10"
    );
}

#[test]
fn single_blade_propeller_matches_rod() {
    let omega = 60.0 * std::f64::consts::PI;
    let rod = RotatingRod::new(LENGTH_M, omega, FREQ_HZ);
    let rod_ts = rcs_time_series(&rod, DURATION_S, PRF_HZ);

    // Build a 1-blade propeller with matching effective rod length.
    // Propeller's RotatingRod template uses length = 2·blade_length.
    let prop = Propeller::new(
        1,
        LENGTH_M / 2.0,
        omega * 60.0 / (2.0 * std::f64::consts::PI),
        FREQ_HZ,
    );
    let prop_ts = prop.rcs_time_series(DURATION_S, PRF_HZ);

    assert_eq!(rod_ts.len(), prop_ts.len());
    for ((tr, cr), (tp, cp)) in rod_ts.iter().zip(prop_ts.iter()) {
        assert!((tr - tp).abs() < 1e-12);
        assert!((cr - cp).norm() < 1e-9, "rod={cr:?} prop={cp:?} t={tr}");
    }
}
