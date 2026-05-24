use super::*;
use crate::rcs::Polarization;

// ---------------------------------------------------------------
// Wave 4.5 Lane H2 — polarization-agility primitive tests.
//
// These verify that:
//   (1) `RadarSimConfig::default()` falls back to `(Vv, Vv)` so
//       pre-Lane-H2 fixtures are byte-stable,
//   (2) `polarization_for_pulse` cycles the sequence modulo
//       `pulse_count` (the classic VV/HH alternation per
//       Skolnik §7.5.3), and
//   (3) VV-only vs HH-only synthesis episodes differ in
//       integrated-profile peak by the expected ~1 dB amplitude
//       scaling, proving the polarization channel is actually
//       wired into the per-pulse target return.
// ---------------------------------------------------------------

/// (1) bridged — `RadarSimConfig::default()` has no
/// polarization sequences configured, so every pulse must resolve
/// to `(Vv, Vv)`. Existing reproduction fixtures predate the
/// polarization sequence and depend on this recovery.
#[test]
fn polarization_default_is_vv_backward_compatible() {
    let config = RadarSimConfig::default();
    assert!(
        config.pol_tx_sequence.is_none(),
        "default pol_tx_sequence must be None for bridged"
    );
    assert!(
        config.pol_rx_sequence.is_none(),
        "default pol_rx_sequence must be None for bridged"
    );
    // Pulse-zero must resolve to the canonical (Vv, Vv) pair.
    assert_eq!(
        config.polarization_for_pulse(0),
        (Polarization::Vv, Polarization::Vv),
        "polarization_for_pulse(0) must default to (Vv, Vv)"
    );
    // Higher pulse indices must also resolve to (Vv, Vv) when no
    // sequence is set — the modulo-cycling is a no-op when the
    // sequence is `None`.
    for pulse_idx in [1, 7, 32, 1024, usize::MAX / 2] {
        assert_eq!(
            config.polarization_for_pulse(pulse_idx),
            (Polarization::Vv, Polarization::Vv),
            "polarization_for_pulse({pulse_idx}) must default to (Vv, Vv)"
        );
    }
}

/// (2) Sequence wiring — with `pol_tx_sequence = Some(vec![Vv, Hh])`,
/// pulses 0, 2, 4, … must resolve to `Vv` and pulses 1, 3, 5, …
/// must resolve to `Hh`. This is the canonical pulse-to-pulse
/// polarization agility used by modern AESA radars for clutter
/// diversity (Skolnik §7.5.3).
///
/// When `pol_rx_sequence` is `None`, rx must mirror tx (the
/// matched/co-polar receive convention).
#[test]
fn polarization_sequence_alternates_vv_hh() {
    let config = RadarSimConfig {
        pol_tx_sequence: Some(vec![Polarization::Vv, Polarization::Hh]),
        ..RadarSimConfig::default()
    };

    // First period.
    assert_eq!(
        config.polarization_for_pulse(0),
        (Polarization::Vv, Polarization::Vv),
        "pulse 0 must be VV (rx mirrors tx when pol_rx_sequence is None)"
    );
    assert_eq!(
        config.polarization_for_pulse(1),
        (Polarization::Hh, Polarization::Hh),
        "pulse 1 must be HH (rx mirrors tx when pol_rx_sequence is None)"
    );

    // Second period — modulo cycling must wrap cleanly.
    assert_eq!(
        config.polarization_for_pulse(2),
        (Polarization::Vv, Polarization::Vv),
        "pulse 2 must wrap back to VV"
    );
    assert_eq!(
        config.polarization_for_pulse(3),
        (Polarization::Hh, Polarization::Hh),
        "pulse 3 must wrap to HH"
    );

    // Larger indices — full modulo cycle.
    for pulse_idx in 0..16 {
        let expected = if pulse_idx % 2 == 0 {
            Polarization::Vv
        } else {
            Polarization::Hh
        };
        let (tx, rx) = config.polarization_for_pulse(pulse_idx);
        assert_eq!(tx, expected, "pulse {pulse_idx} tx mismatch");
        assert_eq!(rx, expected, "pulse {pulse_idx} rx mirrors tx");
    }

    // Cross-pol case — separate tx/rx sequences enable HV / VH
    // depolarization measurements (Ulaby & Long §10.2).
    let cross_pol_config = RadarSimConfig {
        pol_tx_sequence: Some(vec![Polarization::Hh]),
        pol_rx_sequence: Some(vec![Polarization::Vv]),
        ..RadarSimConfig::default()
    };
    assert_eq!(
        cross_pol_config.polarization_for_pulse(0),
        (Polarization::Hh, Polarization::Vv),
        "cross-pol: tx=HH, rx=VV (i.e. HV — the cross-polar channel)"
    );

    // Empty sequence must fall back to defaults (defensive — avoids
    // a division by zero in the modulo cycling).
    let empty_config = RadarSimConfig {
        pol_tx_sequence: Some(vec![]),
        ..RadarSimConfig::default()
    };
    assert_eq!(
        empty_config.polarization_for_pulse(0),
        (Polarization::Vv, Polarization::Vv),
        "empty sequence must fall back to default (Vv, Vv)"
    );
}

/// (3) End-to-end synthesis — same target, same noise, same seed
/// run twice with VV-only vs HH-only polarization sequences. The
/// integrated-profile peak must differ by ~1 dB (the HH amplitude
/// scaling: linear 10^(+1/20) ≈ 1.122). If the two peaks were
/// identical, the polarization channel would NOT be wired into the
/// per-pulse target amplitude.
///
/// We zero out stochastic noise / scintillation so the peak ratio
/// is a deterministic function of the polarization scaling.
#[test]
fn polarization_vv_vs_hh_changes_target_amp() {
    let base_config = RadarSimConfig {
        pulse_count: 16,
        ..RadarSimConfig::default()
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    // Zero stochastic terms so the integrated peak is a clean
    // function of `target_amp * pol_scale`.
    noise.amplitude_scintillation_sigma = 0.0;
    noise.phase_noise_std_rad = 0.0;
    noise.rfi_probability = 0.0;
    noise.clutter_sigma = 0.0;
    noise.ground_glint_count = 0;
    noise.awgn_sigma = 0.001;

    let config_vv = RadarSimConfig {
        pol_tx_sequence: Some(vec![Polarization::Vv]),
        ..base_config.clone()
    };
    let config_hh = RadarSimConfig {
        pol_tx_sequence: Some(vec![Polarization::Hh]),
        ..base_config
    };

    let episode_vv = synthesize_takeoff_episode(
        config_vv,
        TakeoffProfile::default(),
        noise,
        EpisodeSeed(2026),
    );
    let episode_hh = synthesize_takeoff_episode(
        config_hh,
        TakeoffProfile::default(),
        noise,
        EpisodeSeed(2026),
    );

    let peak_vv = episode_vv
        .integrated_range_profile
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let peak_hh = episode_hh
        .integrated_range_profile
        .iter()
        .copied()
        .fold(0.0f32, f32::max);

    assert!(
        peak_vv > 0.0 && peak_hh > 0.0,
        "both peaks must be positive"
    );

    // HH amplitude scale = 10^(+1/20) ≈ 1.122 (see
    // `polarization_amplitude_scale`). The integrated profile is a
    // mean of magnitudes, which scales linearly with target_amp;
    // therefore peak_hh / peak_vv should be ~1.122 (i.e. ~+1 dB).
    let ratio_db = 20.0 * (peak_hh / peak_vv).log10();
    assert!(
        ratio_db.abs() >= 0.5,
        "VV-only vs HH-only integrated peaks must differ by ≥0.5 dB \
         (the +1 dB HH scaling); got ratio_db = {ratio_db:.3} \
         (peak_vv = {peak_vv:.6}, peak_hh = {peak_hh:.6})"
    );
    // And we expect the sign to be positive: HH > VV by the
    // first-order proxy.
    assert!(
        peak_hh > peak_vv,
        "HH peak should exceed VV peak by ~1 dB; got peak_hh={peak_hh:.6} \
         vs peak_vv={peak_vv:.6}"
    );
}
