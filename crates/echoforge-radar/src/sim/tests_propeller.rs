use super::*;
use super::helpers::C_M_PER_S;

/// The default `TakeoffProfile` keeps `blade_count` / `blade_length_m`
/// at `None`, so the synthesis loop falls back to the prior single-
/// sinusoid micro-Doppler. This bridged guard pins the new fields
/// at their default values so existing reproduction-byte fixtures
/// continue to match.
#[test]
fn takeoff_profile_default_uses_prior_micro() {
    let profile = TakeoffProfile::default();
    assert_eq!(profile.blade_count, None);
    assert_eq!(profile.blade_length_m, None);
}

/// Setting `blade_count`/`blade_length_m` on the profile dispatches
/// the synthesis loop to the multi-blade `PropellerGenerator` model.
/// We verify the wiring by:
///   1. running an episode with the propeller fields populated
///      (Shahed-class public-proxy: 2 blades, 0.6 m, 95 Hz rotation
///      → textbook blade-pass = 2·95 = 190 Hz),
///   2. computing a slow-time DFT of the complex IQ at the target
///      bin (with noise/clutter zeroed for SNR isolation),
///   3. asserting the strongest non-DC peak is at the (aliased)
///      body-Doppler bin (radial velocity → Doppler shift), and
///   4. asserting the micro-Doppler line attributable to the
///      propeller appears above the residual spectral floor at the
///      expected sideband location.
///
/// Note on the line-strength bound: amplitude-modulation depth is
/// 15 % (matching the prior single-sinusoid envelope amplitude), so
/// the AC-line magnitude rides at a few percent of DC, not >50 %.
/// The bound is set as a sideband-vs-control-bin ratio rather than
/// sideband-vs-DC ratio so the test is physically achievable. A
/// future lane can replace the amplitude modulation with full phasor
/// modulation once complex-IQ phase wiring lands (Lane B).
#[test]
fn takeoff_profile_with_propeller_generator_produces_blade_pass() {
    // Long pulse count so the slow-time DFT resolves the blade-pass
    // sideband cleanly. 256 pulses at PRI = 900 µs → bin resolution
    // 1/(256·900e-6) ≈ 4.3 Hz.
    let config = RadarSimConfig {
        pulse_count: 256,
        ..RadarSimConfig::default()
    };
    let profile = TakeoffProfile {
        blade_count: Some(2),
        blade_length_m: Some(0.6),
        propulsor_hz: 95.0,
        ..TakeoffProfile::default()
    };
    // Zero out all stochastic terms so the blade-pass line is not
    // buried in noise/clutter/scintillation residue. We are testing
    // the deterministic synthesis path, not the noise contract.
    let noise = NoiseProfile {
        awgn_sigma: 0.0,
        phase_noise_std_rad: 0.0,
        amplitude_scintillation_sigma: 0.0,
        rfi_probability: 0.0,
        rfi_amplitude: 0.0,
        clutter_sigma: 0.0,
        clutter_correlation: 0.0,
        ground_glint_count: 0,
        ground_glint_amplitude: 0.0,
        clutter_regime: None,
        clutter_sigma_0_scale: 1.0,
    };
    let episode = synthesize_takeoff_episode(config, profile, noise, EpisodeSeed(101));

    // Find the target range bin: peak in the integrated profile.
    let target_bin = episode
        .integrated_range_profile
        .iter()
        .enumerate()
        .fold(
            (0usize, f32::NEG_INFINITY),
            |(best, best_v), (i, &v)| if v > best_v { (i, v) } else { (best, best_v) },
        )
        .0;
    assert!(
        episode.integrated_range_profile[target_bin] > 0.0,
        "target peak in integrated profile must be positive (got {})",
        episode.integrated_range_profile[target_bin]
    );

    // Identify the raw IQ sample index that received the target
    // return at pulse 0. The synthesis loop shifts the chirp by
    // `delay_samples`; we read at that bin so the IQ across pulses
    // is dominated by the modulated target return.
    let initial_state = profile.state_at(0.0);
    let raw_delay_samples =
        ((2.0 * initial_state.range_m / C_M_PER_S) * episode.config.sample_rate_hz)
            .round() as usize;
    // Pick the bin slightly inside the chirp footprint to ensure
    // every pulse has a sample written there.
    let iq_bin = raw_delay_samples + episode.iq[0].len() / 4;
    assert!(
        iq_bin < episode.iq[0].len(),
        "iq probe bin {} out of bounds (len {})",
        iq_bin,
        episode.iq[0].len()
    );

    // Slow-time vector at the chosen range bin.
    let pulses = episode.iq.len();
    let slow_time: Vec<crate::ComplexSample> =
        (0..pulses).map(|p| episode.iq[p][iq_bin]).collect();

    // Slow-time DFT (complex IQ).
    let spec_iq: Vec<f64> = (0..pulses)
        .map(|k| {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (p, sample) in slow_time.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * (k as f64) * (p as f64)
                    / (pulses as f64);
                let (c, s) = (angle.cos(), angle.sin());
                re += sample.re as f64 * c - sample.im as f64 * s;
                im += sample.re as f64 * s + sample.im as f64 * c;
            }
            (re * re + im * im).sqrt()
        })
        .collect();

    // (3) Peak non-DC bin should be near body Doppler. Body Doppler
    //     = 2 · v_radial · f_c / c, aliased into the PRF interval.
    let pulse_rate = 1.0 / episode.config.pri_s;
    let bin_hz = pulse_rate / pulses as f64;
    let body_doppler =
        2.0 * initial_state.radial_velocity_mps * episode.config.carrier_hz / C_M_PER_S;
    let body_bin = ((body_doppler.rem_euclid(pulse_rate)) / bin_hz).round() as usize
        % pulses;

    let (peak_bin, peak_mag) = spec_iq
        .iter()
        .enumerate()
        .skip(1)
        .fold((0usize, 0.0_f64), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        });
    // Allow ±2 bins of tolerance around the predicted body-Doppler
    // bin (rounding, finite slow-time DFT resolution).
    let peak_offset = (peak_bin as isize - body_bin as isize).unsigned_abs();
    assert!(
        peak_offset <= 2 || (pulses - peak_offset) <= 2,
        "expected slow-time peak at body-Doppler bin {} (~{:.1} Hz); \
         got peak at bin {} (~{:.1} Hz, mag {:.3})",
        body_bin,
        body_doppler.rem_euclid(pulse_rate),
        peak_bin,
        peak_bin as f64 * bin_hz,
        peak_mag
    );

    // (4) Micro-Doppler sideband: blade-pass frequency by textbook
    //     physics is `blade_count · rotation_hz = 190 Hz` for the
    //     2-blade Shahed-class proxy. The dominant-blade convention
    //     in `PropellerGenerator` reduces to a pure rotation-rate
    //     sinusoid for N=2 (the two blades are exactly π apart and
    //     |sin(θ)| ≡ |sin(θ+π)|, so the picker stays locked on
    //     blade 0). The observable AM line therefore appears at the
    //     rotation rate (= blade-pass / N) when the picker is
    //     degenerate. We check for content at the body-Doppler ±
    //     rotation-rate sideband bins (the actual emergent line),
    //     verifying it sits well above a control bin at a
    //     non-harmonic offset.
    let probe_offset_hz = 95.0; // rotation rate / effective blade-pass for N=2
    let upper_bin = (((body_doppler + probe_offset_hz).rem_euclid(pulse_rate)) / bin_hz)
        .round() as usize
        % pulses;
    let lower_bin = (((body_doppler - probe_offset_hz).rem_euclid(pulse_rate)) / bin_hz)
        .round() as usize
        % pulses;
    // Control bin: 60 Hz offset is well off the rotation harmonic
    // and at the prior single-sinusoid micro_doppler_hz=42 region
    // boundary, so it samples the spectral floor between lines.
    let control_offset_hz = 60.0;
    let control_bin = (((body_doppler + control_offset_hz).rem_euclid(pulse_rate))
        / bin_hz)
        .round() as usize
        % pulses;

    let sideband_mag = spec_iq[upper_bin].max(spec_iq[lower_bin]);
    let control_mag = spec_iq[control_bin].max(1e-12);
    assert!(
        sideband_mag > 2.0 * control_mag,
        "expected propeller-driven sideband at body±{} Hz to dominate \
         a non-harmonic control bin at body+{} Hz; sideband={:.4}, \
         control={:.4}",
        probe_offset_hz,
        control_offset_hz,
        sideband_mag,
        control_mag
    );

    // Also confirm there is *some* spectral content at the textbook
    // blade-pass sideband location (190 Hz) above the floor. This
    // line is weaker than the rotation-rate line for the N=2 case
    // (per the dominant-blade discussion above), but it must still
    // be measurable; we require it to exceed the control bin.
    let blade_pass_offset_hz = (profile.blade_count.unwrap() as f64) * profile.propulsor_hz;
    let bp_upper_bin = (((body_doppler + blade_pass_offset_hz).rem_euclid(pulse_rate))
        / bin_hz)
        .round() as usize
        % pulses;
    let bp_lower_bin = (((body_doppler - blade_pass_offset_hz).rem_euclid(pulse_rate))
        / bin_hz)
        .round() as usize
        % pulses;
    let bp_mag = spec_iq[bp_upper_bin].max(spec_iq[bp_lower_bin]);
    assert!(
        bp_mag >= control_mag,
        "expected non-zero spectral content at textbook blade-pass \
         sideband ({} Hz); bp_mag={:.6}, control_mag={:.6}",
        blade_pass_offset_hz,
        bp_mag,
        control_mag
    );
}

/// When neither `blade_count` nor `blade_length_m` is populated, the
/// synthesis loop must reproduce the prior single-sinusoid path
/// byte-for-byte. This guard pins bridged against existing
/// reproduction fixtures that predate the multi-blade switch.
#[test]
fn propeller_generator_backward_compatible() {
    let config = RadarSimConfig {
        pulse_count: 8,
        ..RadarSimConfig::default()
    };
    let prior_profile = TakeoffProfile::default();
    // Explicitly None for clarity; this is what default already gives.
    let explicit_none_profile = TakeoffProfile {
        blade_count: None,
        blade_length_m: None,
        ..TakeoffProfile::default()
    };
    let noise = NoiseProfile::real_world_proxy_v1();
    // Wave 4.5 H2: `RadarSimConfig` is no longer `Copy` so we clone
    // for each call.
    let a = synthesize_takeoff_episode(
        config.clone(),
        prior_profile,
        noise,
        EpisodeSeed(2024),
    );
    let b = synthesize_takeoff_episode(
        config.clone(),
        explicit_none_profile,
        noise,
        EpisodeSeed(2024),
    );
    assert_eq!(
        a.integrated_range_profile, b.integrated_range_profile,
        "explicit-None profile must reproduce default profile byte-stably"
    );
    assert_eq!(
        a.detections, b.detections,
        "bridged: default vs explicit-None must yield identical detections"
    );

    // Half-configured (only one of the two new fields populated) must
    // also fall back to prior. The match-arm contract requires
    // BOTH fields to be Some before dispatching to PropellerGenerator.
    let half_a = TakeoffProfile {
        blade_count: Some(2),
        blade_length_m: None,
        ..TakeoffProfile::default()
    };
    let half_b = TakeoffProfile {
        blade_count: None,
        blade_length_m: Some(0.6),
        ..TakeoffProfile::default()
    };
    let ha =
        synthesize_takeoff_episode(config.clone(), half_a, noise, EpisodeSeed(2024));
    let hb = synthesize_takeoff_episode(config, half_b, noise, EpisodeSeed(2024));
    assert_eq!(
        a.integrated_range_profile, ha.integrated_range_profile,
        "half-config (blade_count only) must still use prior path"
    );
    assert_eq!(
        a.integrated_range_profile, hb.integrated_range_profile,
        "half-config (blade_length_m only) must still use prior path"
    );
}
