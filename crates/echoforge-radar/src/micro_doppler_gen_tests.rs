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
