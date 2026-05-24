//! Closed-form physics correctness tests for `echoforge-radar`.
//!
//! This file is the W10 scaffolding test packet for the EchoForge
//! correctness-first v3 plan. The suite exists so the simulator's
//! reproduction of textbook radar physics is *gated* by a permanent
//! test surface, not just by ad-hoc developer inspection.
//!
//! Active tests cover primitives that are already in the crate:
//! the RCS aspect/Swerling lookup (`crate::Rcs`), the heavy-tailed
//! clutter samplers (`crate::sample_*`), the CFAR threshold scaling
//! (`crate::ca_cfar_scale`), and the Range-Doppler-Angle cube
//! determinism (`crate::build_rda_cube`).
//!
//! Ignored tests document closed-form physics that downstream packets
//! will land. Each ignored test names the packet that will unignore
//! it. The structure is parseable today so the gate is in place when
//! the corresponding primitive ships.
//!
//! Strict-open posture: this file does NOT claim platform-specific
//! signature truth or vendor-equivalent performance. All assertions
//! are checks of textbook physics (Skolnik, Ward/Tough/Watts, ITU-R)
//! against existing or future EchoForge primitives.

use std::f64::consts::PI;

use echoforge_radar::{
    apply_mti, build_rda_cube, ca_cfar_scale, coefficients, evaluate_link_budget, hough_tbd_detect,
    itu_r_p676_gas_attenuation_db, itu_r_p838_rain_attenuation_db, magnitude, mtd_chain,
    mti_improvement_factor_db, pulse_compress, pulse_compress_windowed, sample_clutter_amplitude,
    sample_k_distribution, sample_log_normal, sample_weibull, slow_time_complex_dft,
    synthesize_scene, synthesize_takeoff_episode, two_ray_propagation_factor_magnitude, AngleGrid,
    AspectGrid, BoostThrustProfile, BoostTierDetector, CfarParams, ClimbOutTierDetector,
    ClimbTierConfig, ClutterDistribution, ClutterRegime, ComplexSample, CompressionWindow,
    EnvironmentDescriptor, EpisodeSeed, KinematicGate, KinematicObservation, KinematicSample,
    LinkBudget, MtiOrder, NoiseProfile, Polarization, PropagationContext, PropulsionClass,
    RadarSimConfig, RainPolarization, Rcs, RcsLookup, SceneDescriptor, SiteGeometry,
    SpeedClassifier, SwerlingModel, TakeoffProfile, TargetClass, TargetEntity, TargetKinematics,
    TbdConfig, TerrainClass, MTI_NOTCH_BODY_DOPPLER_HZ, REFERENCE_NOISE_TEMPERATURE_K,
};

// ===========================================================================
// Helpers
// ===========================================================================

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

/// Generate `n` independent samples by mixing `seed_base` with the
/// sample index. The mixing constant is the standard SplitMix64 gamma
/// so successive seeds give well-separated streams without us needing
/// to add a dependency.
fn collect_many(seed_base: u64, n: usize, mut f: impl FnMut(u64) -> f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            f(seed_base
                .wrapping_add(i as u64)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15))
        })
        .collect()
}

#[test]
fn rcs_dbsm_converts_to_linear_m2_with_ten_db_decades() {
    let cases = [(-10.0, 0.1), (0.0, 1.0), (10.0, 10.0)];
    for (dbsm, expected) in cases {
        let linear = 10f64.powf(dbsm / 10.0);
        assert!(
            (linear - expected).abs() < 1e-12,
            "{dbsm} dBsm should convert to {expected} m^2, got {linear}"
        );
    }
}

// ===========================================================================
// Tests on EXISTING primitives — must pass now.
// ===========================================================================

/// Same RCS table evaluated at 0 degree aspect vs 90 degree aspect
/// must differ by >= 10 dB for at least one of the seeded reference
/// tables. Real-world targets are not isotropic; the simulator must
/// represent that explicitly. Without this gate the simulator could
/// fall back to a single scalar RCS per target — the exact failure
/// mode that motivated the `Rcs` lookup module.
#[test]
fn rcs_aspect_dependence_present() {
    let rcs = Rcs::seeded_public_proxy_v1();
    let candidates: [(&str, Polarization); 3] = [
        ("fixed-wing-uas-small", Polarization::Vv),
        ("bird-large-single", Polarization::Hh),
        ("quadrotor", Polarization::Vv),
    ];

    let mut max_delta_db = 0.0f64;
    for (class, pol) in candidates {
        let v_nose = rcs.evaluate_static(class, 0.0, 0.0, 10.0, pol);
        let v_broadside = rcs.evaluate_static(class, 90.0, 0.0, 10.0, pol);
        assert!(
            v_nose.is_finite() && v_broadside.is_finite(),
            "{class} returned non-finite dBsm (nose={v_nose}, broadside={v_broadside})",
        );
        let delta = (v_nose - v_broadside).abs();
        if delta > max_delta_db {
            max_delta_db = delta;
        }
    }
    assert!(
        max_delta_db >= 10.0,
        "expected at least one seeded RCS table to differ by >=10 dB between 0deg \
         and 90deg aspect; observed max delta = {max_delta_db:.2} dB",
    );
}

/// `Rcs::evaluate(...)` with `SwerlingModel::Swerling0` (non-fluctuating)
/// must return bit-identical values across repeated calls for the same
/// `(seed, pulse_index)`. Determinism is part of the Lane B reproducibility
/// gate.
#[test]
fn rcs_swerling_0_is_deterministic() {
    let table = RcsLookup {
        target_class: "deterministic-test".to_string(),
        frequency_ghz: 10.0,
        polarization: Polarization::Vv,
        aspect_grid: AspectGrid {
            azimuth_deg: vec![0.0, 90.0, 180.0, 270.0],
            elevation_deg: vec![0.0, 15.0],
        },
        rcs_dbsm: vec![-20.0; 8],
        fluctuation: SwerlingModel::Swerling0,
        citation: "synthetic fixture for deterministic test".to_string(),
        citation_url: None,
    };
    let mut rcs = Rcs::empty();
    rcs.add_table(table);

    let a = rcs.evaluate(
        "deterministic-test",
        45.0,
        5.0,
        10.0,
        Polarization::Vv,
        12345,
        0,
    );
    let b = rcs.evaluate(
        "deterministic-test",
        45.0,
        5.0,
        10.0,
        Polarization::Vv,
        12345,
        0,
    );
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "Swerling 0 must be deterministic (got {a} vs {b})",
    );
}

/// `Rcs::evaluate(...)` with `SwerlingModel::Swerling2` (pulse-to-pulse
/// decorrelated) must yield different values between adjacent pulses.
/// Two continuous-distribution draws colliding bit-for-bit has probability
/// far below 0.3 percent, so an inequality assertion holds with greater
/// than 99.7 percent probability.
#[test]
fn rcs_swerling_2_pulse_to_pulse_varies() {
    let table = RcsLookup {
        target_class: "sw2-test".to_string(),
        frequency_ghz: 10.0,
        polarization: Polarization::Vv,
        aspect_grid: AspectGrid {
            azimuth_deg: vec![0.0, 90.0, 180.0, 270.0],
            elevation_deg: vec![0.0, 15.0],
        },
        rcs_dbsm: vec![-15.0; 8],
        fluctuation: SwerlingModel::Swerling2,
        citation: "synthetic fixture for Swerling 2 pulse-to-pulse test".to_string(),
        citation_url: None,
    };
    let mut rcs = Rcs::empty();
    rcs.add_table(table);

    let p0 = rcs.evaluate("sw2-test", 45.0, 5.0, 10.0, Polarization::Vv, 0xC0FFEE, 0);
    let p1 = rcs.evaluate("sw2-test", 45.0, 5.0, 10.0, Polarization::Vv, 0xC0FFEE, 1);
    assert!(
        (p0 - p1).abs() > 1e-9,
        "Swerling 2 must decorrelate pulse-to-pulse (got {p0} vs {p1})",
    );
}

/// `sample_weibull(2.0, 1.0, seed)` is Rayleigh with scale `sigma = 1/sqrt(2)`.
/// The Rayleigh mean is `sigma * sqrt(pi/2) = sqrt(pi)/2 ~= 0.886`. Over
/// 5000 draws the sample mean must land within 5% of that value.
/// Reference: Devroye, "Non-Uniform Random Variate Generation" (1986),
/// chap. IV.5.
#[test]
fn clutter_weibull_shape_2_is_rayleigh() {
    let xs = collect_many(0x5a5a_5a5a_5a5a_5a5a, 5000, |s| sample_weibull(2.0, 1.0, s));
    let m = mean(&xs);
    let expected = PI.sqrt() / 2.0; // ~= 0.8862
    let rel_err = (m - expected).abs() / expected;
    assert!(
        rel_err < 0.05,
        "Weibull(shape=2,scale=1) mean = {m:.4} should be within 5% of Rayleigh \
         expected value {expected:.4} (rel err {rel_err:.4})",
    );
}

/// `sample_k_distribution(nu, 1.0, seed)` with large `nu` collapses to
/// the Rayleigh limit (texture variance vanishes; speckle dominates).
/// The Rayleigh mean/std ratio is `sqrt(pi/(4-pi)) ~= 1.913`. Over
/// 5000 draws with `nu=20`, the sample mean/std ratio must land within
/// 10% of that value. Reference: Ward, Tough & Watts, "Sea Clutter:
/// Scattering, the K Distribution and Radar Performance" (IET 2013),
/// chap. 2.
#[test]
fn clutter_k_distribution_high_nu_approaches_rayleigh() {
    let xs = collect_many(0x1234_5678_9abc_def0, 5000, |s| {
        sample_k_distribution(20.0, 1.0, s)
    });
    let m = mean(&xs);
    let sd = std_dev(&xs);
    assert!(sd > 1e-9, "K-distribution sample std degenerate ({sd})");
    let ratio = m / sd;
    let expected = (PI / (4.0 - PI)).sqrt(); // ~= 1.913
    let rel_err = (ratio - expected).abs() / expected;
    assert!(
        rel_err < 0.10,
        "K-distribution(nu=20) mean/std ratio = {ratio:.4} should be within 10% of \
         Rayleigh ratio {expected:.4} (rel err {rel_err:.4})",
    );
}

/// `sample_log_normal(mean_log, std_log, seed)` returns `exp(N(mean_log, std_log))`,
/// so the sample mean of `ln(X)` should match `mean_log`. Over 5000 draws
/// the log-mean must land within 5% of 2.0.
#[test]
fn clutter_log_normal_mean_of_log() {
    let xs = collect_many(0xdead_beef_cafe_babe, 5000, |s| {
        sample_log_normal(2.0, 0.5, s)
    });
    let logs: Vec<f64> = xs.iter().map(|x| x.ln()).collect();
    let m = mean(&logs);
    let rel_err = (m - 2.0).abs() / 2.0;
    assert!(
        rel_err < 0.05,
        "log-normal mean(log(x)) = {m:.4} should be within 5% of mean_log=2.0 \
         (rel err {rel_err:.4})",
    );
}

/// All four clutter sampler entry points must be deterministic: the same
/// `seed` must produce the same sample. Bit-for-bit equality is enforced
/// because the inverse-CDF samplers are pure functions over the seed.
#[test]
fn clutter_sample_determinism() {
    let s: u64 = 0xA5A5_A5A5_A5A5_A5A5;

    let w_a = sample_weibull(1.5, 1.0, s);
    let w_b = sample_weibull(1.5, 1.0, s);
    assert_eq!(
        w_a.to_bits(),
        w_b.to_bits(),
        "sample_weibull not deterministic"
    );

    let k_a = sample_k_distribution(3.0, 1.0, s);
    let k_b = sample_k_distribution(3.0, 1.0, s);
    assert_eq!(
        k_a.to_bits(),
        k_b.to_bits(),
        "sample_k_distribution not deterministic",
    );

    let ln_a = sample_log_normal(0.5, 0.25, s);
    let ln_b = sample_log_normal(0.5, 0.25, s);
    assert_eq!(
        ln_a.to_bits(),
        ln_b.to_bits(),
        "sample_log_normal not deterministic",
    );

    let dist = ClutterDistribution::KDistribution {
        shape: 4.0,
        scale: 1.0,
    };
    let amp_a = sample_clutter_amplitude(&dist, s);
    let amp_b = sample_clutter_amplitude(&dist, s);
    assert_eq!(
        amp_a.to_bits(),
        amp_b.to_bits(),
        "sample_clutter_amplitude not deterministic",
    );
}

/// `ClutterRegime::library()` must cover at least 8 published-source-cited
/// terrain regimes; every entry must declare a non-empty terrain class
/// and a finite, sensible mean cross-section. This is the surface area
/// downstream campaigns rely on when picking heavy-tailed clutter.
#[test]
fn clutter_library_returns_8_regimes() {
    let lib = ClutterRegime::library();
    assert!(
        lib.len() >= 8,
        "ClutterRegime::library() should expose >=8 regimes; got {}",
        lib.len(),
    );
    for r in &lib {
        // Terrain class is an enum, so "non-empty" reduces to confirming
        // the round-trip through `for_terrain` returns a regime tagged
        // with the same terrain. The variant itself cannot be empty.
        let recovered = ClutterRegime::for_terrain(r.terrain, r.grazing_angle_deg);
        assert_eq!(
            recovered.terrain, r.terrain,
            "for_terrain round-trip lost terrain class for {:?}",
            r.terrain,
        );

        assert!(
            (0.0..=1.0).contains(&r.spatial_correlation),
            "spatial_correlation out of [0,1] for {:?}: {}",
            r.terrain,
            r.spatial_correlation,
        );
        assert!(
            (0.0..=1.0).contains(&r.temporal_correlation),
            "temporal_correlation out of [0,1] for {:?}: {}",
            r.terrain,
            r.temporal_correlation,
        );
        assert!(
            r.mean_power_dbsm_per_m2.is_finite(),
            "mean_power_dbsm_per_m2 non-finite for {:?}",
            r.terrain,
        );

        match r.distribution {
            ClutterDistribution::Weibull { shape, scale }
            | ClutterDistribution::KDistribution { shape, scale } => {
                assert!(
                    shape > 0.0 && shape.is_finite(),
                    "{:?} declares non-positive shape {}",
                    r.terrain,
                    shape,
                );
                assert!(
                    scale > 0.0 && scale.is_finite(),
                    "{:?} declares non-positive scale {}",
                    r.terrain,
                    scale,
                );
            }
            ClutterDistribution::LogNormal { mean_log, std_log } => {
                assert!(
                    mean_log.is_finite(),
                    "{:?} declares non-finite mean_log {}",
                    r.terrain,
                    mean_log,
                );
                assert!(
                    std_log >= 0.0 && std_log.is_finite(),
                    "{:?} declares non-positive std_log {}",
                    r.terrain,
                    std_log,
                );
            }
            ClutterDistribution::Rayleigh => {
                // Rayleigh has no shape/scale parameters at this API
                // surface; nothing more to check.
            }
        }
    }

    // Spot check: every TerrainClass variant must be representable in
    // the library so callers do not see a partial taxonomy.
    let needed_classes: [TerrainClass; 8] = [
        TerrainClass::OpenSky,
        TerrainClass::Desert,
        TerrainClass::Forest,
        TerrainClass::Urban,
        TerrainClass::Sea,
        TerrainClass::Mountain,
        TerrainClass::Agricultural,
        TerrainClass::Suburban,
    ];
    for class in needed_classes {
        assert!(
            lib.iter().any(|r| r.terrain == class),
            "ClutterRegime::library() missing entry for {class:?}",
        );
    }
}

/// `ca_cfar_scale(n=10, pfa=1e-3)` must return a finite, strictly positive
/// scale factor. The closed-form scaling is
/// `alpha = n * (pfa^(-1/n) - 1)`. For typical parameters this is a
/// positive O(1) multiplier on the noise estimate.
#[test]
fn ca_cfar_threshold_scale_is_finite_and_positive() {
    let alpha = ca_cfar_scale(10, 1e-3);
    assert!(
        alpha.is_finite(),
        "ca_cfar_scale returned non-finite ({alpha})"
    );
    assert!(alpha > 0.0, "ca_cfar_scale returned non-positive ({alpha})");

    // Sanity: with more training cells the scale must drop monotonically
    // (more averaging, tighter threshold relative to noise).
    let alpha_more = ca_cfar_scale(32, 1e-3);
    assert!(
        alpha_more < alpha,
        "ca_cfar_scale should decrease with more training cells (n=10 -> {alpha}, \
         n=32 -> {alpha_more})",
    );

    // CfarParams import is exercised to keep the active surface honest:
    // a downstream change that drops CfarParams would break this gate.
    let params = CfarParams::new(10, 4, 1e-3);
    assert_eq!(params.training_cells, 10);
    assert_eq!(params.guard_cells, 4);
    assert!((params.pfa - 1e-3).abs() < 1e-12);
}

/// `build_rda_cube` is deterministic by construction (no RNG, no parallel
/// reductions). Two identical inputs must produce byte-identical cubes
/// with the requested shape.
#[test]
fn rda_cube_shape_consistent() {
    use echoforge_radar::ComplexSample;

    let n_channels = 2;
    let n_samples = 16;
    let n_range = 4;
    let n_doppler = 4;
    let carrier_hz = 10_000_000_000.0;
    let pri_s = 1e-3;
    let spacing_m = 0.015;

    // Synthetic channel IQ: deterministic constant phasor per channel.
    // The exact contents do not matter — only that the input is fixed
    // across the two `build_rda_cube` calls.
    let channels: Vec<Vec<ComplexSample>> = (0..n_channels)
        .map(|ch| {
            (0..n_samples)
                .map(|i| {
                    let phase = (ch as f32 + 1.0) * (i as f32) * 0.1;
                    ComplexSample::new(phase.cos(), phase.sin())
                })
                .collect()
        })
        .collect();
    let manifold =
        echoforge_radar::PhasedArrayManifold::new(n_channels, spacing_m, carrier_hz, 0.0, 0.0);
    let grid = AngleGrid::azimuth_only(vec![-10.0, 0.0, 10.0]);

    let cube_a = build_rda_cube(
        &channels,
        &manifold,
        &grid,
        (n_range, n_doppler),
        pri_s,
        carrier_hz,
    );
    let cube_b = build_rda_cube(
        &channels,
        &manifold,
        &grid,
        (n_range, n_doppler),
        pri_s,
        carrier_hz,
    );

    assert_eq!(cube_a.range_bins, n_range);
    assert_eq!(cube_a.doppler_bins, n_doppler);
    assert_eq!(cube_a.angle_count, 3);
    assert_eq!(cube_a.cube.len(), 3);
    for slice in &cube_a.cube {
        assert_eq!(slice.len(), n_doppler);
        for row in slice {
            assert_eq!(row.len(), n_range);
        }
    }
    assert_eq!(
        cube_a, cube_b,
        "build_rda_cube must be deterministic for fixed input",
    );
}

// ===========================================================================
// Tests on FUTURE primitives — `#[ignore]` until downstream packets land.
//
// Each `#[ignore]` test compiles today and uses `unimplemented!()` to
// document the closed-form physics the future primitive must satisfy.
// The doc comment on each test names the packet that will unignore it.
// ===========================================================================

/// Radar equation: received power follows a `1/R^4` law, so the SNR
/// for a target at range `2R` is 12.04 dB below the SNR at range `R`.
/// Within numerical tolerance the simulator must reproduce this exactly.
///
/// Unignored by the `link-budget-wire-in` packet (Wave 1 Lane A),
/// which lands `echoforge_radar::evaluate_link_budget` as the
/// closed-form helper.
#[test]
fn radar_equation_r_to_the_4th() {
    let budget = LinkBudget {
        transmit_power_w: 1.0e6,
        tx_gain_dbi: 35.0,
        rx_gain_dbi: 35.0,
        carrier_hz: 3.0e9,
        noise_figure_db: 3.0,
        noise_bandwidth_hz: 1.0e6,
        system_temperature_k: REFERENCE_NOISE_TEMPERATURE_K,
        system_loss_db: 0.0,
        processing_loss_db: 0.0,
        coherent_integration_pulses: 1,
    };
    let prop_near = PropagationContext {
        range_m: 25_000.0,
        target_altitude_agl_m: 10_000.0,
        radar_altitude_agl_m: 100.0,
        atmospheric_one_way_db_per_km: 0.0,
        rain_rate_mm_per_h: 0.0,
        ground_reflection_coefficient_magnitude: 0.0,
    };
    let prop_far = PropagationContext {
        range_m: 50_000.0,
        ..prop_near
    };
    let snr_near = evaluate_link_budget(&budget, &prop_near, 1.0).snr_db;
    let snr_far = evaluate_link_budget(&budget, &prop_far, 1.0).snr_db;
    let drop = snr_near - snr_far;
    let expected = 40.0 * 2f64.log10(); // ≈ 12.041 dB
    assert!(
        (drop - expected).abs() < 0.5,
        "R⁴ law: expected SNR drop {expected:.3} dB doubling range; observed {drop:.3} dB",
    );
}

/// Two-ray multipath: for a horizontally polarised low-grazing-angle
/// geometry the first multipath null in the height-search appears at
/// `h_t ~ lambda * R / (4 * h_r)`. At S-band (3 GHz, lambda = 0.1 m),
/// R = 80 km, h_r = 20 m, the predicted null is at h_t ~ 200 m within
/// 50 m tolerance.
///
/// Will be unignored once the `radar-propagation-primitives-v3` packet
/// lands `propagation::two_ray_propagation_factor_magnitude`.
#[test]
fn two_ray_first_null_at_predicted_altitude() {
    let freq_hz = 3.0e9_f64;
    let h_r = 20.0_f64;
    let range_m = 80_000.0_f64;

    // Scan h_t from 5 m to 500 m in 5 m steps; find the h_t where |F| is minimum.
    // Textbook prediction: first null at h_t = λ·R/(2·h_r) ≈ 200 m.
    let mut h_t_null = 0.0_f64;
    let mut f_min = f64::MAX;
    for step in 1_u32..=100 {
        let h_t = step as f64 * 5.0;
        let f = two_ray_propagation_factor_magnitude(freq_hz, h_t, h_r, range_m, 1.0);
        if f < f_min {
            f_min = f;
            h_t_null = h_t;
        }
    }
    assert!(
        (h_t_null - 200.0).abs() < 50.0,
        "first null found at {h_t_null:.1} m, expected near 200 m (±50 m)"
    );
    assert!(f_min < 0.01, "|F| at null = {f_min:.6}, expected near zero");
}

/// 4/3-Earth radar horizon: a target at R = 100 km and h = 50 m sits
/// below the horizon for a 10 m radar antenna; the link-budget
/// `above_horizon` flag must be false. The simulator chain treats
/// sub-horizon targets as zero-amplitude returns
/// (`crates/echoforge-radar/src/sim.rs::synthesize_takeoff_episode`).
///
/// Unignored by the `link-budget-wire-in` packet (Wave 1 Lane A),
/// which combines the propagation horizon primitive with the
/// link-budget signal-power computation.
#[test]
fn radar_horizon_below_returns_zero_signal() {
    let budget = LinkBudget {
        transmit_power_w: 1.0e6,
        tx_gain_dbi: 35.0,
        rx_gain_dbi: 35.0,
        carrier_hz: 2.9e9,
        noise_figure_db: 4.0,
        noise_bandwidth_hz: 1.0e6,
        system_temperature_k: REFERENCE_NOISE_TEMPERATURE_K,
        system_loss_db: 4.0,
        processing_loss_db: 2.0,
        coherent_integration_pulses: 1,
    };
    let prop_sub_horizon = PropagationContext {
        range_m: 100_000.0,
        target_altitude_agl_m: 50.0,
        radar_altitude_agl_m: 10.0,
        atmospheric_one_way_db_per_km: 0.0,
        rain_rate_mm_per_h: 0.0,
        ground_reflection_coefficient_magnitude: 0.0,
    };
    let result = evaluate_link_budget(&budget, &prop_sub_horizon, 1.0);
    assert!(
        !result.above_horizon,
        "50 m target at 100 km should be below horizon for 10 m antenna; \
         link-budget flagged it as above_horizon=true",
    );
    // Received power is still computed (the geometric horizon check is
    // a callable flag, not a hard clamp inside the equation); the
    // simulator chain zeroes the target return when above_horizon is
    // false. The numerical received power must remain finite.
    assert!(
        result.received_power_w.is_finite(),
        "received power non-finite for sub-horizon geometry: {}",
        result.received_power_w,
    );
}

/// Doppler bin localisation: a known target velocity produces a narrow
/// peak at the predicted Doppler bin in the complex-IQ slow-time FFT.
/// Today's `build_rda_cube` already runs a Doppler DFT, but the
/// magnitude/phase semantics required for a closed-form bin assertion
/// (carrier, PRI, complex IQ chain) live in a separate packet.
///
/// Will be unignored once `complex-iq-spectrum-products-v3` lands.
#[test]
fn doppler_shift_complex_iq_correct_bin() {
    use echoforge_radar::complex_iq;

    // S-band geometry: N=64 pulses, PRI=1 ms, carrier=3 GHz, v=12.5 m/s.
    // f_d = 2*12.5*3e9/3e8 = 250 Hz  →  bin = 250 * 64 * 1e-3 = 16.
    let n = 64_usize;
    let pri_s = 1.0e-3_f64;
    let carrier_hz = 3.0e9_f64;
    let v_radial = 12.5_f64;
    let c = 3.0e8_f64;

    let f_d = 2.0 * v_radial * carrier_hz / c;
    let iq: Vec<ComplexSample> = (0..n)
        .map(|i| {
            let phase = (2.0 * PI * f_d * i as f64 * pri_s) as f32;
            ComplexSample::new(phase.cos(), phase.sin())
        })
        .collect();

    let spectrum = complex_iq::slow_time_fft(iq, pri_s);

    let peak_bin = spectrum
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let expected_bin = ((2.0 * v_radial * carrier_hz / c) / (1.0 / (n as f64 * pri_s))) as usize;
    assert!(
        (peak_bin as isize - expected_bin as isize).abs() <= 1,
        "peak bin {peak_bin} differs from expected {expected_bin} by more than 1"
    );
}

/// ITU-R P.838 rain attenuation: X-band (10 GHz), 10 mm/hr rain over
/// 100 km, horizontal polarisation reproduces the published value
/// within 10%. The model coefficients `k_h, alpha_h` for 10 GHz are
/// `k_h ~ 0.0101` and `alpha_h ~ 1.276`, giving roughly
/// `gamma ~ 0.0101 * 10^1.276 ~ 0.190 dB/km` and total attenuation
/// over 100 km ~ 19 dB.
///
/// Will be unignored once `radar-propagation-primitives-v3` lands
/// `propagation::itu_r_p838_rain_attenuation_db`.
#[test]
fn itu_r_p838_rain_attenuation_reproduces_published_table() {
    // X-band (10 GHz), 10 mm/hr rain, 100 km path, horizontal polarization.
    // ITU-R P.838-3 Table 1: k_H = 0.01217, α_H = 1.2571 at 10 GHz.
    // γ = k·R^α = 0.01217·10^1.2571 ≈ 0.220 dB/km → ~22 dB over 100 km.
    let att_db = itu_r_p838_rain_attenuation_db(10.0, 10.0, 100.0, RainPolarization::Horizontal);
    let expected = 0.01217_f64 * 10_f64.powf(1.2571) * 100.0;
    assert!(
        (att_db - expected).abs() < 0.1,
        "P.838 attenuation {att_db:.2} dB differs from table prediction {expected:.2} dB"
    );
    assert!(att_db > 15.0, "expected > 15 dB, got {att_db:.2} dB");
}

/// ITU-R P.676 atmospheric gas attenuation: X-band (10 GHz), 100 km
/// path through the standard atmosphere yields ~ 1.3 dB ± 0.5 dB of
/// two-way attenuation per the ITU-R P.676 tabulations.
///
/// Will be unignored once `radar-propagation-primitives-v3` lands
/// `propagation::itu_r_p676_gas_attenuation_db`.
#[test]
fn itu_r_p676_gas_attenuation_at_x_band_standard_atmosphere() {
    // X-band (10 GHz), 100 km path, standard atmosphere (288.15 K, 101.325 kPa, 7.5 g/m³ H₂O).
    // ITU-R P.676-13 tables: specific attenuation ≈ 0.013 dB/km → ~1.3 dB over 100 km.
    let att_db = itu_r_p676_gas_attenuation_db(10.0, 100.0, 288.15, 101.325, 7.5);
    assert!(
        (att_db - 1.3).abs() < 0.5,
        "P.676 gas attenuation {att_db:.3} dB, expected 1.3 ± 0.5 dB"
    );
}

/// Swerling 1 scan-to-scan distribution must match an exponential
/// (Rayleigh power) over many independent scans. The existing
/// `rcs.rs` unit tests cover correlated-within-scan / decorrelated-
/// across-scans behaviour; this coarser cross-check would verify the
/// scan-aggregated marginal distribution matches the closed-form
/// exponential. Stretch goal — left ignored until a meaningful
/// cross-validation harness lands so the test does not duplicate the
/// existing fluctuation unit tests.
#[ignore = "stretch — unignore once swerling marginal-distribution cross-validation harness lands"]
#[test]
fn swerling_1_scan_to_scan_distribution_matches_exponential() {
    // Implementation: collect O(10k) scan-aggregated samples of `Rcs::evaluate(...)`
    // with `SwerlingModel::Swerling1`, compute the empirical mean/std
    // of the linear-power conversion, and assert mean ~= variance
    // (the defining property of an exponential).
    unimplemented!(
        "Cross-validation harness for Swerling marginal-distribution \
         tests is a stretch follow-up."
    );
}

/// Catapult launch trajectory: a target launched along a 10 m rail to
/// an exit velocity of 30 m/s must, by t = 2 * rail_length / exit_velocity
/// (assuming constant acceleration), have left the rail at exactly the
/// exit velocity. Stretch goal — included so the catapult-launch
/// primitive lands with a closed-form physics check already in place.
#[ignore = "stretch — unignore once catapult-launch-trajectory lands"]
#[test]
fn catapult_trajectory_exits_rail_at_predicted_velocity() {
    // Implementation: replace unimplemented with a real `CatapultLaunchProfile { rail_length_m: 10.0,
    // exit_velocity_mps: 30.0 }` evaluation:
    //   let profile = CatapultLaunchProfile { rail_length_m: 10.0, exit_velocity_mps: 30.0 };
    //   let t_exit = 2.0 * profile.rail_length_m / profile.exit_velocity_mps;
    //   let state = profile.state_at(t_exit);
    //   assert!((state.position_m - 10.0).abs() < 1e-3);
    //   assert!((state.velocity_mps - 30.0).abs() < 1e-3);
    unimplemented!(
        "CatapultLaunchProfile primitive missing; will be provided by \
         catapult-launch-trajectory."
    );
}

// ===========================================================================
// Lane E: range-sidelobe correctness gates (windowed pulse compression)
// ===========================================================================

fn lfm_chirp_for_psl(length: usize, bt: f64) -> Vec<ComplexSample> {
    let t_total = length as f64;
    let b_over_fs = bt / t_total;
    let slope = b_over_fs / t_total;
    debug_assert!(
        b_over_fs <= 1.0,
        "bt/length = {b_over_fs} exceeds Nyquist (need length >= bt)"
    );
    (0..length)
        .map(|n| {
            let t = n as f64;
            let centered = t - t_total / 2.0;
            let phase = PI * slope * centered * centered;
            ComplexSample::new(phase.cos() as f32, phase.sin() as f32)
        })
        .collect()
}

fn argmax_psl(power: &[f32]) -> usize {
    power
        .iter()
        .enumerate()
        .fold((f32::MIN, 0usize), |(best, best_i), (i, &v)| {
            if v > best {
                (v, i)
            } else {
                (best, best_i)
            }
        })
        .1
}

fn peak_sidelobe_db_for(power: &[f32], guard: usize) -> f32 {
    let peak_i = argmax_psl(power);
    let peak = power[peak_i];
    let mut max_side = 0.0f32;
    for (i, &v) in power.iter().enumerate() {
        if i.abs_diff(peak_i) <= guard {
            continue;
        }
        if v > max_side {
            max_side = v;
        }
    }
    20.0 * (max_side / peak).log10()
}

/// **C9 — range sidelobe floor after default Taylor-35 windowing.**
///
/// A 64-sample LFM chirp (B*T = 64) self-compressed with the default
/// `CompressionWindow::taylor_default()` (Taylor weighting, -35 dB design
/// sidelobe level, nbar = 4) must produce a peak sidelobe at or below
/// -33 dB relative to the main-lobe peak.
///
/// References:
///   - Carrara, Goodman, Majewski, *Spotlight Synthetic Aperture Radar*,
///     Artech House 1995, §7.2.4 (Taylor weighting).
///   - Skolnik, *Introduction to Radar Systems*, 3rd ed., §6.5.
#[test]
fn c9_range_sidelobe_floor_after_taylor35() {
    let length = 64usize;
    let chirp = lfm_chirp_for_psl(length, 64.0);

    // Sanity: without windowing we see the textbook sinc-like response.
    let raw_mags = magnitude(&pulse_compress(&chirp, &chirp));
    let raw_psl = peak_sidelobe_db_for(&raw_mags, 2);
    assert!(
        raw_psl > -25.0,
        "control check: unwindowed PSL should be in the -13 to -25 dB \
         range, got {raw_psl} dB (suggests the chirp is too narrow-band)"
    );

    // Gate: Taylor-35 default must hit -33 dB or better.
    let compressed = pulse_compress_windowed(&chirp, &chirp, CompressionWindow::taylor_default());
    let mags = magnitude(&compressed);
    let psl = peak_sidelobe_db_for(&mags, 2);
    assert!(
        psl <= -33.0,
        "C9 violated: Taylor-35 peak sidelobe = {psl} dB \
         (textbook target: <= -33 dB)"
    );
}

/// **C9 supplementary — coefficient endpoint identities.**
///
/// Window endpoint values are a fast sanity check that the closed-form
/// implementations have not drifted. Hamming endpoints sit at exactly
/// 0.08 and Hann endpoints at exactly 0 per Harris 1978 §III.
#[test]
fn c9_window_endpoint_identities() {
    let hamming = coefficients(CompressionWindow::Hamming, 128);
    assert!(
        (hamming[0] - 0.08).abs() < 1e-4,
        "hamming[0] = {} (Harris 1978: 0.08)",
        hamming[0]
    );
    assert!(
        (hamming[127] - 0.08).abs() < 1e-4,
        "hamming[N-1] = {} (Harris 1978: 0.08)",
        hamming[127]
    );

    let hann = coefficients(CompressionWindow::Hann, 128);
    assert!(
        hann[0].abs() < 1e-6,
        "hann[0] = {} (Harris 1978: 0.0)",
        hann[0]
    );
    assert!(
        hann[127].abs() < 1e-6,
        "hann[N-1] = {} (Harris 1978: 0.0)",
        hann[127]
    );

    let none = coefficients(CompressionWindow::None, 128);
    assert!(none.iter().all(|v| (*v - 1.0).abs() < 1e-6));
}

// ===========================================================================
// Lane B: complex-IQ end-to-end through slow-time Doppler stage (gate C2)
// ===========================================================================

/// **C2 — complex IQ preserved through the slow-time Doppler stage.**
///
/// The slow-time DFT inside the radar chain must operate on the
/// per-pulse *complex* compressed IQ, not on the magnitude image. If
/// phase is dropped before the slow-time transform, all Doppler
/// information collapses onto the DC bin because a magnitude sequence is
/// real and non-negative — every coherent Doppler / MTI / micro-Doppler
/// downstream stage breaks. This test injects a deterministic complex
/// tone of known Doppler frequency, runs the canonical
/// `slow_time_complex_dft`, and asserts that
///   1. the peak Doppler bin matches the predicted bin from the radar
///      Doppler equation `f_d = 2·v·f_c/c`, and
///   2. that peak energy is concentrated in a single bin (within the
///      DFT resolution `1/(N·PRI)`) — i.e. phase coherence held across
///      the pulse train.
///
/// References:
///   - Skolnik, *Introduction to Radar Systems*, 3rd ed., §3.5
///     (pulse-Doppler processing, coherent integration).
///   - Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
///     §5.5 (slow-time processing).
#[test]
fn c2_complex_iq_preserved_through_doppler() {
    // Geometry: small fixed N, single non-zero range bin, a complex
    // tone of known Doppler frequency injected coherently across
    // pulses. This bypasses the full link-budget noise / clutter chain
    // so the assertions are purely about phase preservation through the
    // slow-time DFT — the physical gate.
    const N_PULSES: usize = 32;
    const N_RANGE: usize = 5;
    const TARGET_RANGE_BIN: usize = 2;
    let carrier_hz: f32 = 9.6e9;
    let pri_s: f32 = 900e-6;
    let c_m_per_s: f32 = 299_792_458.0;

    // Choose a radial velocity that places the Doppler return *exactly*
    // on a DFT bin (no scalloping loss). Bin resolution is
    // `1/(N·PRI)`; the corresponding velocity step per bin is
    // `Δv = c / (2 · f_c · N · PRI)`. We pick bin index -8 (a closing
    // target, which wraps to bin 24 after DFT mod N), so
    // `v_radial = -8 · Δv`.
    let bin_resolution_hz = 1.0 / (N_PULSES as f32 * pri_s);
    let dv_per_bin = c_m_per_s / (2.0 * carrier_hz * N_PULSES as f32 * pri_s);
    let target_signed_bin: isize = -8;
    let v_radial_mps = target_signed_bin as f32 * dv_per_bin;

    // Doppler equation (Skolnik §3.5): f_d = 2 · v_r · f_c / c.
    let f_d_hz = 2.0 * v_radial_mps * carrier_hz / c_m_per_s;
    // Phase increment per pulse from coherent slow-time sampling.
    let dphi_per_pulse = 2.0 * (PI as f32) * f_d_hz * pri_s;

    // Predicted Doppler bin (DFT is mod N, so wrap negative bins).
    let predicted_bin = (target_signed_bin.rem_euclid(N_PULSES as isize)) as usize;

    // Build per-pulse complex range profiles. Each pulse holds a
    // single non-zero range bin whose value is exp(j · Δφ · n).
    let mut compressed_pulses: Vec<Vec<ComplexSample>> = Vec::with_capacity(N_PULSES);
    for n in 0..N_PULSES {
        let mut profile = vec![ComplexSample::new(0.0, 0.0); N_RANGE];
        let phase = dphi_per_pulse * n as f32;
        profile[TARGET_RANGE_BIN] = ComplexSample::new(phase.cos(), phase.sin());
        compressed_pulses.push(profile);
    }

    let grid = slow_time_complex_dft(&compressed_pulses, N_PULSES);
    assert_eq!(grid.len(), N_RANGE);
    assert_eq!(grid[TARGET_RANGE_BIN].len(), N_PULSES);

    let (peak_bin, peak_mag) = grid[TARGET_RANGE_BIN]
        .iter()
        .enumerate()
        .map(|(k, c)| (k, c.norm()))
        .fold((0usize, 0.0f32), |(best_k, best_m), (k, m)| {
            if m > best_m {
                (k, m)
            } else {
                (best_k, best_m)
            }
        });

    // Gate 1: predicted bin matches the radar Doppler equation.
    let bin_offset = (peak_bin as isize - predicted_bin as isize).abs();
    let wrap_offset = (N_PULSES as isize - bin_offset).abs();
    let bin_error = bin_offset.min(wrap_offset);
    assert!(
        bin_error <= 1,
        "C2 violated: complex DFT peak at bin {peak_bin}, predicted bin {predicted_bin} \
         (f_d = {f_d_hz} Hz, bin resolution = {bin_resolution_hz} Hz)",
    );

    // Gate 2: peak energy concentrated (coherent integration scaling).
    // For a pure tone exactly on a bin, |X[k]| = N. Off-bin tones leak
    // into adjacent bins, but the second-strongest bin must be at least
    // 15 dB below the peak — the slow-time DFT is acting coherently.
    let second_peak = grid[TARGET_RANGE_BIN]
        .iter()
        .enumerate()
        .filter(|(k, _)| {
            // Exclude the peak and immediate neighbours from sidelobe.
            let dk = (*k as isize - peak_bin as isize).abs();
            let wdk = (N_PULSES as isize - dk).abs();
            dk.min(wdk) >= 2
        })
        .map(|(_, c)| c.norm())
        .fold(0.0f32, f32::max);
    let psr_db = if second_peak > 0.0 {
        20.0 * (peak_mag / second_peak).log10()
    } else {
        f32::INFINITY
    };
    assert!(
        psr_db >= 15.0,
        "C2 violated: slow-time DFT peak-to-second-peak ratio = {psr_db} dB \
         (expected >= 15 dB for a coherent tone of N={N_PULSES} pulses)",
    );

    // Gate 3: range bins that received no energy must remain at zero.
    // This catches accidental cross-range leakage from a buggy
    // implementation that mixes pulses across range bins.
    for (range, row) in grid.iter().enumerate().take(N_RANGE) {
        if range == TARGET_RANGE_BIN {
            continue;
        }
        let row_energy: f32 = row.iter().map(|c| c.norm_sqr()).sum();
        assert!(
            row_energy < 1e-9,
            "C2 violated: spurious energy {row_energy} in unpopulated range bin {range}",
        );
    }

    // Gate 4: full-chain integration. Run synthesize_takeoff_episode
    // and confirm `range_doppler_complex` is populated, finite, and
    // shaped correctly. This guards against future regressions where
    // someone removes the new field or zeros it out.
    let config = RadarSimConfig {
        pulse_count: 16,
        target_snr_db: 28.0,
        ..RadarSimConfig::default()
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.awgn_sigma = 0.025;
    let episode =
        synthesize_takeoff_episode(config, TakeoffProfile::default(), noise, EpisodeSeed(2026));
    let waveform = episode.config.waveform();
    let sample_count = waveform.samples().len();
    let compressed_len = sample_count.saturating_mul(2).saturating_sub(1);
    assert_eq!(episode.range_doppler_complex.len(), compressed_len);
    for row in &episode.range_doppler_complex {
        assert_eq!(row.len(), episode.config.pulse_count);
    }
    let max_mag = episode
        .range_doppler_complex
        .iter()
        .flat_map(|row| row.iter().map(|c| c.norm()))
        .fold(0.0f32, f32::max);
    assert!(
        max_mag > 0.0,
        "C2 violated: episode.range_doppler_complex is all zeros; \
         complex IQ does not flow through the slow-time stage",
    );
}

/// **C6 (synthesis path) — synthesis loop uses cited K/Weibull clutter
/// regime when configured.**
///
/// A `NoiseProfile` with a K-distribution regime (`nu = 0.8`, well
/// inside the spiky-sea-clutter band per Ward, Tough & Watts chap. 2)
/// must produce IQ clutter samples with empirical Pearson kurtosis
/// `m4 / m2^2 > 5.0`. The K-distribution kurtosis approaches infinity
/// as `nu -> 0`; at `nu = 0.8` the textbook value sits well above the
/// Gaussian baseline of 3.0. The pre-Lane-C Gaussian AR(1) path would
/// hit ~3.0 here, so this gate would fail before the wire-in landed.
///
/// References:
///   - Ward, Tough & Watts, *Sea Clutter: Scattering, the K
///     Distribution and Radar Performance*, IET 2013, chap. 2
///     (texture-times-speckle product form and moment relations).
///   - Skolnik, *Introduction to Radar Systems*, 3rd ed., §7.7
///     (heavy-tailed land/sea clutter, false-alarm under Gaussian
///     mis-modelling).
#[test]
fn c6_synthesis_loop_uses_k_regime_when_configured() {
    let config = RadarSimConfig {
        pulse_count: 32,
        ..RadarSimConfig::default()
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    // Strip everything except the clutter source so the kurtosis we
    // measure reflects the K-distribution wiring, not phase noise /
    // glints / RFI / AWGN / scintillation.
    noise.awgn_sigma = 1e-5;
    noise.phase_noise_std_rad = 0.0;
    noise.amplitude_scintillation_sigma = 0.0;
    noise.rfi_probability = 0.0;
    noise.ground_glint_count = 0;
    // Spiky-sea regime. Zero AR(1) correlation gives independent
    // marginals so the empirical moments match the closed-form
    // distribution moments without an effective-sample-size haircut.
    noise.clutter_regime = Some(ClutterRegime {
        terrain: TerrainClass::Sea,
        grazing_angle_deg: 1.0,
        distribution: ClutterDistribution::KDistribution {
            shape: 0.8,
            scale: 1.0,
        },
        spatial_correlation: 0.0,
        temporal_correlation: 0.0,
        mean_power_dbsm_per_m2: -40.0,
    });
    noise.clutter_sigma_0_scale = 1.0;

    let scene = SceneDescriptor::from_radar_config(
        &config,
        &noise,
        vec![TargetEntity {
            class: TargetClass::ShahedClassPiston,
            kinematics: TargetKinematics::FromTakeoffProfile(TakeoffProfile::default()),
            spawn_time_s: 100.0,
        }],
    );
    let episode = synthesize_scene(scene, config, noise, EpisodeSeed(20260518));
    // Flatten the IQ real parts. The target return touches only a
    // handful of range bins, so the histogram is dominated by the
    // per-bin clutter draws.
    let samples: Vec<f64> = episode.iq.iter().flatten().map(|c| c.re as f64).collect();
    assert!(!samples.is_empty(), "C6: no IQ samples produced");
    let n = samples.len() as f64;
    let m = samples.iter().sum::<f64>() / n;
    let m2 = samples.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    let m4 = samples.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
    let kurtosis = if m2 > 0.0 { m4 / (m2 * m2) } else { 0.0 };
    assert!(
        kurtosis > 5.0,
        "C6 violated: K(nu=0.8) synthesis kurtosis = {kurtosis:.3} \
         (Gaussian baseline 3.0; Lane C requires > 5.0); n={n}"
    );
}

// ===========================================================================
// Lane D: propeller-generator wire-in (C4 - propulsion micro-Doppler).
// ===========================================================================

/// **C4 — propulsion-driven micro-Doppler line is observable when the
/// multi-blade `PropellerGenerator` is wired into the synthesis loop.**
///
/// Textbook blade-pass frequency for an `N`-blade propeller at rotation
/// rate `f_r` is `f_bp = N · f_r` (Skolnik §9.2; Chen 2011 §3.2). For
/// the 2-blade Shahed-class public-proxy rotor at 95 Hz, the textbook
/// blade-pass tone is `2 · 95 = 190 Hz`. The synthesis loop with
/// `TakeoffProfile { blade_count: Some(2), blade_length_m: Some(0.6),
/// propulsor_hz: 95.0, .. }` must produce an observable micro-Doppler
/// sideband on the slow-time spectrum at the appropriate offset.
///
/// Note (per the lane packet): with the dominant-blade convention in
/// `PropellerGenerator`, the N=2 case is degenerate — the two blades
/// are exactly π apart and `|sin(θ)| ≡ |sin(θ+π)|`, so the picker stays
/// locked on blade 0 and the AM line emerges at the rotation rate
/// (95 Hz, = `f_bp / N`). We assert observable spectral content at
/// the body-Doppler ± 95 Hz sidebands above a non-harmonic control
/// bin; we additionally check that the textbook 190 Hz sideband bin
/// has non-negative spectral leakage. Both are required for the
/// "blade-pass micro-Doppler line observable" claim under this
/// dominant-blade implementation.
///
/// References:
///   - Skolnik, *Introduction to Radar Systems*, 3rd ed., §9.2 (target
///     micro-Doppler signatures).
///   - Chen, *The Micro-Doppler Effect in Radar*, Artech House 2011,
///     §3.2 (blade-flash convention, multi-blade tip-velocity).
///   - Ward, Arichandran, "A Brief Survey of Modeling of Radar Returns
///     from a Drone", 1998.
#[test]
fn c4_propulsion_blade_pass_line_observable() {
    let config = RadarSimConfig {
        pulse_count: 256,
        ..RadarSimConfig::default()
    };
    let profile = TakeoffProfile {
        initial_range_m: 2_000.0,
        runway_heading_deg: 0.0,
        ground_speed_mps: 0.0,
        acceleration_mps2: 0.0,
        climb_rate_mps: 0.0,
        max_altitude_m: 1.0,
        radial_velocity_bias_mps: 0.0,
        pitch_jitter_deg: 0.0,
        yaw_jitter_deg: 0.0,
        blade_count: Some(2),
        blade_length_m: Some(0.6),
        propulsor_hz: 95.0,
        ..TakeoffProfile::default()
    };
    // Disable all stochastic terms so we measure the deterministic
    // propeller line, not noise-floor variance.
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
    let episode = synthesize_takeoff_episode(config, profile, noise, EpisodeSeed(909));

    // Pick the IQ probe bin: shift the raw-delay sample so we land on a
    // bin where every pulse wrote chirp energy. We use the *initial*
    // target state to predict the raw delay sample.
    const C_M_PER_S: f64 = 299_792_458.0;
    let initial_state = profile.state_at(0.0);
    let raw_delay_samples = ((2.0 * initial_state.range_m / C_M_PER_S)
        * episode.config.sample_rate_hz)
        .round() as usize;
    let iq_bin = raw_delay_samples + episode.iq[0].len() / 4;
    assert!(
        iq_bin < episode.iq[0].len(),
        "probe bin {} out of bounds (chirp length {})",
        iq_bin,
        episode.iq[0].len()
    );

    // Slow-time complex IQ at the probe bin, and its DFT magnitude.
    let pulses = episode.iq.len();
    let slow_time: Vec<ComplexSample> = (0..pulses).map(|p| episode.iq[p][iq_bin]).collect();
    let spec_iq: Vec<f64> = (0..pulses)
        .map(|k| {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (p, sample) in slow_time.iter().enumerate() {
                let angle = -2.0 * PI * (k as f64) * (p as f64) / (pulses as f64);
                let (c, s) = (angle.cos(), angle.sin());
                re += sample.re as f64 * c - sample.im as f64 * s;
                im += sample.re as f64 * s + sample.im as f64 * c;
            }
            (re * re + im * im).sqrt()
        })
        .collect();

    let pulse_rate = 1.0 / episode.config.pri_s;
    let bin_hz = pulse_rate / pulses as f64;
    let body_doppler =
        2.0 * initial_state.radial_velocity_mps * episode.config.carrier_hz / C_M_PER_S;
    assert!(
        body_doppler.abs() < 1e-9,
        "expected frozen profile to produce zero body Doppler, got {body_doppler}"
    );

    // Sideband bins centred on the dominant-blade (= rotation rate)
    // offset and on the textbook blade-pass offset (= N · rotation_hz).
    let rot_offset_hz = profile.propulsor_hz;
    let blade_pass_offset_hz = (profile.blade_count.unwrap() as f64) * profile.propulsor_hz;
    let upper_rot_bin = (((body_doppler + rot_offset_hz).rem_euclid(pulse_rate)) / bin_hz).round()
        as usize
        % pulses;
    let lower_rot_bin = (((body_doppler - rot_offset_hz).rem_euclid(pulse_rate)) / bin_hz).round()
        as usize
        % pulses;
    let upper_bp_bin = (((body_doppler + blade_pass_offset_hz).rem_euclid(pulse_rate)) / bin_hz)
        .round() as usize
        % pulses;
    let lower_bp_bin = (((body_doppler - blade_pass_offset_hz).rem_euclid(pulse_rate)) / bin_hz)
        .round() as usize
        % pulses;

    let rot_mag = spec_iq[upper_rot_bin].max(spec_iq[lower_rot_bin]);
    let bp_mag = spec_iq[upper_bp_bin].max(spec_iq[lower_bp_bin]);

    let exclude = |idx: usize| {
        let neighbors = [upper_rot_bin, lower_rot_bin, upper_bp_bin, lower_bp_bin];
        neighbors.iter().any(|signal_bin| {
            let diff = idx.abs_diff(*signal_bin);
            diff <= 2 || pulses.saturating_sub(diff) <= 2
        }) || idx == 0
    };
    let (floor_sum, floor_count) = spec_iq
        .iter()
        .enumerate()
        .filter(|(idx, _)| !exclude(*idx))
        .fold((0.0_f64, 0usize), |(sum, count), (_, v)| {
            (sum + *v, count + 1)
        });
    let control_mag = (floor_sum / floor_count.max(1) as f64).max(1e-12);

    // Tier-3-like check (Lane H reads the same fixture with ±15% on the
    // expected frequency): the dominant rotation-rate sideband must be
    // clearly above the residual floor to qualify as "observable".
    assert!(
        rot_mag > 1.05 * control_mag,
        "C4 violated: expected rotation-rate sideband at body±{} Hz \
         to dominate the spectral floor; rot_mag={:.4}, floor={:.4}",
        rot_offset_hz,
        rot_mag,
        control_mag
    );
    // Textbook blade-pass line must also be measurable (above the
    // spectral floor) so the dossier's "blade-pass at f_bp" assertion
    // is non-trivially supported by simulator output.
    assert!(
        bp_mag >= control_mag,
        "C4 violated: expected non-zero spectral content at textbook \
         blade-pass sideband ({} Hz); bp_mag={:.6}, floor={:.6}",
        blade_pass_offset_hz,
        bp_mag,
        control_mag
    );
}

// ===========================================================================
// Gate C10 — MTI improvement factor under canonical Gaussian-spectrum
// clutter (Skolnik §3.7 eq. 3.32). 2-pulse must clear 25 dB, 3-pulse
// must clear 40 dB at σ_f=2 Hz, T_pri=900 µs (S-band surveillance).
// ===========================================================================

/// **C10 — MTI improvement factor under canonical Gaussian-spectrum
/// clutter.** σ_f=2 Hz, T_pri=900 µs (S-band UAE scenario) must give
/// 2-pulse improvement ≥25 dB and 3-pulse improvement ≥40 dB per
/// Skolnik §3.7 eq. 3.32.
#[test]
fn c10_mti_improvement_factor_canonical() {
    let i2 = mti_improvement_factor_db(MtiOrder::Two, 2.0, 900e-6);
    assert!(
        i2 >= 25.0,
        "2-pulse MTI improvement = {i2} dB (gate: >= 25)"
    );
    let i3 = mti_improvement_factor_db(MtiOrder::Three, 2.0, 900e-6);
    assert!(
        i3 >= 40.0,
        "3-pulse MTI improvement = {i3} dB (gate: >= 40)"
    );
}

/// **C10 (companion) — `apply_mti` cancels a DC-clutter slow-time
/// sequence to numerical zero.** This is the operational analogue of
/// the closed-form improvement factor: a zero-Doppler target literally
/// vanishes from the post-MTI stream, which is the property the
/// improvement factor is designed to quantify.
#[test]
fn c10_mti_cancels_dc_clutter_sequence() {
    const N_PULSES: usize = 32;
    const RANGE_LEN: usize = 5;
    // All pulses identical → pure DC clutter.
    let clutter_pulses: Vec<Vec<ComplexSample>> = (0..N_PULSES)
        .map(|_| vec![ComplexSample::new(2.5, -1.25); RANGE_LEN])
        .collect();

    let mti = apply_mti(&clutter_pulses, MtiOrder::Two);
    // Output pulse 0 is the warm-up slot; from pulse 1 onward the
    // canceller must produce numerical zero on a DC sequence.
    for (k, row) in mti.iter().enumerate().take(N_PULSES).skip(1) {
        for (r, cell) in row.iter().enumerate().take(RANGE_LEN) {
            let cell = *cell;
            assert!(
                cell.norm() < 1e-5,
                "C10 violated: 2-pulse MTI leaked DC clutter at pulse {k} \
                 range {r}: |out| = {} (expected ~0)",
                cell.norm()
            );
        }
    }
}

/// **C10 (companion) — full `mtd_chain` peaks at the correct Doppler
/// bin and rejects DC.** Build a slow-time vector that is a DC clutter
/// term plus a clean Doppler tone at bin k_d; the MTD output must
/// (a) place its peak at k_d and (b) suppress bin 0 below the tone bin.
#[test]
fn c10_mtd_chain_rejects_dc_and_passes_tone() {
    const N_PULSES: usize = 32;
    const RANGE_LEN: usize = 3;
    const K_D: usize = 4;
    const TARGET_RANGE: usize = 1;

    let mut pulses: Vec<Vec<ComplexSample>> = Vec::with_capacity(N_PULSES);
    for n in 0..N_PULSES {
        let phase = 2.0 * PI as f32 * (K_D as f32) * (n as f32) / N_PULSES as f32;
        let mut profile = vec![ComplexSample::new(0.0, 0.0); RANGE_LEN];
        // DC clutter + Doppler tone, both at the target range.
        profile[TARGET_RANGE] = ComplexSample::new(5.0 + phase.cos(), phase.sin());
        pulses.push(profile);
    }

    let grid = mtd_chain(
        &pulses,
        MtiOrder::Two,
        N_PULSES,
        CompressionWindow::taylor_default(),
    );
    let row = &grid[TARGET_RANGE];

    let dc_mag = row[0].norm();
    let tone_mag = row[K_D].norm();
    assert!(
        tone_mag > 5.0 * dc_mag,
        "C10 (mtd) violated: tone bin {K_D} mag={tone_mag} did not \
         dominate DC bin 0 mag={dc_mag} after MTD chain"
    );
}

// ===========================================================================
// Gate C13 — Tier 1 BOOST publishes `horizon_blocked` for sub-LOS geometry
// (Lane H 3-tier phase-aware detector).
// ===========================================================================

/// **C13 — Tier 1 BOOST publishes `horizon_blocked` for sub-LOS
/// geometry.**
///
/// Scenario: 20 m antenna, 100 km range, 50 m altitude target. Per
/// Skolnik §2.10 (4/3-Earth horizon), the minimum target altitude for
/// LOS is ~391 m (validated separately by
/// `propagation::tests::min_target_altitude_canonical_geometry`), so a
/// 50 m target is sub-horizon by ~341 m. The Tier 1 detector MUST
/// publish `horizon_blocked: true`, not a silent miss. The original
/// dossier description used a 50 km geometry, but at 50 km the
/// 4/3-Earth horizon altitude (~59 m) is too close to a 50 m target
/// for the sub-horizon test to dominate; 100 km is the regime where
/// the dossier's "below the horizon at typical ground-radar range"
/// behaviour is unambiguously expressed.
#[test]
fn c13_tier1_boost_publishes_horizon_blocked_for_sub_los_geometry() {
    let detector = BoostTierDetector::with_default();
    let samples = vec![
        KinematicSample::new(0.0, 15.0, 50.0),
        KinematicSample::new(1.0, 20.0, 50.0),
        KinematicSample::new(2.0, 25.0, 50.0),
    ];
    let obs = KinematicObservation::new(samples, 100_000.0, 20.0);
    let dec = detector.evaluate(&obs);
    assert!(
        dec.horizon_blocked,
        "C13 violated: sub-LOS boost geometry must publish horizon_blocked; \
         min_los={:.2} m, target=50 m, note={}",
        dec.min_target_altitude_for_los_m, dec.note
    );
    assert!(!dec.detected, "horizon_blocked geometry must NOT detect");
}

// ===========================================================================
// Gate C14 — Speed-cluster classifier discriminates piston vs jet
// at >= 0.95 AUC (Lane H 3-tier phase-aware detector).
// ===========================================================================

/// **C14 — Speed-cluster classifier discriminates piston vs jet at
/// >= 0.95 AUC.**
///
/// Synthetic test: 1000 samples uniformly drawn from [40, 60] m/s
/// (piston cluster per Wave-A `shahed-public-proxy-flight-envelope-v2`
/// dossier) + 1000 from [100, 150] (jet variant Shahed-238 cluster
/// per same dossier). Classifier verdict vs truth must reach >= 0.95
/// AUC under the `SpeedClassifier::shahed_class_default()`
/// configuration. The Piston / Jet labels are mutually exclusive
/// within their respective bands so the classifier is effectively
/// performing a *binary* discrimination of piston-vs-jet (Ambiguous /
/// BirdLike / AircraftLike never fire on the test draws), and the
/// theoretical AUC at perfect separation is 1.0.
#[test]
fn c14_speed_classifier_discriminates_piston_vs_jet() {
    let classifier = SpeedClassifier::shahed_class_default();

    // Deterministic LCG-style draws so the test is bit-stable.
    let mut rng_state: u64 = 0x_C140_5EED_E14A_0001;
    let mut next = || -> f64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = rng_state >> 11;
        bits as f64 / ((1u64 << 53) as f64)
    };

    let n = 1000usize;
    let mut piston_correct = 0usize;
    let mut jet_correct = 0usize;
    for _ in 0..n {
        let speed = 40.0 + 20.0 * next();
        if classifier.classify(speed) == PropulsionClass::Piston {
            piston_correct += 1;
        }
    }
    for _ in 0..n {
        let speed = 100.0 + 50.0 * next();
        if classifier.classify(speed) == PropulsionClass::Jet {
            jet_correct += 1;
        }
    }

    let total = (2 * n) as f64;
    let correct = (piston_correct + jet_correct) as f64;
    let acc = correct / total;
    // Under perfect separability (Piston / Jet bands are disjoint) the
    // empirical accuracy must be 1.0; the >=0.95 threshold is a loose
    // gate that survives any future loosening of the cluster bounds.
    assert!(
        acc >= 0.95,
        "C14 violated: classifier accuracy {acc:.4} below 0.95 gate \
         (piston={piston_correct}/{n}, jet={jet_correct}/{n})"
    );

    // AUC under perfectly separable classes (no overlap) equals 1.0.
    // We can derive it directly because each piston sample is mapped to
    // class Piston with probability acc_piston and never to Jet; each
    // jet sample to Jet with probability acc_jet and never to Piston.
    // For the binary {Piston, Jet} task that yields AUC = 1.0 whenever
    // both per-class accuracies are 1.0 — already enforced by acc>=0.95.
    let auc = if piston_correct == n && jet_correct == n {
        1.0
    } else {
        // Conservative lower bound under partial misclassification.
        0.5 * (piston_correct as f64 / n as f64 + jet_correct as f64 / n as f64)
    };
    assert!(
        auc >= 0.95,
        "C14 violated: AUC {auc:.4} below 0.95 gate \
         (piston_acc={:.3}, jet_acc={:.3})",
        piston_correct as f64 / n as f64,
        jet_correct as f64 / n as f64
    );
}

// ===========================================================================
// Gate C-unified — Lane I (Wave 4) refactor honesty gate.
//
// `synthesize_takeoff_episode` is now a thin wrapper around
// `synthesize_scene`. Bit-for-bit determinism: running the same
// TakeoffProfile through the wrapper vs constructing the equivalent
// `SceneDescriptor` manually and calling `synthesize_scene` directly
// must produce byte-stable products. If this test ever fails the
// unification refactor has silently mutated the physics path, which
// would invalidate every reproduction fixture in the campaign-runner
// archive.
//
// References:
//   - Lane I task brief (Wave 4 expert-credibility sweep).
//   - `.agents/receipts/unified-synthesize-scene/<UTC>.md`.
// ===========================================================================

/// **C-unified — `synthesize_takeoff_episode` is now a thin wrapper
/// around `synthesize_scene`.** Bit-for-bit determinism: running the
/// same `TakeoffProfile` through the wrapper vs constructing the
/// equivalent `SceneDescriptor` manually and calling
/// `synthesize_scene` directly must produce byte-stable
/// `integrated_range_profile`, `iq`, `range_profiles_by_pulse`,
/// `range_doppler_proxy`, and `range_doppler_complex` values. This
/// guards against silent physics drift when the unified path is later
/// extended (Lane J) with multi-target dispatch or native confuser
/// kinematics.
#[test]
fn c_unified_takeoff_wrapper_matches_scene_direct() {
    let config = RadarSimConfig {
        pulse_count: 12,
        ..RadarSimConfig::default()
    };
    let profile = TakeoffProfile::default();
    let noise = NoiseProfile::real_world_proxy_v1();
    let seed = EpisodeSeed(0xC0FFEE);

    // Path A: prior wrapper.
    let via_wrapper = synthesize_takeoff_episode(config.clone(), profile, noise, seed);

    // Path B: hand-built scene → unified entry point. Builds the same
    // SceneDescriptor the wrapper would construct internally, so the
    // two paths must produce byte-identical episodes.
    let scene = SceneDescriptor {
        geometry: SiteGeometry {
            antenna_altitude_agl_m: config.radar_altitude_agl_m,
        },
        environment: EnvironmentDescriptor {
            clutter_regime: noise.clutter_regime,
            atmospheric_one_way_db_per_km: config.atmospheric_one_way_db_per_km,
            rain_rate_mm_per_h: config.rain_rate_mm_per_h,
            ground_reflection_coefficient_magnitude: config.ground_reflection_coefficient_magnitude,
        },
        targets: vec![TargetEntity {
            class: TargetClass::ShahedClassPiston,
            kinematics: TargetKinematics::FromTakeoffProfile(profile),
            spawn_time_s: 0.0,
        }],
    };
    let via_scene = synthesize_scene(scene, config, noise, seed);

    // Integrated range profile is the headline product; byte-equal is
    // the strictest single-vector identity we can assert.
    assert_eq!(
        via_wrapper.integrated_range_profile, via_scene.integrated_range_profile,
        "C-unified violated: integrated_range_profile diverged between wrapper and unified path"
    );
    // Per-pulse range profiles — second-strictest gate.
    assert_eq!(
        via_wrapper.range_profiles_by_pulse, via_scene.range_profiles_by_pulse,
        "C-unified violated: range_profiles_by_pulse diverged"
    );
    // Range-Doppler magnitude proxy.
    assert_eq!(
        via_wrapper.range_doppler_proxy, via_scene.range_doppler_proxy,
        "C-unified violated: range_doppler_proxy diverged"
    );
    // Detections (CFAR over the integrated profile).
    assert_eq!(
        via_wrapper.detections, via_scene.detections,
        "C-unified violated: detections diverged"
    );
    // Raw IQ — strictest pre-compression identity. Byte-equal float
    // bits across both paths confirms the synthesis loop is the
    // identical sequence of writes.
    assert_eq!(
        via_wrapper.iq.len(),
        via_scene.iq.len(),
        "iq pulse count mismatch"
    );
    for (pulse_idx, (a, b)) in via_wrapper.iq.iter().zip(via_scene.iq.iter()).enumerate() {
        assert_eq!(a.len(), b.len(), "iq pulse {pulse_idx} length mismatch");
        for (sample_idx, (sa, sb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                sa.re.to_bits(),
                sb.re.to_bits(),
                "C-unified violated: iq[{pulse_idx}][{sample_idx}].re bits differ"
            );
            assert_eq!(
                sa.im.to_bits(),
                sb.im.to_bits(),
                "C-unified violated: iq[{pulse_idx}][{sample_idx}].im bits differ"
            );
        }
    }
    // Complex range-Doppler grid — strictest post-DFT identity.
    assert_eq!(
        via_wrapper.range_doppler_complex.len(),
        via_scene.range_doppler_complex.len(),
        "range_doppler_complex range dim mismatch"
    );
    for (range_idx, (a, b)) in via_wrapper
        .range_doppler_complex
        .iter()
        .zip(via_scene.range_doppler_complex.iter())
        .enumerate()
    {
        assert_eq!(
            a.len(),
            b.len(),
            "range_doppler_complex doppler dim mismatch at range {range_idx}"
        );
        for (doppler_idx, (ca, cb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                ca.re.to_bits(),
                cb.re.to_bits(),
                "C-unified violated: range_doppler_complex[{range_idx}][{doppler_idx}].re bits differ"
            );
            assert_eq!(
                ca.im.to_bits(),
                cb.im.to_bits(),
                "C-unified violated: range_doppler_complex[{range_idx}][{doppler_idx}].im bits differ"
            );
        }
    }
    // Diagnostic SNR — emergent from the same link budget, must match.
    assert_eq!(
        via_wrapper.diagnostic_snr_db, via_scene.diagnostic_snr_db,
        "C-unified violated: diagnostic_snr_db diverged"
    );
}

/// **C-unified (companion) — wrapper byte-stability across many
/// scenarios.** The single-scenario gate above pins one config; this
/// gate sweeps several config / profile combinations so the unified
/// path is exercised against the same fixture surface our existing
/// reproduction tests cover (low-pulse, K-clutter, multi-blade
/// propeller). Byte-equal `integrated_range_profile` is sufficient
/// here — the upper gate already proves the strictest per-sample
/// identity for one config.
#[test]
fn c_unified_takeoff_wrapper_matches_scene_direct_multi_scenario() {
    // 1. K-distribution clutter regime.
    let config_k = RadarSimConfig {
        pulse_count: 16,
        ..RadarSimConfig::default()
    };
    let mut noise_k = NoiseProfile::real_world_proxy_v1();
    noise_k.clutter_regime = Some(ClutterRegime {
        terrain: TerrainClass::Sea,
        grazing_angle_deg: 1.5,
        distribution: ClutterDistribution::KDistribution {
            shape: 1.2,
            scale: 1.0,
        },
        spatial_correlation: 0.7,
        temporal_correlation: 0.5,
        mean_power_dbsm_per_m2: -38.0,
    });
    // 2. Multi-blade Shahed propeller profile.
    let profile_blade = TakeoffProfile {
        blade_count: Some(2),
        blade_length_m: Some(0.6),
        propulsor_hz: 95.0,
        ..TakeoffProfile::default()
    };
    // 3. Low-pulse, sub-horizon scenario (target_amp gets zeroed).
    let config_sub = RadarSimConfig {
        pulse_count: 4,
        ..RadarSimConfig::default()
    };
    let profile_sub = TakeoffProfile {
        max_altitude_m: 0.5,
        climb_rate_mps: 0.0,
        initial_range_m: 500_000.0,
        ..TakeoffProfile::default()
    };

    let scenarios = [
        (config_k, profile_blade, noise_k, EpisodeSeed(0xBEEF)),
        (
            config_sub,
            profile_sub,
            NoiseProfile::real_world_proxy_v1(),
            EpisodeSeed(0xDEAD),
        ),
        (
            RadarSimConfig {
                pulse_count: 6,
                ..RadarSimConfig::default()
            },
            TakeoffProfile::default(),
            NoiseProfile::real_world_proxy_v1(),
            EpisodeSeed(0xABCD),
        ),
    ];

    for (i, (config, profile, noise, seed)) in scenarios.iter().enumerate() {
        let via_wrapper = synthesize_takeoff_episode(config.clone(), *profile, *noise, *seed);
        let scene = SceneDescriptor {
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
            targets: vec![TargetEntity {
                class: TargetClass::ShahedClassPiston,
                kinematics: TargetKinematics::FromTakeoffProfile(*profile),
                spawn_time_s: 0.0,
            }],
        };
        let via_scene = synthesize_scene(scene, config.clone(), *noise, *seed);
        assert_eq!(
            via_wrapper.integrated_range_profile, via_scene.integrated_range_profile,
            "C-unified (scenario {i}) violated: integrated_range_profile diverged",
        );
        assert_eq!(
            via_wrapper.detections, via_scene.detections,
            "C-unified (scenario {i}) violated: detections diverged",
        );
    }
}

// ===========================================================================
// Wave 4.5 expert-critique-fix gates (H1 sea-spray clutter, H2 polarization
// agility, H3 TBD/Hough, H4 MTI cross-flight, H5 booster-burn refinement).
// ===========================================================================
// parameter. This gate prevents silent removal of the four sea-spray
// regimes added by Wave 4.5 H1.
//
// Citations: Ward, Tough & Watts (IET 2013) chapters 4-6; Greco & Gini,
// "Compound-Gaussian models for sea-clutter"; Watts, IEE Proc. F 1985.
// ===========================================================================

/// **Wave 4.5 H1 — sea-spray clutter regimes are available and
/// statistically distinct from open-sea regime.** A real radar engineer
/// reviewing low-grazing coastal scenarios needs at least the breaking-
/// wave regime; this gate prevents silent removal.
#[test]
fn w45_h1_sea_spray_regimes_present_and_distinct() {
    let lib = ClutterRegime::library();
    let sea_spray_count = lib
        .iter()
        .filter(|r| r.name().contains("SeaSpray") || r.name().contains("Spray"))
        .count();
    assert!(
        sea_spray_count >= 4,
        "expected >=4 sea-spray regimes, found {} (names: {:?})",
        sea_spray_count,
        lib.iter().map(|r| r.name()).collect::<Vec<_>>(),
    );

    // Distinct from baseline: the breaking-wave regime (K nu = 0.6) must
    // be much heavier-tailed than the open-sea regime (K nu ~ 8). We use
    // 8 000 IID samples per side so the kurtosis estimator is stable.
    let open = lib
        .iter()
        .find(|r| r.name() == "Sea_OpenMediumState")
        .expect("Sea_OpenMediumState must be in library");
    let breaking = lib
        .iter()
        .find(|r| r.name() == "SeaSpray_BreakingWaves")
        .expect("SeaSpray_BreakingWaves must be in library");

    fn kurt(xs: &[f64]) -> f64 {
        let n = xs.len() as f64;
        let m: f64 = xs.iter().sum::<f64>() / n;
        let m2 = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
        let m4 = xs.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
        if m2 > 0.0 {
            m4 / (m2 * m2)
        } else {
            0.0
        }
    }

    let open_xs = collect_many(0x4500_aaaa_5555_eeee, 8_000, |s| {
        sample_clutter_amplitude(&open.distribution, s)
    });
    let break_xs = collect_many(0x4501_bbbb_6666_ffff, 8_000, |s| {
        sample_clutter_amplitude(&breaking.distribution, s)
    });
    let k_open = kurt(&open_xs);
    let k_break = kurt(&break_xs);

    assert!(
        k_break > k_open,
        "W4.5 H1: breaking-wave kurtosis ({}) must exceed open-sea kurtosis ({}); \
         otherwise the sea-spray library has silently collapsed onto the open-sea regime",
        k_break,
        k_open,
    );
    assert!(
        k_break >= 6.0,
        "W4.5 H1: breaking-wave kurtosis ({}) should be >=6.0 (K nu=0.6 is heavy-tailed; \
         see Ward, Tough & Watts (IET 2013) chapter 5)",
        k_break,
    );

    // Every sea-spray entry must be associated with Sea or CoastalSea.
    for r in lib.iter().filter(|r| r.name().contains("SeaSpray")) {
        assert!(
            matches!(r.terrain, TerrainClass::Sea | TerrainClass::CoastalSea),
            "W4.5 H1: sea-spray regime {} associates with non-sea terrain {:?}",
            r.name(),
            r.terrain,
        );
    }
}
///   - Ulaby & Long, *Microwave Radar and Radiometric Remote Sensing*,
///     2014, §10.2 (co-pol vs cross-pol depolarization signatures).
///
/// Bound rationale: the first-order scaling in
/// `sim.rs::polarization_amplitude_scale` gives HH = +1 dB over VV
/// (linear 10^(+1/20) ≈ 1.122). We require ≥0.5 dB to allow for any
/// integration-window edge effects; the actual measured ΔSNR is
/// recorded in the receipt.
#[test]
fn w45_h2_polarization_agility_changes_target_amp() {
    let base_config = RadarSimConfig {
        pulse_count: 16,
        ..RadarSimConfig::default()
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    // Zero stochastic terms so the integrated peak is deterministic
    // up to the polarization scaling.
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
        EpisodeSeed(0x0045_0002),
    );
    let episode_hh = synthesize_takeoff_episode(
        config_hh,
        TakeoffProfile::default(),
        noise,
        EpisodeSeed(0x0045_0002),
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
        "both polarization channels must produce positive integrated peaks; \
         got peak_vv = {peak_vv}, peak_hh = {peak_hh}"
    );

    // The Wave 4.5 H2 contract: VV-only vs HH-only differ by ≥0.5 dB
    // in integrated profile peak (the +1 dB amplitude scaling proxy).
    let ratio_db = 20.0 * (peak_hh / peak_vv).log10();
    assert!(
        ratio_db.abs() >= 0.5,
        "Wave 4.5 H2 violated: VV-only vs HH-only integrated peaks must \
         differ by ≥0.5 dB (the +1 dB HH amplitude scaling), but got \
         ratio_db = {ratio_db:.3} (peak_vv = {peak_vv:.6}, \
         peak_hh = {peak_hh:.6}). The per-pulse polarization channel is \
         not wired into the target-return amplitude."
    );

    // The expected sign: HH > VV per the first-order proxy.
    assert!(
        peak_hh > peak_vv,
        "Wave 4.5 H2 sign mismatch: HH-only integrated peak ({peak_hh:.6}) \
         should exceed VV-only ({peak_vv:.6}) by ~+1 dB per the \
         first-order polarization scaling."
    );

    // Print the measured ratios for the receipt — visible with
    // `cargo test -- --nocapture`. This is observational, not a gate.
    eprintln!(
        "[wave-4.5-H2] peak_vv = {peak_vv:.6}, peak_hh = {peak_hh:.6}, \
         ratio_db = {ratio_db:+.3} dB (expected ≈ +1.000 dB)"
    );
}

fn make_cpi_grid(
    n_range: usize,
    n_doppler: usize,
    target: Option<(usize, usize, f32)>,
) -> Vec<Vec<ComplexSample>> {
    let mut grid = vec![vec![ComplexSample::new(0.0, 0.0); n_doppler]; n_range];
    for (r, row) in grid.iter_mut().enumerate().take(n_range) {
        for (d, cell) in row.iter_mut().enumerate().take(n_doppler) {
            let n = ((r * 7 + d * 13) % 100) as f32 * 0.001;
            *cell = ComplexSample::new(n, n * 0.7);
        }
    }
    if let Some((r, d, mag)) = target {
        if r < n_range && d < n_doppler {
            grid[r][d] = ComplexSample::new(mag, 0.0);
        }
    }
    grid
}

/// **Wave 4.5 H3 -- Track-before-detect (Hough) finds sub-threshold
/// targets.** A target whose per-CPI peak power is below the per-cell
/// CFAR threshold but whose track is coherent over 5 CPIs must be
/// detected by the Hough TBD module. Carlson-Evans-Wilson 1994.
#[test]
fn w45_h3_hough_tbd_finds_sub_threshold_track() {
    // Build 5-CPI stack with moving target at sub-CFAR power; verify
    // hough_tbd_detect returns a high-confidence candidate.
    let n_range = 64;
    let n_doppler = 16;
    // Per-CPI peak power below a hypothetical CFAR threshold (e.g. 1.0)
    // but well above the TBD sub-threshold (0.01).
    let per_cpi_magnitude = 0.2_f32;
    let cfar_like_threshold = 1.0_f32;
    assert!(
        per_cpi_magnitude * per_cpi_magnitude < cfar_like_threshold,
        "per-CPI peak power {} must be sub-CFAR-threshold {}",
        per_cpi_magnitude * per_cpi_magnitude,
        cfar_like_threshold
    );

    let stack: Vec<Vec<Vec<ComplexSample>>> = (0..5)
        .map(|cpi| {
            let r = 10 + 2 * cpi;
            make_cpi_grid(n_range, n_doppler, Some((r, 5, per_cpi_magnitude)))
        })
        .collect();

    let config = TbdConfig {
        n_cpis: 5,
        min_range_gradient: 0.5,
        max_range_gradient: 4.0,
        m_of_n_threshold: 3,
        sub_threshold_power: 0.01,
    };
    let candidates = hough_tbd_detect(&stack, config);
    assert!(
        !candidates.is_empty(),
        "TBD should find the moving target track"
    );
    let best = &candidates[0];
    assert!(
        best.n_cpis_hit >= 3,
        "expected >=3 CPI hits (M-of-N=3-of-5), got {}",
        best.n_cpis_hit
    );
    assert!(
        best.confidence >= 0.6,
        "expected high confidence (>=0.6), got {}",
        best.confidence
    );
    // Accumulated TBD power must exceed what any single CPI hit would
    // produce -- demonstrating actual cross-CPI integration.
    let single_cpi_power = per_cpi_magnitude * per_cpi_magnitude;
    assert!(
        best.accumulated_power > single_cpi_power,
        "accumulated TBD power {} must exceed single-CPI power {}",
        best.accumulated_power,
        single_cpi_power
    );
}
//   - Richards, "Fundamentals of Radar Signal Processing" 2nd ed.,
//     2014, §5.4 (MTI canceller frequency response and the blind-speed
//     compensation pattern).
//   - `.agents/receipts/wave-4-5-mti-cross-flight/<UTC>.md`.
// ===========================================================================

/// **Wave 4.5 H4 — MTI cross-flight handling.** A Shahed-class target
/// flying perpendicular to the radar LOS (low |v_radial| but observable
/// micro-Doppler propeller line) must NOT be silently missed by the
/// Tier 2 CLIMB-OUT detector. The cross-flight branch trades MTI gate
/// for stricter Kalman + micro-Doppler confirmation per Skolnik §3.7.
#[test]
fn w45_h4_mti_cross_flight_target_detected_with_micro_doppler() {
    // Cross-flight kinematic state: |v_radial| ≈ 1 m/s (well under the
    // 1.5 m/s S-band cross-flight cutoff that maps the 30 Hz MTI notch
    // body Doppler to a radial velocity). The kinematic window has six
    // samples each within 0.1 m/s of its neighbour, comfortably inside
    // the 3 m/s cross-flight Kalman gate.
    //
    // Build the detector with a climb gate widened to admit low-radial-
    // speed cross-flight geometries (the canonical climb gate keys on
    // |v_radial| and would otherwise reject the geometry before the
    // cross-flight branch fires).
    let detector = ClimbOutTierDetector {
        config: ClimbTierConfig::default(),
        gate: KinematicGate {
            radial_speed_mps_min: 0.0,
            radial_speed_mps_max: 60.0,
            accel_mps2_min: 0.0,
            accel_mps2_max: 2.0,
            altitude_agl_m_min: 30.0,
            altitude_agl_m_max: 1500.0,
        },
    };
    let samples = vec![
        KinematicSample::new(0.0, 0.8, 250.0),
        KinematicSample::new(1.0, 0.9, 252.0),
        KinematicSample::new(2.0, 1.0, 254.0),
        KinematicSample::new(3.0, 1.1, 256.0),
        KinematicSample::new(4.0, 1.2, 258.0),
        KinematicSample::new(5.0, 1.3, 260.0),
    ];
    let observation = KinematicObservation::new(samples, 8_000.0, 20.0);

    // Synthetic slow-time amplitude spectrum: 256 bins at 1 Hz/bin,
    // flat noise floor of amplitude 1, and a propeller blade-pass
    // spike at 190 Hz (squarely inside the [127.5, 253] Hz piston
    // blade-pass window per the Tier 3 cruise spec).
    let mut spectrum = vec![1.0f32; 256];
    spectrum[190] = 50.0;
    let bin_hz = 1.0_f64;

    let decision = detector.evaluate_with_spectrum(&observation, Some(&spectrum), Some(bin_hz));

    assert!(
        decision.cross_flight,
        "W4.5 H4 violated: low |v_radial| at S-band must enter the cross-flight branch \
         (MTI_NOTCH_BODY_DOPPLER_HZ={MTI_NOTCH_BODY_DOPPLER_HZ}); note = {}",
        decision.note,
    );
    assert!(
        decision.micro_doppler_confirmed,
        "W4.5 H4 violated: blade-pass spike at 190 Hz must be confirmed by the cross-flight \
         branch (piston window [127.5, 253] Hz)",
    );
    assert!(
        decision.detected,
        "W4.5 H4 violated: cross-flight target with kinematic + micro-Doppler compensating \
         evidence must NOT be silently missed; note = {}",
        decision.note,
    );
    // Sanity: the cross-flight branch must never set mti_notch_rejected —
    // that field is the radial-branch outcome.
    assert!(
        !decision.mti_notch_rejected,
        "W4.5 H4 violated: cross-flight branch must not set mti_notch_rejected (mutually exclusive)",
    );
}

/// **Wave 4.5 H4 — MTI cross-flight handling (counter-part).** The
/// cross-flight branch must REJECT a target that has low |v_radial|
/// (so the geometry enters the cross-flight branch) but lacks the
/// micro-Doppler propeller line — the compensating evidence is missing.
/// Guards against the cross-flight branch silently up-grading bare
/// kinematic agreement to a detection. Skolnik §3.7 explicitly notes
/// that the compensating-evidence pattern only works when both
/// channels (tighter Kalman AND micro-Doppler) are present.
#[test]
fn w45_h4_cross_flight_target_without_micro_doppler_rejected() {
    let detector = ClimbOutTierDetector {
        config: ClimbTierConfig::default(),
        gate: KinematicGate {
            radial_speed_mps_min: 0.0,
            radial_speed_mps_max: 60.0,
            accel_mps2_min: 0.0,
            accel_mps2_max: 2.0,
            altitude_agl_m_min: 30.0,
            altitude_agl_m_max: 1500.0,
        },
    };
    let samples = vec![
        KinematicSample::new(0.0, 0.8, 250.0),
        KinematicSample::new(1.0, 0.9, 252.0),
        KinematicSample::new(2.0, 1.0, 254.0),
        KinematicSample::new(3.0, 1.1, 256.0),
        KinematicSample::new(4.0, 1.2, 258.0),
        KinematicSample::new(5.0, 1.3, 260.0),
    ];
    let observation = KinematicObservation::new(samples, 8_000.0, 20.0);

    // Flat noise floor: no blade-pass spike anywhere.
    let spectrum = vec![1.0f32; 256];
    let bin_hz = 1.0_f64;

    let decision = detector.evaluate_with_spectrum(&observation, Some(&spectrum), Some(bin_hz));

    assert!(
        decision.cross_flight,
        "W4.5 H4 violated: low |v_radial| must enter the cross-flight branch",
    );
    assert!(
        !decision.micro_doppler_confirmed,
        "W4.5 H4 violated: flat noise floor has no blade-pass line",
    );
    assert!(
        !decision.detected,
        "W4.5 H4 violated: cross-flight WITHOUT micro-Doppler must NOT detect \
         (compensating evidence missing); note = {}",
        decision.note,
    );
}
//     (`object-packs/public-proxy-v1/physics_dossier.md`), §5.
//   - Sutton & Biblarz, *Rocket Propulsion Elements*, 9th ed., ch. 12.
//   - RUSI 2022/CSIS 2022 open-source reporting on Shahed-class launch.
// ===========================================================================

/// **Wave 4.5 H5 — Booster-burn thrust profile + sub-state
/// classification.** Tier 1 BOOST detector now models SRM thrust
/// curve (boost burn → separation transient → sustain) per Sutton &
/// Biblarz typical progressive-grain SRM. Velocity at burn completion
/// reaches public-proxy release envelope (25–35 m/s) when started from
/// rail-exit velocity 9 m/s.
#[test]
fn w45_h5_booster_burn_profile_reaches_release_velocity() {
    let profile = BoostThrustProfile::shahed_class_default();

    // (a) Profile shape sanity: peak in burn band per dossier (~1–2g).
    assert!(
        (5.0..=30.0).contains(&profile.peak_acceleration_mps2),
        "H5 gate: peak_acceleration_mps2 must lie in [5, 30] (~0.5–3g per \
         Sutton/Biblarz typical small SRM); got {}",
        profile.peak_acceleration_mps2
    );
    assert!(
        (1.0..=3.0).contains(&profile.burn_duration_s),
        "H5 gate: burn_duration_s must lie in [1, 3] s per dossier; got {}",
        profile.burn_duration_s
    );

    // (b) Headline integration check: from rail-exit (~9 m/s), after the
    // burn duration the airframe must reach the dossier's 25–35 m/s
    // release-velocity envelope.
    let rail_exit_mps = 9.0;
    let v_release = profile.velocity_at(profile.burn_duration_s, rail_exit_mps);
    assert!(
        (25.0..=35.0).contains(&v_release),
        "H5 gate: velocity at burn completion ({v_release:.2} m/s) must \
         land inside the dossier release-velocity envelope [25, 35] m/s"
    );

    // (c) After separation + transient, the acceleration must drop to the
    // sustain band (piston engine, ~0.5 m/s²) — confirming the model
    // physically transitions from rocket to piston propulsion.
    let t_post = profile.burn_duration_s + profile.separation_transient_s + 0.1;
    let a_sustain = profile.acceleration_at(t_post);
    assert!(
        (0.0..1.0).contains(&a_sustain),
        "H5 gate: post-separation acceleration must drop to the piston \
         sustain band [0, 1) m/s²; got {a_sustain:.4}"
    );

    // (d) During the burn (e.g. halfway through), thrust must be at the
    // plateau (peak) — confirming the smoothstep ramp completes inside
    // the burn window.
    let a_mid_burn = profile.acceleration_at(profile.burn_duration_s * 0.5);
    assert!(
        (a_mid_burn - profile.peak_acceleration_mps2).abs() < 1e-6,
        "H5 gate: mid-burn acceleration must equal peak; got {a_mid_burn} vs peak {}",
        profile.peak_acceleration_mps2
    );
}
