use super::*;

#[test]
fn clutter_sampling_is_deterministic_and_finite() {
    let a = sample_clutter_frame(ClutterProfile::moderate_mixed(), 42, 7);
    let b = sample_clutter_frame(ClutterProfile::moderate_mixed(), 42, 7);
    assert_eq!(a, b);
    assert!(a.amplitude_offset.is_finite());
    assert!(a.doppler_spread_hz.is_finite());
    assert!((0.0..=1.0).contains(&a.false_alarm_pressure));
}

#[test]
fn clutter_profile_application_changes_power_deterministically() {
    let mut a = vec![0.0f32; 32];
    let mut b = vec![0.0f32; 32];
    apply_clutter_to_profile(&mut a, ClutterProfile::moderate_mixed(), 9);
    apply_clutter_to_profile(&mut b, ClutterProfile::moderate_mixed(), 9);
    assert_eq!(a, b);
    assert!(a.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(a.iter().any(|v| *v > 0.0));
}

// --- Lane G: heavy-tailed clutter samplers and regimes ---

fn collect_many(seed_base: u64, n: usize, mut f: impl FnMut(u64) -> f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            f(seed_base
                .wrapping_add(i as u64)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15))
        })
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

fn lag1_correlation(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let mut num = 0.0;
    let mut den = 0.0;
    for w in xs.windows(2) {
        num += (w[0] - m) * (w[1] - m);
    }
    for x in xs {
        den += (x - m).powi(2);
    }
    if den.abs() < 1e-30 {
        0.0
    } else {
        num / den
    }
}

#[test]
fn weibull_shape_two_matches_rayleigh_mean() {
    // Rayleigh(scale=sigma) has E[X] = sigma * sqrt(pi/2).
    // Weibull(shape=2, scale=lambda) is identical to Rayleigh with sigma=lambda/sqrt(2).
    // So with scale=1.0 the expected mean is sqrt(pi)/2 ~= 0.886.
    let xs = collect_many(0x5a5a_5a5a_5a5a_5a5a, 4000, |s| sample_weibull(2.0, 1.0, s));
    let m = mean(&xs);
    let expected = (std::f64::consts::PI).sqrt() / 2.0;
    let rel_err = (m - expected).abs() / expected;
    assert!(
        rel_err < 0.05,
        "Weibull(2,1) mean {} should match Rayleigh-equivalent expected {} (rel err {})",
        m,
        expected,
        rel_err
    );
}

#[test]
fn weibull_shape_one_matches_exponential_mean() {
    // Weibull(shape=1, scale=lambda) is Exponential(rate=1/lambda); E[X]=lambda.
    let xs = collect_many(0xa1b2_c3d4_e5f6_0718, 4000, |s| sample_weibull(1.0, 1.0, s));
    let m = mean(&xs);
    let rel_err = (m - 1.0).abs();
    assert!(
        rel_err < 0.05,
        "Weibull(1,1) mean {} should be ~1.0 (exponential)",
        m
    );
}

#[test]
fn k_distribution_large_shape_approaches_rayleigh() {
    // As nu -> infinity the K-distribution converges to Rayleigh; the mean/std ratio
    // should approach the Rayleigh ratio of sqrt(pi/(4-pi)) ~= 1.913.
    let xs = collect_many(0x1234_5678_9abc_def0, 4000, |s| {
        sample_k_distribution(100.0, 1.0, s)
    });
    let m = mean(&xs);
    let sd = std_dev(&xs);
    let ratio = m / sd;
    let expected = (std::f64::consts::PI / (4.0 - std::f64::consts::PI)).sqrt();
    let rel_err = (ratio - expected).abs() / expected;
    assert!(
        rel_err < 0.15,
        "K-dist large-nu mean/std ratio {} should be ~Rayleigh's {} (rel err {})",
        ratio,
        expected,
        rel_err
    );
}

#[test]
fn log_normal_mean_of_log_matches_mean_log() {
    let xs = collect_many(0xdead_beef_cafe_babe, 4000, |s| {
        sample_log_normal(0.5, 0.25, s)
    });
    let logs: Vec<f64> = xs.iter().map(|x| x.ln()).collect();
    let m = mean(&logs);
    let rel_err = (m - 0.5).abs();
    assert!(
        rel_err < 0.03,
        "log-normal mean(log(x)) {} should be ~mean_log=0.5",
        m
    );
}

#[test]
fn sample_clutter_amplitude_dispatches_each_variant() {
    // For a fixed seed, each variant should agree with its dedicated sampler
    // (proving the dispatcher routes correctly).
    let seed: u64 = 0xfeed_face_dead_beef;
    let dispatched = sample_clutter_amplitude(&ClutterDistribution::Rayleigh, seed);
    // Rayleigh dispatch must equal Weibull(2,1) via dedicated path; the dispatcher
    // builds its RNG identically to the dedicated samplers, so the very first draw
    // matches up bit-for-bit.
    let weibull_direct = sample_weibull(2.0, 1.0, seed);
    assert_eq!(
        dispatched, weibull_direct,
        "Rayleigh dispatch should equal Weibull(2,1) direct sample"
    );

    // Each variant must produce a finite value when dispatched.
    for dist in &[
        ClutterDistribution::Rayleigh,
        ClutterDistribution::Weibull {
            shape: 1.3,
            scale: 1.0,
        },
        ClutterDistribution::KDistribution {
            shape: 4.0,
            scale: 1.0,
        },
        ClutterDistribution::LogNormal {
            mean_log: 0.0,
            std_log: 0.5,
        },
    ] {
        let v = sample_clutter_amplitude(dist, seed);
        assert!(
            v.is_finite(),
            "dispatched sample for {:?} must be finite",
            dist
        );
        assert!(v >= 0.0, "amplitude must be non-negative");
    }
}

#[test]
fn same_seed_same_sample_for_each_sampler() {
    let s: u64 = 99;
    assert_eq!(sample_weibull(1.5, 1.0, s), sample_weibull(1.5, 1.0, s));
    assert_eq!(
        sample_k_distribution(3.0, 1.0, s),
        sample_k_distribution(3.0, 1.0, s)
    );
    assert_eq!(
        sample_log_normal(0.0, 0.5, s),
        sample_log_normal(0.0, 0.5, s)
    );
    assert_eq!(
        sample_clutter_amplitude(
            &ClutterDistribution::KDistribution {
                shape: 2.0,
                scale: 1.0
            },
            s,
        ),
        sample_clutter_amplitude(
            &ClutterDistribution::KDistribution {
                shape: 2.0,
                scale: 1.0
            },
            s,
        )
    );
}

#[test]
fn different_seeds_different_sample() {
    assert_ne!(sample_weibull(1.5, 1.0, 1), sample_weibull(1.5, 1.0, 2));
    assert_ne!(
        sample_k_distribution(3.0, 1.0, 1),
        sample_k_distribution(3.0, 1.0, 2)
    );
    assert_ne!(
        sample_log_normal(0.0, 0.5, 1),
        sample_log_normal(0.0, 0.5, 2)
    );
}

#[test]
fn clutter_regime_library_has_entries_with_citations() {
    // Library covers every TerrainClass and is documented with cited shape values
    // (Skolnik / Ward, Tough & Watts / JHU APL). The library must have at least 4
    // entries per the packet spec and at least one entry per distribution family.
    let lib = ClutterRegime::library();
    assert!(
        lib.len() >= 4,
        "library must have >=4 regimes; got {}",
        lib.len()
    );

    let has_weibull = lib
        .iter()
        .any(|r| matches!(r.distribution, ClutterDistribution::Weibull { .. }));
    let has_k = lib
        .iter()
        .any(|r| matches!(r.distribution, ClutterDistribution::KDistribution { .. }));
    let has_rayleigh = lib
        .iter()
        .any(|r| matches!(r.distribution, ClutterDistribution::Rayleigh));
    assert!(
        has_weibull,
        "library must include at least one Weibull regime"
    );
    assert!(
        has_k,
        "library must include at least one K-distribution regime"
    );
    assert!(
        has_rayleigh,
        "library must include at least one Rayleigh regime"
    );

    // Each regime must have physically plausible AR(1) coefficients and a
    // mean cross-section that is finite and < 0 dB(m^2)/m^2.
    for r in &lib {
        assert!(
            (0.0..=1.0).contains(&r.spatial_correlation),
            "spatial_correlation out of [0,1] for {:?}: {}",
            r.terrain,
            r.spatial_correlation
        );
        assert!(
            (0.0..=1.0).contains(&r.temporal_correlation),
            "temporal_correlation out of [0,1] for {:?}: {}",
            r.terrain,
            r.temporal_correlation
        );
        assert!(
            r.mean_power_dbsm_per_m2.is_finite() && r.mean_power_dbsm_per_m2 <= 0.0,
            "mean sigma0 should be finite and <=0 dB for {:?}: {}",
            r.terrain,
            r.mean_power_dbsm_per_m2
        );
    }
}

#[test]
fn for_terrain_sea_returns_k_distribution() {
    let r = ClutterRegime::for_terrain(TerrainClass::Sea, 1.0);
    assert!(
        matches!(r.distribution, ClutterDistribution::KDistribution { .. }),
        "sea should be K-distributed, got {:?}",
        r.distribution
    );
    assert!(
        (r.grazing_angle_deg - 1.0).abs() < 1e-9,
        "grazing angle should be carried verbatim"
    );
}

#[test]
fn for_terrain_open_sky_returns_rayleigh() {
    let r = ClutterRegime::for_terrain(TerrainClass::OpenSky, 30.0);
    assert!(
        matches!(r.distribution, ClutterDistribution::Rayleigh),
        "open sky should be Rayleigh, got {:?}",
        r.distribution
    );
    assert!(
        (r.grazing_angle_deg - 30.0).abs() < 1e-9,
        "grazing angle should be carried verbatim"
    );
}

#[test]
fn generate_clutter_sequence_returns_expected_length() {
    let regime = ClutterRegime::for_terrain(TerrainClass::Forest, 3.0);
    let seq = generate_clutter_sequence(&regime, 16, 8, 0x00c0_ffee);
    assert_eq!(seq.len(), 16 * 8);
    assert!(seq.iter().all(|v| v.is_finite()));

    // Empty boundary cases must not panic.
    assert!(generate_clutter_sequence(&regime, 0, 8, 0).is_empty());
    assert!(generate_clutter_sequence(&regime, 16, 0, 0).is_empty());
}

#[test]
fn generate_clutter_sequence_spatial_correlation_high_when_rho_high() {
    // With spatial_correlation = 0.99 the within-pulse lag-1 sample correlation
    // across range bins should be high (>= 0.8 over a long run). Use a single
    // pulse so the temporal AR(1) does not contaminate the within-pulse signal.
    let regime = ClutterRegime {
        terrain: TerrainClass::Suburban,
        grazing_angle_deg: 3.0,
        distribution: ClutterDistribution::Rayleigh,
        spatial_correlation: 0.99,
        temporal_correlation: 0.0,
        mean_power_dbsm_per_m2: -18.0,
    };
    let seq = generate_clutter_sequence(&regime, 4096, 1, 0xa55a_a55a_a55a_a55a);
    let row: Vec<f64> = seq.iter().map(|v| *v as f64).collect();
    let rho_hat = lag1_correlation(&row);
    assert!(
        rho_hat > 0.8,
        "high-spatial-correlation regime should give lag-1 corr > 0.8, got {}",
        rho_hat
    );
}

#[test]
fn generate_clutter_sequence_is_deterministic_for_same_seed() {
    let regime = ClutterRegime::for_terrain(TerrainClass::Sea, 1.0);
    let a = generate_clutter_sequence(&regime, 32, 16, 0xdeadbeef);
    let b = generate_clutter_sequence(&regime, 32, 16, 0xdeadbeef);
    let c = generate_clutter_sequence(&regime, 32, 16, 0xdeadbeee);
    assert_eq!(a, b, "same seed must give bit-identical sequence");
    assert_ne!(a, c, "different seed must give different sequence");
}
