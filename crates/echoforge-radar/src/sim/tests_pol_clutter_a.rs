use super::*;
use crate::clutter::{ClutterDistribution, ClutterRegime, TerrainClass};
use crate::rcs::Polarization;

// ---------------------------------------------------------------
// Lane C — ClutterRegime wiring tests.
//
// These verify that:
//   (a) the default `NoiseProfile::real_world_proxy_v1()` keeps
//       `clutter_regime: None` so existing Gaussian-AR(1) byte-
//       stable fixtures are unchanged,
//   (b) wiring a K-distribution regime through the synthesis loop
//       actually produces heavy-tailed IQ samples (empirical
//       kurtosis well above the Gaussian baseline of 3), and
//   (c) the K/Weibull path is fully deterministic for a fixed
//       seed, matching the determinism guarantee of
//       `generate_clutter_sequence`.
// ---------------------------------------------------------------

fn empirical_kurtosis(xs: &[f64]) -> f64 {
    // Pearson's kurtosis: m4 / m2^2 (NOT excess kurtosis). For a
    // Gaussian this is 3.0; for K-distribution(nu) it diverges as
    // nu -> 0 (Ward, Tough & Watts chap. 2).
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let m2 = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m4 = xs.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
    if m2 == 0.0 {
        0.0
    } else {
        m4 / (m2 * m2)
    }
}

/// (a) bridged — the canonical `real_world_proxy_v1` profile
/// must default to the prior Gaussian AR(1) clutter path so
/// existing byte-stable fixtures stay valid.
#[test]
fn noise_profile_default_uses_gaussian_clutter() {
    let noise = NoiseProfile::real_world_proxy_v1();
    assert!(
        noise.clutter_regime.is_none(),
        "default real_world_proxy_v1 must keep clutter_regime=None \
         for byte-stable bridged"
    );
    assert_eq!(noise.clutter_sigma_0_scale, 1.0);

    // Determinism: with the default Gaussian profile, two runs must
    // remain byte-equal (this is just the pre-existing contract,
    // re-asserted here to gate the bridged property).
    let config = RadarSimConfig {
        pulse_count: 6,
        ..RadarSimConfig::default()
    };
    let a = synthesize_takeoff_episode(
        config.clone(),
        TakeoffProfile::default(),
        noise,
        EpisodeSeed(101),
    );
    let b = synthesize_takeoff_episode(config, TakeoffProfile::default(), noise, EpisodeSeed(101));
    assert_eq!(a.integrated_range_profile, b.integrated_range_profile);
}

/// (b) K-distribution wiring — a `NoiseProfile` with a low-`nu`
/// K-distribution regime should produce IQ clutter with empirical
/// kurtosis well above the Gaussian baseline (3.0). We use a
/// spiky regime (`nu = 0.8`) and check the *real-part marginal*
/// of the IQ samples; with low awgn_sigma the clutter term
/// dominates and the heavy-tailed signature is preserved.
#[test]
fn noise_profile_with_k_regime_uses_k_distribution() {
    let config = RadarSimConfig {
        pulse_count: 32,
        ..RadarSimConfig::default()
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    // Strip orthogonal noise sources so the kurtosis we measure
    // reflects the clutter generator, not phase noise / glints /
    // RFI / AWGN.
    noise.awgn_sigma = 1e-5;
    noise.phase_noise_std_rad = 0.0;
    noise.amplitude_scintillation_sigma = 0.0;
    noise.rfi_probability = 0.0;
    noise.ground_glint_count = 0;
    // Custom regime: K-distribution shape nu = 0.8, zero AR(1)
    // correlation so adjacent samples are independent draws.
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

    let episode =
        synthesize_takeoff_episode(config, TakeoffProfile::default(), noise, EpisodeSeed(2026));
    // Flatten the IQ real parts. The target return is concentrated
    // in a tiny range of bins (small support), so the IQ histogram
    // is dominated by the per-bin clutter draws.
    let samples: Vec<f64> = episode.iq.iter().flatten().map(|c| c.re as f64).collect();
    assert!(!samples.is_empty());
    let kurtosis = empirical_kurtosis(&samples);
    assert!(
        kurtosis > 3.0,
        "K(nu=0.8) clutter must exhibit heavy tails (kurtosis > 3 \
         — Gaussian baseline); got {kurtosis:.3}"
    );
}

/// (c) Reproducibility — same regime + same seed must yield the
/// same IQ. `generate_clutter_sequence` is byte-stable per its
/// docs; this test gates the wiring.
#[test]
fn noise_profile_with_weibull_regime_reproducible() {
    let config = RadarSimConfig {
        pulse_count: 8,
        ..RadarSimConfig::default()
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.clutter_regime = Some(ClutterRegime::for_terrain(TerrainClass::Forest, 3.0));

    let a = synthesize_takeoff_episode(
        config.clone(),
        TakeoffProfile::default(),
        noise,
        EpisodeSeed(31337),
    );
    let b =
        synthesize_takeoff_episode(config, TakeoffProfile::default(), noise, EpisodeSeed(31337));
    assert_eq!(a.integrated_range_profile, b.integrated_range_profile);
    assert_eq!(a.detections, b.detections);
    // IQ byte-equality is the strictest form of determinism.
    for (pulse_a, pulse_b) in a.iq.iter().zip(b.iq.iter()) {
        assert_eq!(pulse_a.len(), pulse_b.len());
        for (sa, sb) in pulse_a.iter().zip(pulse_b.iter()) {
            assert_eq!(sa.re.to_bits(), sb.re.to_bits());
            assert_eq!(sa.im.to_bits(), sb.im.to_bits());
        }
    }
}
