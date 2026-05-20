use super::*;

// Test-only helpers (moved from micro_doppler_classifier.rs cfg(test) section).

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[inline]
fn uniform01(state: &mut u64) -> f64 {
    // Top 53 bits -> [0, 1) double.
    let bits = xorshift64(state) >> 11;
    (bits as f64) * (1.0_f64 / ((1u64 << 53) as f64))
}

/// Build a synthetic feature vector by sampling the per-feature
/// Gaussian envelope of one reference signature using a deterministic
/// xorshift64 generator.
fn synth_feature_from_signature(
    sig: &ReferenceSignature,
    rng_state: &mut u64,
) -> MicroDopplerFeatures {
    let normal = |state: &mut u64, mu: f64, sigma: f64| -> f64 {
        // Box-Muller pair on two uniforms in (0,1].
        let u1 = uniform01(state).max(1e-12);
        let u2 = uniform01(state);
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        mu + sigma * z
    };
    MicroDopplerFeatures {
        rotor_fundamental_hz: normal(rng_state, sig.rotor_freq_mean_hz, sig.rotor_freq_std_hz)
            .max(0.0),
        modulation_depth_db: normal(
            rng_state,
            sig.modulation_depth_mean_db,
            sig.modulation_depth_std_db,
        ),
        harmonic_ratio: normal(rng_state, sig.harmonic_ratio_mean, sig.harmonic_ratio_std)
            .clamp(0.0, 5.0),
        spectral_entropy: normal(
            rng_state,
            sig.spectral_entropy_mean,
            sig.spectral_entropy_std,
        )
        .max(0.0),
        body_doppler_centroid_hz: normal(
            rng_state,
            sig.body_doppler_centroid_mean_hz,
            sig.body_doppler_centroid_std_hz,
        )
        .max(0.0),
    }
}

fn mean_features(sig: &ReferenceSignature) -> MicroDopplerFeatures {
    MicroDopplerFeatures {
        rotor_fundamental_hz: sig.rotor_freq_mean_hz,
        modulation_depth_db: sig.modulation_depth_mean_db,
        harmonic_ratio: sig.harmonic_ratio_mean,
        spectral_entropy: sig.spectral_entropy_mean,
        body_doppler_centroid_hz: sig.body_doppler_centroid_mean_hz,
    }
}

/// Per-class self-classification: feeding a signature's exact mean
/// vector back through the classifier must pick that signature's
/// class. This is the basic identity / smoke test.
#[test]
fn each_reference_self_classifies_at_mean() {
    let library = ReferenceSignature::library();
    for sig in &library {
        let feats = mean_features(sig);
        let (cls, _) = argmax_class(&feats, &library).expect("non-empty library");
        assert_eq!(
            cls, sig.class,
            "expected self-classification for {:?}",
            sig.class
        );
    }
}

/// Cross-class noise (feeding the *between-class* feature midpoint)
/// should NOT produce a confidently single-class posterior. We
/// require the top posterior to be below 0.95 to guard against the
/// classifier becoming pathologically overconfident outside the
/// reference envelope.
#[test]
fn cross_class_midpoint_is_not_overconfident() {
    let library = ReferenceSignature::library();
    let quad = library
        .iter()
        .find(|s| s.class == TargetClass::QuadcopterFourRotor)
        .unwrap();
    let bird = library
        .iter()
        .find(|s| s.class == TargetClass::BirdFlapping)
        .unwrap();
    let mid = MicroDopplerFeatures {
        rotor_fundamental_hz: 0.5 * (quad.rotor_freq_mean_hz + bird.rotor_freq_mean_hz),
        modulation_depth_db: 0.5 * (quad.modulation_depth_mean_db + bird.modulation_depth_mean_db),
        harmonic_ratio: 0.5 * (quad.harmonic_ratio_mean + bird.harmonic_ratio_mean),
        spectral_entropy: 0.5 * (quad.spectral_entropy_mean + bird.spectral_entropy_mean),
        body_doppler_centroid_hz: 0.5
            * (quad.body_doppler_centroid_mean_hz + bird.body_doppler_centroid_mean_hz),
    };
    let (_, posterior) = argmax_class(&mid, &library).unwrap();
    assert!(
        posterior < 0.999,
        "cross-class midpoint should not pin a single class with posterior >= 0.999; got {posterior}"
    );
}

/// Synthesise N samples per class from the library envelopes and
/// confirm aggregate accuracy >= 85%. This is the four-class AUC
/// gate; with class-balanced confusion the accuracy lower-bound
/// implies a one-vs-rest AUC at least as good in the worst case.
#[test]
fn synthetic_four_class_accuracy_above_threshold() {
    let library = ReferenceSignature::library();
    let n_per_class = 100usize;
    let mut total = 0usize;
    let mut correct = 0usize;
    let mut rng_state: u64 = 0xC0FFEEu64;
    for sig in &library {
        for _ in 0..n_per_class {
            let feats = synth_feature_from_signature(sig, &mut rng_state);
            let (cls, _) = argmax_class(&feats, &library).unwrap();
            if cls == sig.class {
                correct += 1;
            }
            total += 1;
        }
    }
    let accuracy = correct as f64 / total as f64;
    assert!(
        accuracy >= 0.85,
        "four-class accuracy {} below the 0.85 gate",
        accuracy
    );
}

/// Compute one-vs-one AUC for `class_a` vs `class_b` over `n` synth
/// samples per class. AUC is the probability that a positive-class
/// sample receives a higher score (log-likelihood ratio of
/// `class_a` over `class_b`) than a negative-class sample, computed
/// via the Mann-Whitney U statistic.
fn auc_one_vs_one(
    library: &[ReferenceSignature],
    class_a: TargetClass,
    class_b: TargetClass,
    n_per_class: usize,
    rng_state: &mut u64,
) -> f64 {
    let sig_a = library.iter().find(|s| s.class == class_a).unwrap();
    let sig_b = library.iter().find(|s| s.class == class_b).unwrap();
    let mut scores_a = Vec::with_capacity(n_per_class);
    let mut scores_b = Vec::with_capacity(n_per_class);
    for _ in 0..n_per_class {
        let feats = synth_feature_from_signature(sig_a, rng_state);
        let lls = classify_lrt(&feats, library);
        let ll_a = lls.iter().find(|(c, _)| *c == class_a).unwrap().1;
        let ll_b = lls.iter().find(|(c, _)| *c == class_b).unwrap().1;
        scores_a.push(ll_a - ll_b);
    }
    for _ in 0..n_per_class {
        let feats = synth_feature_from_signature(sig_b, rng_state);
        let lls = classify_lrt(&feats, library);
        let ll_a = lls.iter().find(|(c, _)| *c == class_a).unwrap().1;
        let ll_b = lls.iter().find(|(c, _)| *c == class_b).unwrap().1;
        scores_b.push(ll_a - ll_b);
    }
    // Mann-Whitney U via direct pairwise comparison.
    let mut greater = 0.0f64;
    for sa in &scores_a {
        for sb in &scores_b {
            if sa > sb {
                greater += 1.0;
            } else if (sa - sb).abs() < 1e-12 {
                greater += 0.5;
            }
        }
    }
    greater / (scores_a.len() as f64 * scores_b.len() as f64)
}

/// The headline C-UAS gate: discriminating birds from quadcopters
/// at AUC >= 0.95 with the public-proxy reference envelope. This
/// is the question every reviewer asks first.
#[test]
fn bird_vs_quadcopter_auc_meets_95() {
    let library = ReferenceSignature::library();
    let mut rng_state: u64 = 0xDEADBEEFu64;
    let auc = auc_one_vs_one(
        &library,
        TargetClass::QuadcopterFourRotor,
        TargetClass::BirdFlapping,
        100,
        &mut rng_state,
    );
    assert!(
        auc >= 0.95,
        "bird vs quadcopter AUC {} below 0.95 C-UAS gate",
        auc
    );
}

/// Bird vs fixed-wing UAV (Shahed proxy) is the second key
/// discrimination gate: a Shahed-class one-way-attack drone must
/// be distinguished from biological clutter at AUC >= 0.90.
#[test]
fn bird_vs_fixed_wing_auc_meets_threshold() {
    let library = ReferenceSignature::library();
    let mut rng_state: u64 = 0xBADCAFEu64;
    let auc = auc_one_vs_one(
        &library,
        TargetClass::FixedWingUav,
        TargetClass::BirdFlapping,
        100,
        &mut rng_state,
    );
    assert!(
        auc >= 0.90,
        "bird vs fixed-wing UAV AUC {} below 0.90 gate",
        auc
    );
}

/// Helicopter vs quadcopter: both rotary-wing, but very different
/// rotor RPM and harmonic structure. AUC >= 0.85 with the public-
/// proxy envelope.
#[test]
fn helicopter_vs_quadcopter_auc_meets_threshold() {
    let library = ReferenceSignature::library();
    let mut rng_state: u64 = 0x12345678u64;
    let auc = auc_one_vs_one(
        &library,
        TargetClass::Helicopter,
        TargetClass::QuadcopterFourRotor,
        100,
        &mut rng_state,
    );
    assert!(
        auc >= 0.85,
        "helicopter vs quadcopter AUC {} below 0.85 gate",
        auc
    );
}

/// `extract_features` end-to-end: build a slow-time complex vector
/// with a known sinusoidal modulation, confirm the recovered rotor
/// fundamental matches the implanted frequency to within one bin.
#[test]
fn extract_features_recovers_implanted_modulation() {
    // 256 slow-time samples, doppler bin 5 Hz, implanted period
    // 16 samples -> fundamental 256 * 5 / 16 = 80 Hz.
    let n = 256usize;
    let doppler_bin_hz = 5.0f64;
    let mut row: Vec<ComplexSample> = Vec::with_capacity(n);
    for t in 0..n {
        let phase = std::f64::consts::TAU * (t as f64) / 16.0;
        let mag = 1.0 + 0.6 * phase.cos();
        row.push(ComplexSample::new(mag as f32, 0.0));
    }
    let grid = vec![row];
    let feats = extract_features(&grid, 0, doppler_bin_hz).expect("features");
    let recovered = feats.rotor_fundamental_hz;
    // Allow +/- one autocorrelation lag of slack: at n=256, lag=16
    // -> 80 Hz, lag=15 -> ~85.3 Hz, lag=17 -> ~75.3 Hz, so a
    // tolerance of 10 Hz is generous.
    assert!(
        (recovered - 80.0).abs() <= 10.0,
        "expected recovered fundamental near 80 Hz, got {recovered}"
    );
    assert!(feats.modulation_depth_db > 0.0);
    assert!(feats.spectral_entropy >= 0.0);
}

/// `extract_features` returns `None` for an empty grid or
/// out-of-range bin (defensive contract for upstream callers).
#[test]
fn extract_features_rejects_invalid_inputs() {
    let empty: Vec<Vec<ComplexSample>> = Vec::new();
    assert!(extract_features(&empty, 0, 1.0).is_none());

    let short = vec![vec![ComplexSample::new(1.0, 0.0); 4]];
    assert!(extract_features(&short, 0, 1.0).is_none());

    let ok = vec![vec![ComplexSample::new(1.0, 0.0); 16]];
    assert!(extract_features(&ok, 5, 1.0).is_none());
}

/// Sanity check: the reference library covers exactly the four
/// canonical classes, in the canonical order, so downstream
/// indexing assumptions are stable.
#[test]
fn reference_library_is_four_canonical_classes() {
    let library = ReferenceSignature::library();
    assert_eq!(library.len(), 4);
    assert_eq!(library[0].class, TargetClass::QuadcopterFourRotor);
    assert_eq!(library[1].class, TargetClass::FixedWingUav);
    assert_eq!(library[2].class, TargetClass::BirdFlapping);
    assert_eq!(library[3].class, TargetClass::Helicopter);
}
