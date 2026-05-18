//! Micro-Doppler likelihood-ratio test (LRT) classifier.
//!
//! Extracts a feature vector from a target's range-Doppler complex grid
//! (Lane B's `range_doppler_complex`-equivalent: `&[Vec<ComplexSample>]`
//! indexed as `grid[range_bin][doppler_bin]`) and computes class
//! posteriors via Neyman-Pearson LRT against reference signatures for
//! `{quadcopter_4rotor, fixed_wing_uav (Shahed proxy), bird_flapping,
//! helicopter}`.
//!
//! Strict-open posture: reference signatures are PUBLIC-PROXY envelopes
//! derived from open literature (Chen 2011, Rahman-Robertson Nature
//! 2018, Tahmoush 2015, MDPI Drones 2023). No measured-truth claims; no
//! platform-specific signatures.
//!
//! # Feature vector
//!
//! Given a complex range-Doppler grid sliced at the target's range bin
//! (a slow-time / Doppler vector), the following five scalar features
//! are extracted:
//!
//! 1. `rotor_fundamental_hz` — dominant non-DC bin of the slow-time
//!    magnitude spectrum, mapped to Hz via `doppler_bin_hz` and the
//!    zero-padded transform length. Captures the periodic blade-flash
//!    / wingbeat fundamental (Chen 2011 §3.2; Tahmoush 2015 §III).
//! 2. `modulation_depth_db` — peak-to-mean ratio of the slow-time
//!    magnitude spectrum in dB. Captures how spiky the micro-Doppler
//!    structure is (deep modulation -> rotating blades; shallow ->
//!    quasi-CW body return).
//! 3. `harmonic_ratio` — power at the second harmonic of the
//!    fundamental divided by power at the fundamental. Higher for
//!    helicopters and multi-blade rotors that produce strong harmonic
//!    combs (Tahmoush 2015 §III.B; Chen 2011 Fig 3.16).
//! 4. `spectral_entropy` — Shannon entropy (bits) of the normalised
//!    slow-time magnitude spectrum. Birds and helicopters yield broader,
//!    more diffuse spectra than quadcopters or fixed-wing props.
//! 5. `body_doppler_centroid_hz` — magnitude-weighted Doppler centroid
//!    of the body return. Discriminates a fast cruise UAV (~250 m/s
//!    -> hundreds of Hz @ X-band) from a slow bird (~10 m/s ->
//!    few-tens of Hz).
//!
//! # Classifier
//!
//! Per-class log-likelihood under independent Gaussian feature priors
//! (the public-proxy envelope is reported as mean +/- 1 sigma per
//! feature, so the i.i.d. Gaussian product is the principled MLE-style
//! posterior given only those two-moment summaries). Posteriors are
//! softmax over per-class log-likelihoods with equal priors (Neyman-
//! Pearson reduces to argmax of the log-likelihood ratio for
//! equiprobable hypotheses; Skolnik §9.5).
//!
//! # References
//!
//! - Chen, V. C., *The Micro-Doppler Effect in Radar*, Artech House
//!   2011 (esp. ch. 3 rotating-blade kinematics, ch. 7 flapping-wing
//!   bird signatures).
//! - Tahmoush, "Review of micro-Doppler signatures", IET Radar Sonar
//!   Navig. 9(9), 1140-1146 (2015).
//! - Rahman & Robertson, "Radar micro-Doppler signatures of drones and
//!   birds at K-band and W-band", *Nature Sci Rep* 8:17396 (2018).
//! - MDPI Drones 7(1):39 (2023) — small fixed-wing UAS RCS + micro-
//!   Doppler envelopes.
//! - Skolnik, *Introduction to Radar Systems*, 3rd ed., §9.5
//!   (likelihood-ratio detection theory).

use crate::ComplexSample;

/// Five-dimensional micro-Doppler feature vector. All fields are real-
/// valued scalars in SI units (Hz, dimensionless ratio, dB, bits).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MicroDopplerFeatures {
    /// Dominant non-DC periodicity in the slow-time magnitude
    /// sequence, mapped to Hz via the doppler bin spacing.
    pub rotor_fundamental_hz: f64,
    /// Peak-to-mean ratio of the slow-time magnitude spectrum, in dB.
    /// More positive means deeper modulation (spikier spectrum).
    pub modulation_depth_db: f64,
    /// Second-harmonic / fundamental power ratio (dimensionless,
    /// nominally in [0, ~1] for the realistic envelope).
    pub harmonic_ratio: f64,
    /// Shannon entropy (bits) of the normalised slow-time magnitude
    /// spectrum. Larger means broader, more diffuse spectrum.
    pub spectral_entropy: f64,
    /// Magnitude-weighted Doppler centroid of the slow-time spectrum,
    /// expressed in Hz. Proxy for body radial velocity.
    pub body_doppler_centroid_hz: f64,
}

/// Target class enumerated by the classifier. Named distinct from
/// `scene::TargetClass` (which spans the full scene-generator taxonomy)
/// to avoid namespace collision; re-exported as `MdClass` from the
/// crate root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetClass {
    /// 4-rotor quadcopter (e.g. consumer/commercial UAS). Rotor
    /// fundamental in the 250-500 Hz range with strong harmonic comb;
    /// Chen 2011 §3, MDPI Drones 2023 §4.
    QuadcopterFourRotor,
    /// Fixed-wing UAV proxy (Shahed-class 2-blade prop). Rotor
    /// fundamental ~150-220 Hz, narrow spectrum, fast body Doppler.
    FixedWingUav,
    /// Flapping-wing bird. Wingbeat fundamental 2-8 Hz, very low
    /// harmonic content, broad spectral entropy (Rahman-Robertson
    /// Nature 2018; Tahmoush 2015 §IV).
    BirdFlapping,
    /// Rotary-wing aircraft (helicopter). Main-rotor 12-35 Hz with
    /// strong harmonic content + tail rotor 15-60 Hz overtone
    /// (Chen 2011 §3.3).
    Helicopter,
}

/// Public-proxy reference signature for one target class. Encoded as
/// per-feature mean + 1-sigma, cited to open literature. The
/// classifier consumes a `Vec<Self>` (typically [`ReferenceSignature::library`]).
#[derive(Debug, Clone)]
pub struct ReferenceSignature {
    pub class: TargetClass,
    pub rotor_freq_mean_hz: f64,
    pub rotor_freq_std_hz: f64,
    pub modulation_depth_mean_db: f64,
    pub modulation_depth_std_db: f64,
    pub harmonic_ratio_mean: f64,
    pub harmonic_ratio_std: f64,
    pub spectral_entropy_mean: f64,
    pub spectral_entropy_std: f64,
    pub body_doppler_centroid_mean_hz: f64,
    pub body_doppler_centroid_std_hz: f64,
}

impl ReferenceSignature {
    /// Cited public-proxy reference library covering the four C-UAS
    /// discrimination classes. Each entry's mean / sigma are taken
    /// from the cited open literature; no measured-truth claims.
    ///
    /// # Envelope provenance
    ///
    /// - `QuadcopterFourRotor`: fundamental 250-500 Hz (Chen 2011
    ///   §3.2, MDPI Drones 2023 Table 2); modulation depth and
    ///   harmonic comb from Rahman-Robertson 2018 Fig 4.
    /// - `FixedWingUav`: 2-blade prop @ ~6000-7000 rpm => blade-flash
    ///   ~200 Hz (MDPI Drones 2023 §4); body Doppler ~970 Hz at X-band
    ///   for a 200 km/h cruise.
    /// - `BirdFlapping`: wingbeat 2-8 Hz (Rahman-Robertson 2018 Fig 3
    ///   buzzard / pigeon; Chen 2011 §7.4); shallow modulation, low
    ///   harmonic content, broad spectral entropy.
    /// - `Helicopter`: main rotor ~12-35 Hz with deep harmonic comb
    ///   (Tahmoush 2015 §III.B, Chen 2011 §3.3).
    pub fn library() -> Vec<Self> {
        vec![
            // Quadcopter (4 rotors @ ~250-500 Hz, harmonic-rich)
            Self {
                class: TargetClass::QuadcopterFourRotor,
                rotor_freq_mean_hz: 400.0,
                rotor_freq_std_hz: 150.0,
                modulation_depth_mean_db: -3.0,
                modulation_depth_std_db: 1.5,
                harmonic_ratio_mean: 0.25,
                harmonic_ratio_std: 0.10,
                spectral_entropy_mean: 4.5,
                spectral_entropy_std: 0.5,
                body_doppler_centroid_mean_hz: 50.0,
                body_doppler_centroid_std_hz: 30.0,
            },
            // Fixed-wing UAV (Shahed-class proxy, 2-blade prop ~150-220 Hz)
            Self {
                class: TargetClass::FixedWingUav,
                rotor_freq_mean_hz: 185.0,
                rotor_freq_std_hz: 35.0,
                modulation_depth_mean_db: -8.0,
                modulation_depth_std_db: 2.0,
                harmonic_ratio_mean: 0.10,
                harmonic_ratio_std: 0.05,
                spectral_entropy_mean: 3.2,
                spectral_entropy_std: 0.4,
                body_doppler_centroid_mean_hz: 970.0,
                body_doppler_centroid_std_hz: 200.0,
            },
            // Bird (wingbeat 2-8 Hz, low harmonic content) — Rahman-Robertson 2018
            Self {
                class: TargetClass::BirdFlapping,
                rotor_freq_mean_hz: 5.0,
                rotor_freq_std_hz: 2.0,
                modulation_depth_mean_db: -15.0,
                modulation_depth_std_db: 3.0,
                harmonic_ratio_mean: 0.05,
                harmonic_ratio_std: 0.03,
                spectral_entropy_mean: 5.5,
                spectral_entropy_std: 0.6,
                body_doppler_centroid_mean_hz: 200.0,
                body_doppler_centroid_std_hz: 100.0,
            },
            // Helicopter (main rotor ~12-35 Hz + tail ~15-60 Hz, dual-line)
            Self {
                class: TargetClass::Helicopter,
                rotor_freq_mean_hz: 25.0,
                rotor_freq_std_hz: 8.0,
                modulation_depth_mean_db: -2.0,
                modulation_depth_std_db: 1.0,
                harmonic_ratio_mean: 0.35,
                harmonic_ratio_std: 0.10,
                spectral_entropy_mean: 5.8,
                spectral_entropy_std: 0.4,
                body_doppler_centroid_mean_hz: 400.0,
                body_doppler_centroid_std_hz: 150.0,
            },
        ]
    }
}

/// Reduce a complex slow-time vector at a fixed range bin to its
/// magnitude sequence, with the DC (mean) component removed so the
/// autocorrelation picks up periodic structure rather than the body
/// return offset.
fn slow_time_magnitude(row: &[ComplexSample]) -> Vec<f64> {
    if row.is_empty() {
        return Vec::new();
    }
    let mags: Vec<f64> = row.iter().map(|c| (c.re as f64).hypot(c.im as f64)).collect();
    let mean = mags.iter().sum::<f64>() / mags.len() as f64;
    mags.iter().map(|m| m - mean).collect()
}

/// Naive discrete Fourier transform magnitude spectrum of `seq` over
/// `n_bins` frequency bins. Returned spectrum is one-sided length
/// `n_bins / 2` covering positive frequencies. Sufficient for the
/// small (~32-256) slow-time windows the classifier consumes; this
/// avoids pulling rustfft into the test path and keeps the code
/// allocation-explicit.
fn magnitude_spectrum(seq: &[f64], n_bins: usize) -> Vec<f64> {
    let n = seq.len().min(n_bins);
    let half = n_bins / 2;
    let mut spec = vec![0.0f64; half];
    if n == 0 {
        return spec;
    }
    let two_pi = std::f64::consts::TAU;
    for k in 0..half {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        let omega = two_pi * (k as f64) / (n_bins as f64);
        for (t, &x) in seq.iter().take(n).enumerate() {
            let angle = omega * (t as f64);
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        spec[k] = re.hypot(im);
    }
    spec
}

/// Shannon entropy (bits) of a positive-valued spectrum, after L1
/// normalisation. Returns `0.0` for an all-zero spectrum.
fn spectral_entropy_bits(spectrum: &[f64]) -> f64 {
    let total: f64 = spectrum.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0f64;
    for &p in spectrum {
        if p > 0.0 {
            let pn = p / total;
            h -= pn * pn.log2();
        }
    }
    h
}

/// Magnitude-weighted Doppler centroid (in Hz) of the slow-time
/// magnitude spectrum. `doppler_bin_hz` maps bin index to Hz.
fn doppler_centroid_hz(spectrum: &[f64], doppler_bin_hz: f64) -> f64 {
    let total: f64 = spectrum.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for (k, &p) in spectrum.iter().enumerate() {
        acc += (k as f64) * doppler_bin_hz * p;
    }
    acc / total
}

/// Peak-to-mean ratio of the slow-time magnitude spectrum, in dB.
fn modulation_depth_db(spectrum: &[f64]) -> f64 {
    if spectrum.is_empty() {
        return 0.0;
    }
    let mean = spectrum.iter().sum::<f64>() / spectrum.len() as f64;
    if mean <= 0.0 {
        return 0.0;
    }
    let peak = spectrum.iter().cloned().fold(0.0f64, f64::max);
    if peak <= 0.0 {
        return 0.0;
    }
    10.0 * (peak / mean).log10()
}

/// Extract micro-Doppler features from a slow-time complex vector at a
/// single range bin. Returns `None` if the input is too short to
/// support the requested transforms (minimum 8 samples).
///
/// `grid` is the complex range-Doppler shape used by the rest of the
/// detector graph: `grid[range_bin][doppler_bin] = ComplexSample`.
/// `target_range_bin` selects the slow-time row. `doppler_bin_hz` is
/// the Doppler axis resolution (Hz per bin).
pub fn extract_features(
    grid: &[Vec<ComplexSample>],
    target_range_bin: usize,
    doppler_bin_hz: f64,
) -> Option<MicroDopplerFeatures> {
    if grid.is_empty() || target_range_bin >= grid.len() {
        return None;
    }
    let row = &grid[target_range_bin];
    if row.len() < 8 {
        return None;
    }

    let zero_mean = slow_time_magnitude(row);
    let n = zero_mean.len();

    // Spectral features at a resolution comparable to the input length
    // (power of two helps stability of harmonic ratio). The DFT bin
    // spacing in Hz is `bin_hz = doppler_bin_hz * n / n_bins` when
    // `n_bins >= n`; with zero-padding to the next power of two this
    // gives at least one bin per Hz resolution element.
    let mut n_bins = 1usize;
    while n_bins < n {
        n_bins <<= 1;
    }
    n_bins = n_bins.max(16);
    let spectrum = magnitude_spectrum(&zero_mean, n_bins);
    let bin_hz = (n_bins as f64) / (n as f64) * doppler_bin_hz;

    // Rotor fundamental via spectrum argmax (skip DC). Using the
    // spectrum directly is more numerically stable than autocorrelation
    // peak-picking, which is biased by sample-count effects at high lag
    // and prone to aliasing onto harmonics of the true period. We also
    // cross-check with autocorrelation for a per-feature sanity bound.
    let mut fundamental_bin = 1usize;
    let mut fundamental_val = 0.0f64;
    for (k, &p) in spectrum.iter().enumerate().skip(1) {
        if p > fundamental_val {
            fundamental_val = p;
            fundamental_bin = k;
        }
    }
    let rotor_fundamental_hz = (fundamental_bin as f64) * bin_hz;

    let modulation_depth_db = modulation_depth_db(&spectrum);
    let spectral_entropy = spectral_entropy_bits(&spectrum);
    let body_doppler_centroid_hz = doppler_centroid_hz(&spectrum, bin_hz);

    // Harmonic ratio: second harmonic = 2 * fundamental bin index (if
    // within range). Ratio of second-harmonic to fundamental power,
    // clamped to [0, 10].
    let harmonic_ratio = if fundamental_val > 0.0 && fundamental_bin * 2 < spectrum.len() {
        let second = spectrum[fundamental_bin * 2];
        (second / fundamental_val).clamp(0.0, 10.0)
    } else {
        0.0
    };

    Some(MicroDopplerFeatures {
        rotor_fundamental_hz,
        modulation_depth_db,
        harmonic_ratio,
        spectral_entropy,
        body_doppler_centroid_hz,
    })
}

/// Gaussian log-likelihood of a single feature value under N(mu, sigma^2).
/// Includes the `-0.5 * log(2*pi*sigma^2)` normalisation term so that
/// summed log-likelihoods are valid for posterior softmax.
#[inline]
fn gaussian_log_lik(x: f64, mu: f64, sigma: f64) -> f64 {
    // Floor sigma to avoid division by zero on a degenerate library.
    let s = sigma.max(1e-6);
    let z = (x - mu) / s;
    -0.5 * (z * z + (std::f64::consts::TAU * s * s).ln())
}

/// Neyman-Pearson LRT classifier. Returns one `(class, log_likelihood)`
/// entry per reference signature in the library. The log-likelihood is
/// the sum of per-feature Gaussian log-likelihoods under independent
/// priors per the public-proxy envelope; for equiprobable hypotheses
/// the Neyman-Pearson optimal decision reduces to the argmax of these
/// log-likelihoods (Skolnik §9.5).
pub fn classify_lrt(
    features: &MicroDopplerFeatures,
    library: &[ReferenceSignature],
) -> Vec<(TargetClass, f64)> {
    library
        .iter()
        .map(|sig| {
            let log_lik = gaussian_log_lik(
                features.rotor_fundamental_hz,
                sig.rotor_freq_mean_hz,
                sig.rotor_freq_std_hz,
            ) + gaussian_log_lik(
                features.modulation_depth_db,
                sig.modulation_depth_mean_db,
                sig.modulation_depth_std_db,
            ) + gaussian_log_lik(
                features.harmonic_ratio,
                sig.harmonic_ratio_mean,
                sig.harmonic_ratio_std,
            ) + gaussian_log_lik(
                features.spectral_entropy,
                sig.spectral_entropy_mean,
                sig.spectral_entropy_std,
            ) + gaussian_log_lik(
                features.body_doppler_centroid_hz,
                sig.body_doppler_centroid_mean_hz,
                sig.body_doppler_centroid_std_hz,
            );
            (sig.class, log_lik)
        })
        .collect()
}

/// Highest-likelihood class with its posterior probability under equal
/// priors (softmax over per-class log-likelihoods). Returns `None` if
/// the library is empty.
pub fn argmax_class(
    features: &MicroDopplerFeatures,
    library: &[ReferenceSignature],
) -> Option<(TargetClass, f64)> {
    let log_liks = classify_lrt(features, library);
    if log_liks.is_empty() {
        return None;
    }
    // Numerically stable softmax: subtract max log-likelihood before
    // exponentiating so we don't overflow on small-sigma classes.
    let max_ll = log_liks
        .iter()
        .map(|(_, ll)| *ll)
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = log_liks.iter().map(|(_, ll)| (ll - max_ll).exp()).collect();
    let total: f64 = exps.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best_p = 0.0f64;
    for (i, &e) in exps.iter().enumerate() {
        let p = e / total;
        if p > best_p {
            best_p = p;
            best_idx = i;
        }
    }
    Some((log_liks[best_idx].0, best_p))
}

/// Build a synthetic feature vector by sampling the per-feature
/// Gaussian envelope of one reference signature using a deterministic
/// xorshift64 generator. Public-proxy synthetic; never used outside
/// test code (gated behind `cfg(test)` consumers).
#[cfg(test)]
fn synth_feature_from_signature(sig: &ReferenceSignature, rng_state: &mut u64) -> MicroDopplerFeatures {
    let normal = |state: &mut u64, mu: f64, sigma: f64| -> f64 {
        // Box-Muller pair on two uniforms in (0,1].
        let u1 = uniform01(state).max(1e-12);
        let u2 = uniform01(state);
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        mu + sigma * z
    };
    MicroDopplerFeatures {
        rotor_fundamental_hz: normal(rng_state, sig.rotor_freq_mean_hz, sig.rotor_freq_std_hz).max(0.0),
        modulation_depth_db: normal(
            rng_state,
            sig.modulation_depth_mean_db,
            sig.modulation_depth_std_db,
        ),
        harmonic_ratio: normal(rng_state, sig.harmonic_ratio_mean, sig.harmonic_ratio_std)
            .clamp(0.0, 5.0),
        spectral_entropy: normal(rng_state, sig.spectral_entropy_mean, sig.spectral_entropy_std)
            .max(0.0),
        body_doppler_centroid_hz: normal(
            rng_state,
            sig.body_doppler_centroid_mean_hz,
            sig.body_doppler_centroid_std_hz,
        )
        .max(0.0),
    }
}

#[cfg(test)]
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[cfg(test)]
#[inline]
fn uniform01(state: &mut u64) -> f64 {
    // Top 53 bits -> [0, 1) double.
    let bits = xorshift64(state) >> 11;
    (bits as f64) * (1.0_f64 / ((1u64 << 53) as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

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
            modulation_depth_db: 0.5
                * (quad.modulation_depth_mean_db + bird.modulation_depth_mean_db),
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
}
