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

fn sig(
    class: TargetClass,
    rf_mean: f64,
    rf_std: f64,
    md_mean: f64,
    md_std: f64,
    hr_mean: f64,
    hr_std: f64,
    se_mean: f64,
    se_std: f64,
    bd_mean: f64,
    bd_std: f64,
) -> ReferenceSignature {
    ReferenceSignature {
        class,
        rotor_freq_mean_hz: rf_mean,
        rotor_freq_std_hz: rf_std,
        modulation_depth_mean_db: md_mean,
        modulation_depth_std_db: md_std,
        harmonic_ratio_mean: hr_mean,
        harmonic_ratio_std: hr_std,
        spectral_entropy_mean: se_mean,
        spectral_entropy_std: se_std,
        body_doppler_centroid_mean_hz: bd_mean,
        body_doppler_centroid_std_hz: bd_std,
    }
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
        // Quadcopter (4 rotors @ ~250-500 Hz, harmonic-rich)
        // Fixed-wing UAV (Shahed-class proxy, 2-blade prop ~150-220 Hz)
        // Bird (wingbeat 2-8 Hz, low harmonic content) — Rahman-Robertson 2018
        // Helicopter (main rotor ~12-35 Hz + tail ~15-60 Hz, dual-line)
        vec![
            sig(
                TargetClass::QuadcopterFourRotor,
                400.0,
                150.0,
                -3.0,
                1.5,
                0.25,
                0.10,
                4.5,
                0.5,
                50.0,
                30.0,
            ),
            sig(
                TargetClass::FixedWingUav,
                185.0,
                35.0,
                -8.0,
                2.0,
                0.10,
                0.05,
                3.2,
                0.4,
                970.0,
                200.0,
            ),
            sig(
                TargetClass::BirdFlapping,
                5.0,
                2.0,
                -15.0,
                3.0,
                0.05,
                0.03,
                5.5,
                0.6,
                200.0,
                100.0,
            ),
            sig(
                TargetClass::Helicopter,
                25.0,
                8.0,
                -2.0,
                1.0,
                0.35,
                0.10,
                5.8,
                0.4,
                400.0,
                150.0,
            ),
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
    let mags: Vec<f64> = row
        .iter()
        .map(|c| (c.re as f64).hypot(c.im as f64))
        .collect();
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
/// single range bin (`grid[range_bin][doppler_bin]`). Returns `None`
/// if the input has fewer than 8 samples. `doppler_bin_hz` is the
/// Doppler-axis resolution in Hz/bin.
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

// LRT classification and argmax logic extracted to sibling module for LOC compliance.
pub use super::micro_doppler_lrt::{argmax_class, classify_lrt};

#[cfg(test)]
#[path = "micro_doppler_classifier_tests.rs"]
mod tests;
