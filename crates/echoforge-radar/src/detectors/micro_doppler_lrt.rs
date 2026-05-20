//! LRT classification functions extracted from `micro_doppler_classifier.rs`
//! to keep that file within the 350-LOC limit.

use super::micro_doppler_classifier::{MicroDopplerFeatures, ReferenceSignature, TargetClass};

/// Gaussian log-likelihood of a single feature value under N(mu, sigma^2).
/// Includes the `-0.5 * log(2*pi*sigma^2)` normalisation term so that
/// summed log-likelihoods are valid for posterior softmax.
#[inline]
pub(super) fn gaussian_log_lik(x: f64, mu: f64, sigma: f64) -> f64 {
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
