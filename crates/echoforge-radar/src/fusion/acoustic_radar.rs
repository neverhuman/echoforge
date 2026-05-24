//! Acoustic-radar Bayesian fusion. Per RUSI 2025 + open reporting on
//! Ukraine's Sky Fortress (9,500-14,000 sensor network) + Zvook,
//! acoustic networks cover the radar low-altitude gap by detecting
//! distinctive piston-engine signature ("lawnmower buzz"). This module
//! fuses a radar track with an acoustic-bearing observation to produce
//! a posterior with combined confidence.
//!
//! Reference: Stone & Streit, *Bayesian Multiple Target Tracking*,
//! 2nd ed., Artech House 2018, ch. 4 (Bayes recursion) + ch. 7
//! (multi-sensor data association). RUSI 2025 commentary on Ukrainian
//! operational practice describes the Sky Fortress / Zvook acoustic
//! networks as the dominant low-altitude C-UAS cueing layer.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadarTrack {
    pub range_m: f64,
    pub range_uncertainty_m: f64,
    pub doppler_hz: f64,
    pub doppler_uncertainty_hz: f64,
    pub azimuth_deg: f64, // sensor frame
    pub azimuth_uncertainty_deg: f64,
    pub confidence: f64, // [0, 1] from CFAR/MTD/3-tier
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticObservation {
    pub bearing_deg: f64, // sensor frame
    pub bearing_uncertainty_deg: f64,
    pub snr_db: f64,             // observed SNR of acoustic detector
    pub source_class_prior: f64, // P(target_class | acoustic features) [0, 1]
    // e.g., piston-engine class = 0.7-0.9 per open-source reporting
    pub timestamp_diff_s: f64, // time since radar track update (for expiry weighting)
}

/// Fusion result. Embeds the original [`RadarTrack`] (range and Doppler
/// are unchanged by acoustic fusion) and adds the fusion-specific outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusedTrack {
    /// Original radar measurement; range/doppler are not updated by acoustic fusion.
    pub radar: RadarTrack,
    /// Gaussian-product combined azimuth (degrees, sensor frame).
    pub azimuth_deg: f64,
    /// Combined azimuth standard deviation (degrees).
    pub azimuth_uncertainty_deg: f64,
    /// Fused confidence in [0, 1]; combines radar and acoustic log-odds.
    pub confidence: f64,
    /// Bearing agreement score in [0, 1]; Gaussian kernel on bearing error.
    pub agreement_score: f64,
    /// P(target_class | radar + acoustic).
    pub class_posterior: f64,
}

/// Combine two independent 1-D Gaussian estimates (mean, variance) into
/// their product distribution. Returns (combined_variance, combined_mean).
/// If either variance is zero the non-degenerate sensor dominates; if
/// both are zero the first sensor is returned as-is.
fn gaussian_product_1d(mu1: f64, var1: f64, mu2: f64, var2: f64) -> (f64, f64) {
    match (var1 > 0.0, var2 > 0.0) {
        (true, true) => {
            let v = 1.0 / (1.0 / var1 + 1.0 / var2);
            (v, v * (mu1 / var1 + mu2 / var2))
        }
        (false, _) => (0.0, mu1),
        (true, false) => (0.0, mu2),
    }
}

/// Fuse one radar track with one acoustic observation. Bayes recursion
/// per Stone-Streit 2018 ch. 4 eq. (4.2):
///   P(state | radar, acoustic) ∝ P(acoustic | state) · P(radar | state) · P(state)
///
/// We assume Gaussian observation models on azimuth for both sensors.
/// Confidence fusion: log-odds addition with an expiry decay per
/// timestamp_diff_s (acoustic ages out with 10 s half-life).
pub fn fuse_acoustic(radar: &RadarTrack, acoustic: &AcousticObservation) -> FusedTrack {
    // 1. Azimuth Gaussian product.
    let radar_var = radar.azimuth_uncertainty_deg.powi(2);
    let acoustic_var = acoustic.bearing_uncertainty_deg.powi(2);
    let (combined_var, combined_az) = gaussian_product_1d(
        radar.azimuth_deg,
        radar_var,
        acoustic.bearing_deg,
        acoustic_var,
    );
    let combined_std = combined_var.sqrt();

    // 2. Agreement: Gaussian kernel on bearing error. The f64::MIN_POSITIVE
    // epsilon avoids a division guard — exp(-inf) = 0 for nonzero diff when
    // both variances are zero, and exp(0) = 1 when bearings agree exactly.
    let bearing_diff = (radar.azimuth_deg - acoustic.bearing_deg).abs();
    let pooled_var = radar_var + acoustic_var;
    let agreement = (-bearing_diff.powi(2) / (2.0 * pooled_var + f64::MIN_POSITIVE)).exp();

    // 3. Confidence fusion via log-odds addition with expiry decay.
    // Half-life is 10 s (matches the Sky Fortress / Zvook acoustic network
    // near-real-time forwarding latency from open-source reporting).
    //
    // The acoustic update is split into two terms scaled independently
    // by `expiry_weight` so that an aged observation contributes
    // nothing (rather than a negative update from the tiny clamped
    // weight). Per Stone-Streit ch. 7, a sensor with no recent
    // observation provides an uninformative likelihood, i.e. zero
    // log-odds contribution.
    let expiry_weight = (-acoustic.timestamp_diff_s.abs() / 10.0).exp();
    let prior_clamped = acoustic.source_class_prior.clamp(1e-6, 1.0 - 1e-6);
    let radar_clamped = radar.confidence.clamp(1e-6, 1.0 - 1e-6);
    let radar_logodds = (radar_clamped / (1.0 - radar_clamped)).ln();
    // Raw acoustic log-odds before expiry scaling. Uses the
    // source-class prior as the per-observation likelihood ratio (a
    // strong piston-engine prior > 0.5 yields positive log-odds).
    let raw_acoustic_logodds = (prior_clamped / (1.0 - prior_clamped)).ln();

    // Agreement gates the *sign* of the acoustic update: a sensor that
    // strongly disagrees gives a negative log-odds penalty whose
    // magnitude grows with (1-agreement) and with the raw evidence
    // strength. Expiry scales the whole acoustic contribution down
    // toward 0 (uninformative) rather than letting it pull negative.
    let agreement_term = raw_acoustic_logodds * agreement;
    let disagreement_term = -raw_acoustic_logodds.abs() * (1.0 - agreement);
    let acoustic_contribution = expiry_weight * (agreement_term + disagreement_term);
    let fused_logodds = radar_logodds + acoustic_contribution;
    let fused_conf = 1.0 / (1.0 + (-fused_logodds).exp());

    // 4. Class posterior: blends fused confidence with the acoustic
    // class prior under a uniform-prior assumption on the target class.
    let class_posterior = fused_conf * (1.0 + acoustic.source_class_prior) / 2.0;

    FusedTrack {
        radar: *radar,
        azimuth_deg: combined_az,
        azimuth_uncertainty_deg: combined_std,
        confidence: fused_conf,
        agreement_score: agreement,
        class_posterior,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn radar_baseline() -> RadarTrack {
        RadarTrack {
            range_m: 5000.0,
            range_uncertainty_m: 50.0,
            doppler_hz: 970.0,
            doppler_uncertainty_hz: 30.0,
            azimuth_deg: 100.0,
            azimuth_uncertainty_deg: 2.0,
            confidence: 0.7,
        }
    }

    fn acoustic_obs(
        bearing_deg: f64,
        source_class_prior: f64,
        timestamp_diff_s: f64,
    ) -> AcousticObservation {
        AcousticObservation {
            bearing_deg,
            bearing_uncertainty_deg: 5.0,
            snr_db: 8.0,
            source_class_prior,
            timestamp_diff_s,
        }
    }

    #[test]
    fn fusion_agreement_raises_confidence() {
        // Radar conf 0.7 + acoustic bearing 100° (agrees with radar) →
        // fused conf > 0.7 and agreement_score near 1.0.
        let radar = radar_baseline();
        let acoustic = acoustic_obs(100.5, 0.8, 0.5);
        let fused = fuse_acoustic(&radar, &acoustic);
        assert!(
            fused.confidence > 0.7,
            "expected fused conf > radar conf; got {}",
            fused.confidence
        );
        assert!(
            fused.agreement_score > 0.9,
            "expected agreement > 0.9 for 0.5° offset within pooled sigma; got {}",
            fused.agreement_score
        );
    }

    #[test]
    fn fusion_disagreement_lowers_confidence() {
        // Radar bearing 100°, acoustic bearing 120° (well outside pooled
        // sigma) → agreement near 0, disagreement penalty drops fused
        // conf below radar baseline.
        let radar = radar_baseline();
        let acoustic = acoustic_obs(120.0, 0.8, 0.5);
        let fused = fuse_acoustic(&radar, &acoustic);
        assert!(
            fused.agreement_score < 0.05,
            "expected near-zero agreement for 20° offset; got {}",
            fused.agreement_score
        );
        assert!(
            fused.confidence < radar.confidence,
            "expected disagreement to lower confidence; got {} (was {})",
            fused.confidence,
            radar.confidence
        );
    }

    #[test]
    fn fusion_aged_acoustic_decays_weight() {
        // Acoustic with timestamp_diff_s = 30 s → expiry weight
        // exp(-3) ≈ 0.05 → acoustic weight tiny → fused conf close to
        // radar conf alone (within 0.10).
        let radar = radar_baseline();
        let fresh = acoustic_obs(100.5, 0.8, 0.0);
        let aged = AcousticObservation {
            timestamp_diff_s: 30.0,
            ..fresh
        };
        let fused_fresh = fuse_acoustic(&radar, &fresh);
        let fused_aged = fuse_acoustic(&radar, &aged);
        assert!(
            (fused_aged.confidence - radar.confidence).abs() < 0.10,
            "aged fused conf {} should be near radar conf {}",
            fused_aged.confidence,
            radar.confidence
        );
        assert!(
            fused_fresh.confidence > fused_aged.confidence,
            "fresh fused conf {} should exceed aged fused conf {}",
            fused_fresh.confidence,
            fused_aged.confidence
        );
    }

    #[test]
    fn fusion_azimuth_combines_via_gaussian_product() {
        // Combined uncertainty < each sensor's uncertainty alone.
        // For sigma_r = 2°, sigma_a = 5°, sigma_combined = 1/sqrt(1/4 + 1/25)
        // = 1/sqrt(0.29) ≈ 1.857°.
        let radar = radar_baseline();
        let acoustic = acoustic_obs(102.0, 0.8, 0.5);
        let fused = fuse_acoustic(&radar, &acoustic);
        assert!(
            fused.azimuth_uncertainty_deg < radar.azimuth_uncertainty_deg,
            "combined std {} should be < radar std {}",
            fused.azimuth_uncertainty_deg,
            radar.azimuth_uncertainty_deg
        );
        assert!(
            fused.azimuth_uncertainty_deg < acoustic.bearing_uncertainty_deg,
            "combined std {} should be < acoustic std {}",
            fused.azimuth_uncertainty_deg,
            acoustic.bearing_uncertainty_deg
        );
        // Expected ≈ 1.857°; allow 0.05° tolerance.
        let expected = 1.0_f64 / (1.0_f64 / 4.0 + 1.0_f64 / 25.0).sqrt();
        assert!(
            (fused.azimuth_uncertainty_deg - expected).abs() < 0.05,
            "combined std {} should be ≈ {}",
            fused.azimuth_uncertainty_deg,
            expected
        );
        // Posterior mean is closer to the lower-variance sensor (radar).
        assert!(
            fused.azimuth_deg < (radar.azimuth_deg + acoustic.bearing_deg) / 2.0,
            "posterior mean should be pulled toward radar bearing"
        );
    }

    #[test]
    fn fusion_class_posterior_blends_radar_and_acoustic() {
        // class_posterior = fused_conf · (1 + prior) / 2; sanity check
        // it's bounded in [0, 1].
        let radar = radar_baseline();
        let acoustic = acoustic_obs(100.0, 0.9, 0.0);
        let fused = fuse_acoustic(&radar, &acoustic);
        assert!(
            (0.0..=1.0).contains(&fused.class_posterior),
            "class_posterior {} must be in [0,1]",
            fused.class_posterior
        );
        // With a strong prior and agreement, posterior should exceed
        // the radar-alone confidence.
        assert!(fused.class_posterior > radar.confidence * 0.9);
    }
}
