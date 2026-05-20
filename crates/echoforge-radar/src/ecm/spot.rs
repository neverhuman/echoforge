//! Spot (in-band, narrow) jamming + frequency-agility counter
//! (Skolnik §11.6, §11.10).
//!
//! A spot jammer concentrates all of its ERP into a narrow band centred
//! on the (suspected) victim radar carrier. If the radar's tuned
//! frequency falls inside the jammer's narrow band, the jamming power
//! captured by the matched filter is much higher than for a barrage
//! jammer of equal ERP — concentration in spectrum is the whole point.
//! The standard countermeasure is *frequency agility*: the radar hops
//! its carrier pulse-to-pulse (or CPI-to-CPI) so the jammer's centre
//! frequency is the wrong one most of the time.
//!
//! This module ignores the geometry / Friis link (callers compose with
//! [`super::barrage::jammer_received_power_w`]) and returns just the
//! *spectral correlation* between the jammer and the victim — i.e. the
//! fraction of jammer ERP that ends up inside the radar's matched
//! filter, expressed in dB as a positive "jamming loss" the victim
//! experiences. A loss of 0 dB means the jammer and victim are
//! perfectly spectrally separated; a loss of +30 dB means the matched
//! filter captures essentially all of the jammer ERP.
//!
//! # References
//!
//! - Skolnik, *Introduction to Radar Systems* 3rd ed. (McGraw-Hill 2001),
//!   §11.6 (spot jamming) and §11.10 (frequency agility countermeasure).

/// One spot jammer. `bandwidth_hz` is the jammer's spectral width;
/// `center_frequency_hz` is the carrier the jammer is targeting;
/// `frequency_agility` is `true` when the *jammer* itself can chase the
/// victim's hop pattern (rare — usually the jammer is fixed-frequency
/// and the radar uses agility to escape). When `frequency_agility` is
/// `true` the helper returns zero loss to model the radar dodging away.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpotJammer {
    /// Effective Radiated Power (transmit power × transmit antenna gain),
    /// in dBW. Carried for symmetry with `BarrageJammer` so the same
    /// jammer struct can be used by both helpers; the spot-loss helper
    /// itself only consumes the bandwidth-overlap geometry.
    pub erp_dbw: f64,
    /// Centre frequency of the jammer's narrow band (Hz).
    pub center_frequency_hz: f64,
    /// Bandwidth of the jammer's narrow band (Hz).
    pub bandwidth_hz: f64,
    /// When `true`, the radar successfully escapes the jammer's band
    /// via frequency agility; the helper returns zero spot-jamming loss.
    /// When `false`, the helper evaluates the overlap geometry as usual.
    pub frequency_agility: bool,
}

/// Spot-jamming loss (dB, positive) experienced by the victim. Returns
/// approximately `10·log10(B_v / B_J)` when the victim carrier sits
/// inside the jammer's narrow band (B_v ≤ B_J; the matched filter
/// captures the fraction `B_v / B_J` of the spot ERP, so the *loss*
/// the victim sees from this concentration is the reciprocal). When
/// the victim falls outside the jammer band, returns zero (no overlap).
/// When `frequency_agility` is true, returns zero unconditionally — the
/// radar successfully dodged the jammer's pre-positioned band.
///
/// The convention "loss" here is the positive dB increment a barrage-
/// equivalent helper would have to add to its J/S to match the in-band
/// spot jammer's effect. Callers that want the *signal loss* should
/// negate the return value.
pub fn spot_jamming_loss_db(
    jammer: &SpotJammer,
    victim_frequency_hz: f64,
    victim_bandwidth_hz: f64,
) -> f64 {
    if jammer.frequency_agility {
        return 0.0;
    }
    if jammer.bandwidth_hz <= 0.0 || victim_bandwidth_hz <= 0.0 {
        return 0.0;
    }
    // Check whether the victim's RF window overlaps the jammer's band.
    let v_lo = victim_frequency_hz - 0.5 * victim_bandwidth_hz;
    let v_hi = victim_frequency_hz + 0.5 * victim_bandwidth_hz;
    let j_lo = jammer.center_frequency_hz - 0.5 * jammer.bandwidth_hz;
    let j_hi = jammer.center_frequency_hz + 0.5 * jammer.bandwidth_hz;
    let overlap = (v_hi.min(j_hi) - v_lo.max(j_lo)).max(0.0);
    if overlap <= 0.0 {
        return 0.0;
    }
    // Spot jammer wins when its ERP is concentrated in a band narrower
    // than (or comparable to) the victim bandwidth. The loss the
    // victim sees, relative to a barrage jammer of equal ERP spread
    // over `B_v`, is `10·log10(B_v / B_J)` when `B_J < B_v` (positive
    // dB — concentration helps the jammer). When `B_J >= B_v` the spot
    // jammer is no better than barrage in this band; return zero.
    if jammer.bandwidth_hz >= victim_bandwidth_hz {
        return 0.0;
    }
    10.0 * (victim_bandwidth_hz / jammer.bandwidth_hz).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spot_jammer_in_band_causes_substantial_loss() {
        // Jammer 1 MHz wide on 9.5 GHz; victim is a 50 MHz matched-filter
        // X-band radar listening at 9.5 GHz. Spot jammer captures all of
        // its ERP into the victim's matched filter, but concentrates it
        // in a band 50× narrower than the victim — loss ≈ 10·log10(50)
        // = 17 dB above an equivalent-ERP barrage on the same band.
        let jammer = SpotJammer {
            erp_dbw: 40.0,
            center_frequency_hz: 9.5e9,
            bandwidth_hz: 1.0e6,
            frequency_agility: false,
        };
        let loss_db = spot_jamming_loss_db(&jammer, 9.5e9, 50.0e6);
        assert!(
            loss_db > 10.0,
            "expected > 10 dB spot jamming loss, got {loss_db}"
        );
        assert!(loss_db.is_finite(), "loss must be finite");
    }

    #[test]
    fn spot_jammer_out_of_band_zero_loss() {
        // Jammer on 9.5 GHz, victim on 10.0 GHz with 50 MHz bandwidth.
        // The victim window [9.975, 10.025] GHz does not overlap the
        // jammer [9.4995, 9.5005] GHz; loss must be zero.
        let jammer = SpotJammer {
            erp_dbw: 40.0,
            center_frequency_hz: 9.5e9,
            bandwidth_hz: 1.0e6,
            frequency_agility: false,
        };
        let loss_db = spot_jamming_loss_db(&jammer, 10.0e9, 50.0e6);
        assert_eq!(loss_db, 0.0);
    }

    #[test]
    fn frequency_agility_counter_returns_zero_loss() {
        // Even with the jammer parked exactly on the victim carrier, an
        // agility-capable victim escapes; helper returns zero loss.
        let jammer = SpotJammer {
            erp_dbw: 40.0,
            center_frequency_hz: 9.5e9,
            bandwidth_hz: 1.0e6,
            frequency_agility: true,
        };
        let loss_db = spot_jamming_loss_db(&jammer, 9.5e9, 50.0e6);
        assert_eq!(loss_db, 0.0);
    }

    #[test]
    fn spot_wider_than_victim_returns_zero() {
        // Jammer 100 MHz wide on 9.5 GHz; victim 50 MHz wide. Jammer is
        // wider — it's not really "spot" in this band, return zero (a
        // barrage-equivalent helper would already account for this).
        let jammer = SpotJammer {
            erp_dbw: 40.0,
            center_frequency_hz: 9.5e9,
            bandwidth_hz: 100.0e6,
            frequency_agility: false,
        };
        let loss_db = spot_jamming_loss_db(&jammer, 9.5e9, 50.0e6);
        assert_eq!(loss_db, 0.0);
    }

    #[test]
    fn zero_bandwidth_returns_zero() {
        let jammer = SpotJammer {
            erp_dbw: 40.0,
            center_frequency_hz: 9.5e9,
            bandwidth_hz: 0.0,
            frequency_agility: false,
        };
        assert_eq!(spot_jamming_loss_db(&jammer, 9.5e9, 50.0e6), 0.0);
    }
}
