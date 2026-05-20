//! Range-gate stealing / deception (Schleher chap. 3-4).
//!
//! A range-gate stealing (RGS) jammer captures the victim radar's tracking
//! loop by re-transmitting the radar pulse with a deliberately growing
//! delay. At t = 0 the false return is co-located with the true target
//! and the tracker can't tell them apart. As the jammer's delay ramps up,
//! the tracker's range gate "follows" the bigger return (the jammer) and
//! the true target falls outside the gate. After the *burn-through time*,
//! the true target return becomes stronger than the jammer (typically
//! because the radar has closed range and the jammer can no longer
//! out-power the two-way return), and the track is lost / re-acquired on
//! the true target.
//!
//! This module models the *offset* the jammer imposes on the tracker's
//! range gate as a function of time, returning `NaN` after burn-through
//! so callers can use the value as a track-loss indicator without
//! needing a parallel boolean.
//!
//! # References
//!
//! - Schleher, *Electronic Warfare in the Information Age* (Artech 1999),
//!   chap. 3 (range-gate stealing) and chap. 4 (counter-counter).

/// One range-gate stealing jammer geometry. All quantities in SI (m, s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeceptionJammer {
    /// Initial range offset (m) at t = 0 — typically zero (the false
    /// return starts co-located with the true target). Non-zero values
    /// model jammers that begin with a small offset to avoid an initial
    /// merge / coincidence detector.
    pub initial_range_offset_m: f64,
    /// Pull-off rate (m/s). Schleher chap. 3 recommends ≲ 100 m/s for
    /// typical fire-control trackers — fast enough to drag the gate
    /// inside the loop bandwidth, slow enough that the discriminator
    /// can't recognise the deception.
    pub pull_off_rate_m_per_s: f64,
    /// Burn-through time (s). After this time the helper returns NaN to
    /// signal "the true target return has overtaken the jammer; track
    /// loss / re-acquisition decision belongs to the caller". A value
    /// of `0.0` or non-positive disables burn-through (helper never
    /// returns NaN); `f64::INFINITY` likewise disables it.
    pub burn_through_time_s: f64,
}

/// Range offset (m) the jammer imposes on the tracker at time `t_s`.
/// Returns `NaN` once `t_s > burn_through_time_s` (with the disable
/// rules above). Offset grows linearly with time:
/// `offset(t) = initial + rate · t`.
pub fn range_offset_at_time(jammer: &DeceptionJammer, t_s: f64) -> f64 {
    if jammer.burn_through_time_s.is_finite()
        && jammer.burn_through_time_s > 0.0
        && t_s > jammer.burn_through_time_s
    {
        return f64::NAN;
    }
    jammer.initial_range_offset_m + jammer.pull_off_rate_m_per_s * t_s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_at_t0_equals_initial() {
        let jammer = DeceptionJammer {
            initial_range_offset_m: 5.0,
            pull_off_rate_m_per_s: 100.0,
            burn_through_time_s: 30.0,
        };
        assert_eq!(range_offset_at_time(&jammer, 0.0), 5.0);
    }

    #[test]
    fn offset_grows_linearly_until_burn_through() {
        let jammer = DeceptionJammer {
            initial_range_offset_m: 0.0,
            pull_off_rate_m_per_s: 100.0,
            burn_through_time_s: 30.0,
        };
        assert_eq!(range_offset_at_time(&jammer, 10.0), 1000.0);
        assert_eq!(range_offset_at_time(&jammer, 25.0), 2500.0);
    }

    #[test]
    fn offset_is_nan_after_burn_through() {
        let jammer = DeceptionJammer {
            initial_range_offset_m: 0.0,
            pull_off_rate_m_per_s: 100.0,
            burn_through_time_s: 30.0,
        };
        let r = range_offset_at_time(&jammer, 31.0);
        assert!(r.is_nan(), "expected NaN after burn-through, got {r}");
    }

    #[test]
    fn infinite_burn_through_disables_track_loss() {
        let jammer = DeceptionJammer {
            initial_range_offset_m: 0.0,
            pull_off_rate_m_per_s: 100.0,
            burn_through_time_s: f64::INFINITY,
        };
        let r = range_offset_at_time(&jammer, 1.0e6);
        assert!(r.is_finite() && r == 1.0e8);
    }

    #[test]
    fn zero_burn_through_disables_track_loss() {
        let jammer = DeceptionJammer {
            initial_range_offset_m: 0.0,
            pull_off_rate_m_per_s: 100.0,
            burn_through_time_s: 0.0,
        };
        let r = range_offset_at_time(&jammer, 1.0e3);
        assert!(r.is_finite() && r == 1.0e5);
    }
}
