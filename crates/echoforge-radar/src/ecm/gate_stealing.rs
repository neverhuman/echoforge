//! Velocity-gate stealing (Schleher chap. 4).
//!
//! The Doppler analogue of range-gate stealing. A velocity-gate stealing
//! (VGS) jammer captures the victim's Doppler tracking gate by emitting
//! a return whose Doppler frequency offset ramps linearly with time.
//! Pulse-Doppler radars typically have a narrow velocity gate (a few
//! 10s to 100s of Hz wide) and an automatic-gain-control loop that
//! favours the strongest in-gate return; once the VGS jammer's offset
//! exceeds the true target's relative Doppler, the velocity tracker
//! locks onto the jammer and drifts away from the true target.
//!
//! This module emits only the geometry — the Doppler offset (Hz) the
//! jammer imposes on the velocity tracker as a function of time. It is
//! deliberately the mirror image of [`super::deception::range_offset_at_time`]
//! so callers can compose RGS + VGS as a coordinated attack.
//!
//! # References
//!
//! - Schleher, *Electronic Warfare in the Information Age* (Artech 1999),
//!   chap. 4 — velocity-gate stealing.

/// One velocity-gate stealing jammer geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityGateStealer {
    /// Initial Doppler offset (Hz) at t = 0. Typically zero so the
    /// false return starts on top of the true target's Doppler.
    pub initial_doppler_offset_hz: f64,
    /// Doppler pull-off rate (Hz/s). Schleher chap. 4 notes that
    /// typical velocity-tracker loop bandwidths are 1-10 Hz, so 10-100
    /// Hz/s ramps are slow enough to drag the loop without exciting
    /// the discriminator yet fast enough to exit the gate inside the
    /// CPI.
    pub pull_off_rate_hz_per_s: f64,
}

/// Doppler offset (Hz) the VGS jammer imposes on the velocity tracker
/// at time `t_s`. Offset grows linearly with time:
/// `offset(t) = initial + rate · t`. No burn-through equivalent is
/// modelled here — the VGS jammer typically operates inside the CPI,
/// not on a multi-second track-lifetime time-scale; callers that need
/// a burn-through analogue can clamp the return value themselves.
pub fn doppler_offset_at_time(stealer: &VelocityGateStealer, t_s: f64) -> f64 {
    stealer.initial_doppler_offset_hz + stealer.pull_off_rate_hz_per_s * t_s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doppler_offset_at_t0_equals_initial() {
        let stealer = VelocityGateStealer {
            initial_doppler_offset_hz: 25.0,
            pull_off_rate_hz_per_s: 50.0,
        };
        assert_eq!(doppler_offset_at_time(&stealer, 0.0), 25.0);
    }

    #[test]
    fn doppler_offset_grows_linearly() {
        // At t = 5 s with 50 Hz/s pull-off the Doppler offset shifts by
        // 250 Hz from the initial; starting at 0 Hz, the helper should
        // report 250 Hz.
        let stealer = VelocityGateStealer {
            initial_doppler_offset_hz: 0.0,
            pull_off_rate_hz_per_s: 50.0,
        };
        assert_eq!(doppler_offset_at_time(&stealer, 5.0), 250.0);
    }

    #[test]
    fn doppler_offset_handles_negative_pull_off() {
        // Negative pull-off rate models a VGS jammer dragging the
        // tracker toward lower Doppler (e.g. away from an approaching
        // target). At t = 3 s with -100 Hz/s, the offset is -300 Hz.
        let stealer = VelocityGateStealer {
            initial_doppler_offset_hz: 0.0,
            pull_off_rate_hz_per_s: -100.0,
        };
        assert_eq!(doppler_offset_at_time(&stealer, 3.0), -300.0);
    }

    #[test]
    fn doppler_offset_finite_for_finite_inputs() {
        let stealer = VelocityGateStealer {
            initial_doppler_offset_hz: 10.0,
            pull_off_rate_hz_per_s: 50.0,
        };
        let o = doppler_offset_at_time(&stealer, 1.0e6);
        assert!(o.is_finite());
    }
}
