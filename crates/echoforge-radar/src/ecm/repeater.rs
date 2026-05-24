//! Repeater / DRFM (Digital Radio-Frequency Memory) jammer model
//! (Schleher chap. 4).
//!
//! A repeater jammer captures the victim radar's pulse, holds it briefly
//! in digital memory, and re-transmits one or more delayed copies. Each
//! copy looks to the victim like a real target return at a range offset
//! of `c · τ / 2`, where `τ` is the round-trip-equivalent delay (the
//! one-way travel time of the spoofed return). A DRFM-equipped jammer
//! can produce dozens of independent ghost targets, each at a different
//! range; the victim's CFAR / detection-graph sees them all as real
//! targets unless a downstream classifier identifies the deception.
//!
//! This helper models the *range offsets* of `ghost_count` evenly-spaced
//! ghosts at multiples of `delay_s`. The first ghost is at delay
//! `delay_s` (1× delay → c·delay/2 range), the second at `2·delay_s`,
//! and so on. ERP / signal-strength modelling is left to the caller —
//! at this level we only emit geometry.
//!
//! # References
//!
//! - Schleher, *Electronic Warfare in the Information Age* (Artech 1999),
//!   chap. 4 — DRFM repeater jammers and the analytical / digital
//!   counter-counter techniques.

use crate::propagation::SPEED_OF_LIGHT_M_PER_S;

/// One repeater / DRFM jammer. ERP-style gain is captured in `gain_db`
/// for symmetry with [`super::barrage::BarrageJammer`]; the helper
/// itself only consumes `delay_s` and `ghost_count`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepeaterJammer {
    /// Repeater amplifier gain (dB) applied to the captured pulse before
    /// re-transmission. Carried for downstream link-budget composition;
    /// the geometry helper itself does not consume it.
    pub gain_db: f64,
    /// Per-ghost incremental delay (s). The `n`-th ghost is delayed by
    /// `(n + 1) · delay_s` (so ghost 0 is at delay `delay_s`, not zero,
    /// to avoid a coincidence detection with the true target).
    pub delay_s: f64,
    /// Number of ghost targets to emit. `0` returns an empty Vec.
    pub ghost_count: u32,
}

/// Range offsets (m) of the spoofed ghosts. Each ghost's range offset
/// is `c · τ / 2` where `τ = (idx + 1) · delay_s` for `idx` in
/// `0..ghost_count`. Returns an empty Vec when `ghost_count == 0` or
/// `delay_s <= 0.0`.
pub fn ghost_range_offsets_m(jammer: &RepeaterJammer) -> Vec<f64> {
    if jammer.ghost_count == 0 || jammer.delay_s <= 0.0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(jammer.ghost_count as usize);
    for idx in 0..jammer.ghost_count {
        let tau = jammer.delay_s * (idx as f64 + 1.0);
        out.push(0.5 * SPEED_OF_LIGHT_M_PER_S * tau);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_ghosts_at_100ns_produce_15_30_45_metres() {
        // Speed of light c ≈ 2.998e8 m/s. One-way range offset for a
        // 100 ns round-trip delay is c · 100e-9 / 2 ≈ 14.99 m. With
        // three ghosts at multiples of 100 ns we expect ~ 15, 30, 45 m.
        let jammer = RepeaterJammer {
            gain_db: 30.0,
            delay_s: 100e-9,
            ghost_count: 3,
        };
        let offsets = ghost_range_offsets_m(&jammer);
        assert_eq!(offsets.len(), 3);
        let expected = [14.99, 29.98, 44.97];
        for (got, want) in offsets.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 0.10,
                "ghost offset {got} m should be ≈ {want} m"
            );
        }
    }

    #[test]
    fn zero_ghost_count_returns_empty() {
        let jammer = RepeaterJammer {
            gain_db: 30.0,
            delay_s: 100e-9,
            ghost_count: 0,
        };
        assert!(ghost_range_offsets_m(&jammer).is_empty());
    }

    #[test]
    fn nonpositive_delay_returns_empty() {
        let jammer = RepeaterJammer {
            gain_db: 30.0,
            delay_s: 0.0,
            ghost_count: 5,
        };
        assert!(ghost_range_offsets_m(&jammer).is_empty());
    }

    #[test]
    fn offsets_are_monotonically_increasing() {
        let jammer = RepeaterJammer {
            gain_db: 30.0,
            delay_s: 50e-9,
            ghost_count: 8,
        };
        let offsets = ghost_range_offsets_m(&jammer);
        assert_eq!(offsets.len(), 8);
        for w in offsets.windows(2) {
            assert!(w[1] > w[0], "offsets must be monotonic: {offsets:?}");
        }
    }
}
