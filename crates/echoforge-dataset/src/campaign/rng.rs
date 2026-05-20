use std::time::Instant;

pub(super) use crate::rng::{child_seed, deterministic_shuffle, SplitMix64};

/// Return nanoseconds elapsed since `start`, saturating to u64::MAX.
pub(super) fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_seed_is_deterministic_and_differs_per_index() {
        let root = 0xDEAD_BEEF_1234_5678u64;
        let s0 = child_seed(root, 0);
        let s1 = child_seed(root, 1);
        // Deterministic
        assert_eq!(child_seed(root, 0), s0);
        assert_eq!(child_seed(root, 1), s1);
        // Per-index seeds differ
        assert_ne!(s0, s1);
    }
}
