use std::time::Instant;

pub(super) use crate::rng::SplitMix64;

/// Derive a child seed from a root seed and a per-record index using finalizer mixing.
pub(super) fn child_seed(root: u64, index: u64) -> u64 {
    let mut value = root ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Fisher-Yates shuffle using SplitMix64.
pub(super) fn deterministic_shuffle<T>(items: &mut [T], seed: u64) {
    let mut rng = SplitMix64::new(seed ^ 0x5368_7566_666c_65);
    for index in (1..items.len()).rev() {
        let swap = rng.range_usize(0, index);
        items.swap(index, swap);
    }
}

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
