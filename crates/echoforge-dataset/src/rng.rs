/// Shared SplitMix64 PRNG used across campaign, ml_training, and helpers.
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64 {
    pub state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub(crate) fn unit_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) / ((1u64 << 53) as f64)
    }

    pub(crate) fn unit_f32(&mut self) -> f32 {
        self.unit_f64() as f32
    }

    pub(crate) fn range_f64(&mut self, min: f64, max: f64) -> f64 {
        min + self.unit_f64() * (max - min)
    }

    pub(crate) fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + self.unit_f32() * (max - min)
    }

    pub(crate) fn range_usize(&mut self, min: usize, max: usize) -> usize {
        if max <= min {
            return min;
        }
        min + (self.next_u64() as usize % (max - min + 1))
    }
}
