mod profile;
mod regime;
mod samplers;

pub use profile::{
    apply_clutter_to_profile, sample_clutter_frame, ClutterFrameSample, ClutterProfile,
};
pub use regime::{ClutterDistribution, ClutterRegime, TerrainClass};
pub use samplers::{
    generate_clutter_sequence, sample_clutter_amplitude, sample_k_distribution, sample_log_normal,
    sample_weibull,
};

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1u32 << 24) as f32)
    }

    /// Uniform sample in the open interval (0, 1) with 53-bit precision.
    /// Open at 0 so callers can pass the result to `ln`. Open at 1 keeps
    /// `1 - u` open at 0 for inverse-CDF samplers.
    fn open_unit_f64(&mut self) -> f64 {
        // Standard 53-bit construction: take 53 high bits, divide by 2^53.
        // Add 0.5 ULP so the smallest possible value is > 0 and the largest
        // is < 1.0; safe for inverse-CDF transforms like Weibull and log-normal.
        let bits = self.next_u64() >> 11; // 53 bits
        let denom = (1u64 << 53) as f64;
        let u = (bits as f64 + 0.5) / denom;
        // Defensive clamp; floating-point should already keep u in (0,1)
        // but we belt-and-brace so callers can rely on it.
        if u <= 0.0 {
            f64::EPSILON
        } else if u >= 1.0 {
            1.0 - f64::EPSILON
        } else {
            u
        }
    }
}

#[cfg(test)]
mod tests_a;

#[cfg(test)]
mod tests_b;
