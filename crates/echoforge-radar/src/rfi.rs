use serde::{Deserialize, Serialize};

use crate::prng::SplitMix64;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RfiProfile {
    pub burst_probability: f32,
    pub burst_amplitude: f32,
    pub narrowband_cw_power: f32,
    pub cochannel_emitters: u32,
    pub sidelobe_pressure: f32,
}

impl RfiProfile {
    pub fn contested_low_altitude() -> Self {
        Self {
            burst_probability: 0.015,
            burst_amplitude: 0.8,
            narrowband_cw_power: 0.08,
            cochannel_emitters: 2,
            sidelobe_pressure: 0.08,
        }
    }

    pub fn bounded(self) -> Self {
        Self {
            burst_probability: self.burst_probability.clamp(0.0, 1.0),
            burst_amplitude: self.burst_amplitude.clamp(0.0, 8.0),
            narrowband_cw_power: self.narrowband_cw_power.clamp(0.0, 1.0),
            cochannel_emitters: self.cochannel_emitters.min(64),
            sidelobe_pressure: self.sidelobe_pressure.clamp(0.0, 1.0),
        }
    }

    pub fn pressure(self) -> f32 {
        let p = self.bounded();
        (p.burst_probability * 2.0
            + p.narrowband_cw_power
            + 0.015 * p.cochannel_emitters as f32
            + p.sidelobe_pressure)
            .clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RfiFrameSample {
    pub burst_active: bool,
    pub narrowband_bin: usize,
    pub pressure: f32,
}

pub fn sample_rfi_frame(
    profile: RfiProfile,
    seed: u64,
    frame_index: usize,
    bin_count: usize,
) -> RfiFrameSample {
    let profile = profile.bounded();
    let mut rng = SplitMix64::new(seed ^ (frame_index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03));
    let burst_active = rng.unit_f32() < profile.burst_probability;
    let narrowband_bin = if bin_count == 0 {
        0
    } else {
        ((rng.unit_f32() * bin_count as f32) as usize).min(bin_count - 1)
    };
    RfiFrameSample {
        burst_active,
        narrowband_bin,
        pressure: profile.pressure(),
    }
}

pub fn apply_rfi_to_profile(power: &mut [f32], profile: RfiProfile, seed: u64) {
    if power.is_empty() {
        return;
    }
    let profile = profile.bounded();
    let mut rng = SplitMix64::new(seed);
    let cw_bin = ((rng.unit_f32() * power.len() as f32) as usize).min(power.len() - 1);
    power[cw_bin] += profile.narrowband_cw_power;

    power.iter_mut().for_each(|value| {
        let burst = rng.unit_f32() < profile.burst_probability;
        if burst {
            *value += profile.burst_amplitude * (0.4 + rng.unit_f32());
        }
        *value += profile.sidelobe_pressure * 0.015;
    });
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfi_sampling_is_deterministic_and_bounded() {
        let profile = RfiProfile::contested_low_altitude();
        let a = sample_rfi_frame(profile, 12, 3, 64);
        let b = sample_rfi_frame(profile, 12, 3, 64);
        assert_eq!(a, b);
        assert!(a.narrowband_bin < 64);
        assert!((0.0..=1.0).contains(&a.pressure));
    }

    #[test]
    fn rfi_profile_application_is_finite() {
        let mut a = vec![0.0f32; 32];
        let mut b = vec![0.0f32; 32];
        apply_rfi_to_profile(&mut a, RfiProfile::contested_low_altitude(), 4);
        apply_rfi_to_profile(&mut b, RfiProfile::contested_low_altitude(), 4);
        assert_eq!(a, b);
        assert!(a.iter().all(|v| v.is_finite() && *v >= 0.0));
    }
}
