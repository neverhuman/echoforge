use serde::{Deserialize, Serialize};

use super::SplitMix64;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClutterProfile {
    pub ground_clutter: f32,
    pub vegetation: f32,
    pub buildings: f32,
    pub urban_multipath: f32,
    pub roads_vehicles: f32,
    pub human_size_ground_movers: f32,
    pub power_lines: f32,
    pub turbines: f32,
    pub birds: f32,
    pub rain: f32,
    pub dust_haze: f32,
    pub terrain_only_scene: f32,
}

impl ClutterProfile {
    pub fn moderate_mixed() -> Self {
        Self {
            ground_clutter: 0.35,
            vegetation: 0.25,
            buildings: 0.2,
            urban_multipath: 0.18,
            roads_vehicles: 0.16,
            human_size_ground_movers: 0.08,
            power_lines: 0.08,
            turbines: 0.05,
            birds: 0.08,
            rain: 0.04,
            dust_haze: 0.03,
            terrain_only_scene: 0.0,
        }
    }

    pub fn bounded(self) -> Self {
        Self {
            ground_clutter: self.ground_clutter.clamp(0.0, 1.0),
            vegetation: self.vegetation.clamp(0.0, 1.0),
            buildings: self.buildings.clamp(0.0, 1.0),
            urban_multipath: self.urban_multipath.clamp(0.0, 1.0),
            roads_vehicles: self.roads_vehicles.clamp(0.0, 1.0),
            human_size_ground_movers: self.human_size_ground_movers.clamp(0.0, 1.0),
            power_lines: self.power_lines.clamp(0.0, 1.0),
            turbines: self.turbines.clamp(0.0, 1.0),
            birds: self.birds.clamp(0.0, 1.0),
            rain: self.rain.clamp(0.0, 1.0),
            dust_haze: self.dust_haze.clamp(0.0, 1.0),
            terrain_only_scene: self.terrain_only_scene.clamp(0.0, 1.0),
        }
    }

    pub fn false_alarm_pressure(self) -> f32 {
        let p = self.bounded();
        (0.20 * p.ground_clutter)
            + (0.10 * p.vegetation)
            + (0.10 * p.buildings)
            + (0.14 * p.urban_multipath)
            + (0.08 * p.roads_vehicles)
            + (0.05 * p.human_size_ground_movers)
            + (0.06 * p.power_lines)
            + (0.12 * p.turbines)
            + (0.07 * p.birds)
            + (0.04 * p.rain)
            + (0.02 * p.dust_haze)
            + (0.02 * p.terrain_only_scene)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClutterFrameSample {
    pub amplitude_offset: f32,
    pub doppler_spread_hz: f32,
    pub false_alarm_pressure: f32,
}

pub fn sample_clutter_frame(
    profile: ClutterProfile,
    seed: u64,
    frame_index: usize,
) -> ClutterFrameSample {
    let profile = profile.bounded();
    let mut rng = SplitMix64::new(seed ^ (frame_index as u64).wrapping_mul(0x517c_c1b7_2722_0a95));
    let pressure = profile.false_alarm_pressure().clamp(0.0, 1.0);
    let terrain = profile.ground_clutter
        + 0.7 * profile.vegetation
        + 0.9 * profile.buildings
        + 0.8 * profile.terrain_only_scene;
    let movers = profile.roads_vehicles
        + profile.human_size_ground_movers
        + profile.birds
        + 1.4 * profile.turbines;
    ClutterFrameSample {
        amplitude_offset: (0.02 + 0.18 * terrain + 0.05 * rng.unit_f32()).clamp(0.0, 0.8),
        doppler_spread_hz: (1.5 + 42.0 * movers + 18.0 * profile.rain + 8.0 * rng.unit_f32())
            .clamp(0.0, 180.0),
        false_alarm_pressure: pressure,
    }
}

pub fn apply_clutter_to_profile(power: &mut [f32], profile: ClutterProfile, seed: u64) {
    if power.is_empty() {
        return;
    }
    let profile = profile.bounded();
    let mut rng = SplitMix64::new(seed);
    let pressure = profile.false_alarm_pressure().clamp(0.0, 1.0);
    let glint_count = ((profile.buildings + profile.power_lines + profile.turbines) * 12.0)
        .round()
        .clamp(0.0, 32.0) as usize;

    let mut correlated = 0.0f32;
    for value in power.iter_mut() {
        correlated = 0.92 * correlated + 0.08 * (rng.unit_f32() - 0.5);
        *value = (*value + pressure * 0.04 + correlated * profile.ground_clutter).max(0.0);
    }

    for _ in 0..glint_count {
        let idx = ((rng.unit_f32() * power.len() as f32) as usize).min(power.len() - 1);
        power[idx] += 0.08 + 0.35 * pressure * rng.unit_f32();
    }
}
