//! Takeoff profile used by the public-proxy radar scene wrapper.

use serde::{Deserialize, Serialize};

use super::super::episode::TargetState;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TakeoffProfile {
    pub initial_range_m: f64,
    pub runway_heading_deg: f64,
    pub ground_speed_mps: f64,
    pub acceleration_mps2: f64,
    pub climb_rate_mps: f64,
    pub max_altitude_m: f64,
    pub radial_velocity_bias_mps: f64,
    pub pitch_jitter_deg: f64,
    pub yaw_jitter_deg: f64,
    pub propulsor_hz: f64,
    pub micro_doppler_hz: f64,
    pub rcs_scalar: f64,
    /// Number of rotor/propeller blades. Default `None` preserves the
    /// prior single-sinusoid micro-Doppler used by existing reproduction
    /// fixtures. When `Some(n)` (with `blade_length_m` also `Some(_)`),
    /// the synthesis loop dispatches to
    /// [`crate::micro_doppler_gen::PropellerGenerator`], which models a
    /// multi-blade rotor with blade-flash convention.
    #[serde(default)]
    pub blade_count: Option<usize>,
    /// Blade length in metres (tip radius). Default `None` falls back
    /// to the prior single-sinusoid micro-Doppler.
    #[serde(default)]
    pub blade_length_m: Option<f64>,
}

impl Default for TakeoffProfile {
    fn default() -> Self {
        Self {
            initial_range_m: 1_450.0,
            runway_heading_deg: 18.0,
            ground_speed_mps: 31.0,
            acceleration_mps2: 1.2,
            climb_rate_mps: 4.5,
            max_altitude_m: 380.0,
            radial_velocity_bias_mps: -18.0,
            pitch_jitter_deg: 1.2,
            yaw_jitter_deg: 1.7,
            propulsor_hz: 95.0,
            micro_doppler_hz: 42.0,
            rcs_scalar: 1.0,
            blade_count: None,
            blade_length_m: None,
        }
    }
}

impl TakeoffProfile {
    pub fn state_at(&self, t_s: f64) -> TargetState {
        let speed = self.ground_speed_mps + self.acceleration_mps2 * t_s;
        let along_runway_m = self.ground_speed_mps * t_s + 0.5 * self.acceleration_mps2 * t_s * t_s;
        let heading_rad = self.runway_heading_deg.to_radians();
        let cross_range_m = along_runway_m * heading_rad.sin();
        let down_range_m = self.initial_range_m + along_runway_m * heading_rad.cos();
        let altitude_m = (self.climb_rate_mps * t_s).min(self.max_altitude_m);
        let slant_range_m =
            (down_range_m * down_range_m + cross_range_m * cross_range_m + altitude_m * altitude_m)
                .sqrt();
        let radial_velocity_mps = self.radial_velocity_bias_mps + speed * heading_rad.cos() * 0.35;
        let pitch_deg = (altitude_m / self.max_altitude_m.max(1.0) * 7.0)
            + self.pitch_jitter_deg * (2.0 * std::f64::consts::PI * 0.31 * t_s).sin();
        let yaw_deg = self.yaw_jitter_deg * (2.0 * std::f64::consts::PI * 0.17 * t_s + 0.4).sin();
        let propulsor_phase_rad = 2.0 * std::f64::consts::PI * self.propulsor_hz * t_s;

        TargetState {
            time_s: t_s,
            range_m: slant_range_m,
            altitude_m,
            radial_velocity_mps,
            pitch_deg,
            yaw_deg,
            course_deg: self.runway_heading_deg,
            propulsor_phase_rad,
        }
    }
}
