//! Engine configuration, built-in scenarios, and control commands.

use std::path::PathBuf;

use echoforge_radar::{
    ClutterRegime, NoiseProfile, RadarSimConfig, SceneDescriptor, TakeoffProfile, TargetClass,
    TargetEntity, TargetKinematics, TerrainClass,
};

/// Static engine settings, sourced from `StudioConfig` / environment.
#[derive(Debug, Clone)]
pub struct SimSettings {
    pub default_frame_rate_hz: f64,
    pub broadcast_capacity: usize,
    pub rd_range_bins: usize,
    pub spectrogram_bins: usize,
    pub default_scenario: String,
    pub autostart: bool,
}

impl Default for SimSettings {
    fn default() -> Self {
        Self {
            default_frame_rate_hz: 15.0,
            broadcast_capacity: 64,
            rd_range_bins: 256,
            spectrogram_bins: 256,
            default_scenario: "shahed-ingress".to_string(),
            autostart: true,
        }
    }
}

/// One scene entity plus the display-only bearing the radar console
/// plots it at. The underlying single-beam range-Doppler physics is
/// azimuth-free; bearings model an electronically-scanned (AESA)
/// counter-UAS radar that revisits every track each dwell.
#[derive(Debug, Clone)]
pub struct ScenarioEntity {
    pub class: TargetClass,
    pub kinematics: TargetKinematics,
    pub bearing_deg: f64,
    /// Initial slant range (m) used to resolve kinematics variants that
    /// do not carry their own range (Bird, Helicopter).
    pub display_initial_range_m: f64,
    pub label: String,
}

/// A fully-resolved, runnable scenario.
#[derive(Debug, Clone)]
pub struct ScenarioSpec {
    pub id: String,
    pub label: String,
    pub description: String,
    pub config: RadarSimConfig,
    pub noise: NoiseProfile,
    pub entities: Vec<ScenarioEntity>,
    pub base_seed: u64,
}

impl ScenarioSpec {
    /// Build the `SceneDescriptor` consumed by `synthesize_scene`.
    pub fn scene(&self) -> SceneDescriptor {
        let targets: Vec<TargetEntity> = self
            .entities
            .iter()
            .map(|e| TargetEntity {
                class: e.class.clone(),
                kinematics: e.kinematics.clone(),
                spawn_time_s: 0.0,
            })
            .collect();
        SceneDescriptor::from_radar_config(&self.config, &self.noise, targets)
    }
}

fn shahed_profile(initial_range_m: f64) -> TakeoffProfile {
    TakeoffProfile {
        initial_range_m,
        runway_heading_deg: 14.0,
        ground_speed_mps: 38.0,
        acceleration_mps2: 0.6,
        climb_rate_mps: 4.0,
        max_altitude_m: 900.0,
        radial_velocity_bias_mps: -22.0,
        pitch_jitter_deg: 1.0,
        yaw_jitter_deg: 1.4,
        propulsor_hz: 95.0,
        micro_doppler_hz: 44.0,
        rcs_scalar: 0.1,
        blade_count: Some(2),
        blade_length_m: Some(0.62),
    }
}

/// The three built-in scenarios offered by the studio.
pub fn builtin_scenarios() -> Vec<ScenarioSpec> {
    vec![
        ScenarioSpec {
            id: "shahed-ingress".to_string(),
            label: "Shahed-class ingress".to_string(),
            description: "Single piston Shahed-class UAV on a climbing \
                ingress run, two-blade pusher propeller micro-Doppler active."
                .to_string(),
            config: RadarSimConfig::default(),
            noise: NoiseProfile::real_world_proxy_v1(),
            entities: vec![ScenarioEntity {
                class: TargetClass::ShahedClassPiston,
                kinematics: TargetKinematics::FromTakeoffProfile(shahed_profile(7400.0)),
                bearing_deg: 62.0,
                display_initial_range_m: 7400.0,
                label: "Shahed-class piston".to_string(),
            }],
            base_seed: 0x5EED_1A11,
        },
        ScenarioSpec {
            id: "multi-confuser".to_string(),
            label: "Multi-target with confusers".to_string(),
            description: "One Shahed-class UAV against a bird, a ground \
                vehicle, and a helicopter — exercises multi-class dispatch \
                and classification."
                .to_string(),
            config: RadarSimConfig::default(),
            noise: NoiseProfile::real_world_proxy_v1(),
            entities: vec![
                ScenarioEntity {
                    class: TargetClass::ShahedClassPiston,
                    kinematics: TargetKinematics::FromTakeoffProfile(shahed_profile(6200.0)),
                    bearing_deg: 48.0,
                    display_initial_range_m: 6200.0,
                    label: "Shahed-class piston".to_string(),
                },
                ScenarioEntity {
                    class: TargetClass::Bird,
                    kinematics: TargetKinematics::Bird {
                        cruise_speed_mps: 14.0,
                        altitude_agl_m: 180.0,
                        heading_deg: 35.0,
                        wingbeat_hz: 4.6,
                        wing_length_m: 0.7,
                    },
                    bearing_deg: 138.0,
                    display_initial_range_m: 3100.0,
                    label: "Large bird".to_string(),
                },
                ScenarioEntity {
                    class: TargetClass::GroundVehicle,
                    kinematics: TargetKinematics::GroundVehicle {
                        speed_mps: 22.0,
                        heading_deg: 80.0,
                        initial_range_m: 4500.0,
                    },
                    bearing_deg: 305.0,
                    display_initial_range_m: 4500.0,
                    label: "Ground vehicle".to_string(),
                },
                ScenarioEntity {
                    class: TargetClass::Helicopter,
                    kinematics: TargetKinematics::Helicopter {
                        cruise_speed_mps: 58.0,
                        altitude_agl_m: 420.0,
                        heading_deg: 20.0,
                        main_blade_count: 4,
                        main_rotation_hz: 5.4,
                        main_blade_length_m: 6.7,
                        tail_blade_count: 3,
                        tail_rotation_hz: 28.0,
                        tail_blade_length_m: 1.3,
                    },
                    bearing_deg: 226.0,
                    display_initial_range_m: 9800.0,
                    label: "Helicopter".to_string(),
                },
            ],
            base_seed: 0x5EED_2C0F,
        },
        ScenarioSpec {
            id: "coastal-clutter".to_string(),
            label: "Coastal sea clutter".to_string(),
            description: "Shahed-class UAV ingressing over the sea — \
                K-distributed sea clutter stresses the CFAR detector."
                .to_string(),
            config: RadarSimConfig {
                ground_reflection_coefficient_magnitude: 0.6,
                ..RadarSimConfig::default()
            },
            noise: NoiseProfile {
                clutter_regime: Some(ClutterRegime::for_terrain(TerrainClass::Sea, 1.0)),
                clutter_sigma_0_scale: 1.0,
                ..NoiseProfile::real_world_proxy_v1()
            },
            entities: vec![ScenarioEntity {
                class: TargetClass::ShahedClassPiston,
                kinematics: TargetKinematics::FromTakeoffProfile(shahed_profile(5200.0)),
                bearing_deg: 287.0,
                display_initial_range_m: 5200.0,
                label: "Shahed-class piston".to_string(),
            }],
            base_seed: 0x5EED_C047,
        },
    ]
}

/// Look a scenario up by id, falling back to the first built-in.
pub fn resolve_scenario(id: &str) -> ScenarioSpec {
    let all = builtin_scenarios();
    all.iter()
        .find(|s| s.id == id)
        .cloned()
        .unwrap_or_else(|| all.into_iter().next().expect("at least one scenario"))
}

/// Sparse patch of operator-tunable radar parameters. Every field is
/// optional; `None` leaves the current value untouched.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct RadarParamPatch {
    pub transmit_power_w: Option<f64>,
    pub tx_gain_dbi: Option<f64>,
    pub rx_gain_dbi: Option<f64>,
    pub noise_figure_db: Option<f64>,
    pub cfar_pfa: Option<f32>,
    pub cfar_training_cells: Option<usize>,
    pub cfar_guard_cells: Option<usize>,
    pub rain_rate_mm_per_h: Option<f64>,
    pub atmospheric_one_way_db_per_km: Option<f64>,
}

impl RadarParamPatch {
    /// Overlay the `Some(_)` fields onto a config, clamping to safe ranges.
    pub fn apply_to(&self, cfg: &mut RadarSimConfig) {
        if let Some(v) = self.transmit_power_w {
            cfg.transmit_power_w = v.max(0.0);
        }
        if let Some(v) = self.tx_gain_dbi {
            cfg.tx_gain_dbi = v;
        }
        if let Some(v) = self.rx_gain_dbi {
            cfg.rx_gain_dbi = v;
        }
        if let Some(v) = self.noise_figure_db {
            cfg.noise_figure_db = v.max(0.0);
        }
        if let Some(v) = self.cfar_pfa {
            cfg.cfar_pfa = v.clamp(1e-9, 0.5);
        }
        if let Some(v) = self.cfar_training_cells {
            cfg.cfar_training_cells = v.clamp(2, 128);
        }
        if let Some(v) = self.cfar_guard_cells {
            cfg.cfar_guard_cells = v.clamp(0, 32);
        }
        if let Some(v) = self.rain_rate_mm_per_h {
            cfg.rain_rate_mm_per_h = v.max(0.0);
        }
        if let Some(v) = self.atmospheric_one_way_db_per_km {
            cfg.atmospheric_one_way_db_per_km = v.max(0.0);
        }
    }
}

/// Replay or live source selection for a `Start` command.
#[derive(Debug, Clone)]
pub enum SimMode {
    Live,
    Replay { bundle_path: PathBuf },
}

/// A command applied to the running [`super::engine::SimEngine`].
#[derive(Debug, Clone)]
pub enum ControlCommand {
    Start { scenario_id: String, mode: SimMode },
    Stop,
    Pause,
    Resume,
    SetSpeed(f64),
    SetParams(RadarParamPatch),
}

/// A snapshot of engine state, surfaced over `/api/sim/status` and used
/// to build [`super::frames::SessionInfo`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineStatus {
    pub session_id: u64,
    pub source: String,
    pub scenario_id: String,
    pub scenario_label: String,
    pub running: bool,
    pub paused: bool,
    pub frame_rate_hz: f64,
    pub playback_speed: f64,
}

impl EngineStatus {
    pub fn idle(frame_rate_hz: f64) -> Self {
        Self {
            session_id: 0,
            source: "idle".to_string(),
            scenario_id: String::new(),
            scenario_label: String::new(),
            running: false,
            paused: false,
            frame_rate_hz,
            playback_speed: 1.0,
        }
    }
}
