//! Shared real-time scan driver, plus the `live` entry point.
//!
//! [`drive`] steps `synthesize_scene` on a fixed cadence and broadcasts
//! one [`ScanFrame`] per dwell. The `live` and `replay` drivers differ
//! only in their seed base and their session label — both run [`drive`].

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use echoforge_radar::{synthesize_scene, EpisodeSeed};
use tokio::sync::mpsc;
use tokio::time::{interval, Interval, MissedTickBehavior};

use super::control::{ControlCommand, EngineStatus, ScenarioSpec};
use super::engine::SimEngine;
use super::frames::{ControlFrame, OutboundFrame, StatusFrame};
use super::mapping::{episode_to_scan, FrameContext, TrackBook};

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const ROTATION_PERIOD_S: f64 = 3.2;

pub(super) fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn make_ticker(frame_rate_hz: f64, speed: f64) -> Interval {
    let period = (1.0 / (frame_rate_hz * speed)).clamp(0.004, 4.0);
    let mut ticker = interval(Duration::from_secs_f64(period));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    ticker
}

/// Distinguishes a live run from a replay run within the shared loop.
pub(super) struct DriverSource {
    /// Session source label — `"live"` or `"replay"`.
    pub label: &'static str,
    /// Per-session seed base mixed with the frame index.
    pub seed_base: u64,
    /// Status message broadcast when the driver starts.
    pub startup_message: String,
}

fn status_of(
    source: &str,
    session_id: u64,
    scenario: &ScenarioSpec,
    frame_rate_hz: f64,
    speed: f64,
    paused: bool,
) -> EngineStatus {
    EngineStatus {
        session_id,
        source: source.to_string(),
        scenario_id: scenario.id.clone(),
        scenario_label: scenario.label.clone(),
        running: true,
        paused,
        frame_rate_hz,
        playback_speed: speed,
    }
}

/// The shared scan loop. Runs until the control channel closes.
pub(super) async fn drive(
    engine: Arc<SimEngine>,
    scenario: ScenarioSpec,
    session_id: u64,
    mut ctrl: mpsc::Receiver<ControlCommand>,
    source: DriverSource,
) {
    let settings = engine.settings().clone();
    let frame_rate = settings.default_frame_rate_hz.clamp(1.0, 60.0);
    let mut config = scenario.config.clone();
    let noise = scenario.noise;
    let scene = scenario.scene();
    let mut tracker = TrackBook::default();

    let mut speed = 1.0_f64;
    let mut paused = false;
    let mut frame_index = 0u64;
    let mut sim_time = 0.0_f64;
    let mut beam_az = 0.0_f64;
    let mut last = Instant::now();
    let mut ticker = make_ticker(frame_rate, speed);

    engine.broadcast(OutboundFrame::Control(ControlFrame::Status(
        StatusFrame::info("started", source.startup_message.clone()),
    )));

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                if paused {
                    continue;
                }
                let seed = EpisodeSeed(source.seed_base ^ frame_index.wrapping_mul(GOLDEN));
                let scene_c = scene.clone();
                let config_c = config.clone();
                let t0 = Instant::now();
                let result = tokio::task::spawn_blocking(move || {
                    synthesize_scene(scene_c, config_c, noise, seed)
                })
                .await;
                let compute_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let episode = match result {
                    Ok(ep) => ep,
                    Err(_) => {
                        engine.broadcast(OutboundFrame::Control(ControlFrame::Status(
                            StatusFrame::error("synth_panic", "scene synthesis failed"),
                        )));
                        continue;
                    }
                };
                let now = Instant::now();
                let achieved = 1.0 / now.duration_since(last).as_secs_f64().max(1e-6);
                last = now;
                let ctx = FrameContext {
                    scenario: &scenario,
                    frame_index,
                    sim_time_s: sim_time,
                    wall_time_ms: unix_ms(),
                    beam_azimuth_deg: beam_az,
                    rd_range_bins: settings.rd_range_bins,
                    spectrogram_bins: settings.spectrogram_bins,
                    frame_compute_ms: compute_ms,
                    scan_rate_hz: achieved.min(frame_rate * speed),
                };
                let scan = episode_to_scan(&episode, &ctx, &mut tracker);
                engine.broadcast(OutboundFrame::Scan(scan));
                frame_index += 1;
                sim_time += 1.0 / frame_rate;
                beam_az = (beam_az + 360.0 / (frame_rate * ROTATION_PERIOD_S)) % 360.0;
            }
            cmd = ctrl.recv() => {
                match cmd {
                    None => break,
                    Some(ControlCommand::Pause) => {
                        paused = true;
                        engine.publish_status(status_of(source.label, session_id, &scenario, frame_rate, speed, true));
                        engine.broadcast(OutboundFrame::Control(ControlFrame::Status(
                            StatusFrame::info("paused", "simulation paused"),
                        )));
                    }
                    Some(ControlCommand::Resume) => {
                        paused = false;
                        last = Instant::now();
                        engine.publish_status(status_of(source.label, session_id, &scenario, frame_rate, speed, false));
                        engine.broadcast(OutboundFrame::Control(ControlFrame::Status(
                            StatusFrame::info("resumed", "simulation resumed"),
                        )));
                    }
                    Some(ControlCommand::SetSpeed(s)) => {
                        speed = s.clamp(0.1, 8.0);
                        ticker = make_ticker(frame_rate, speed);
                        engine.publish_status(status_of(source.label, session_id, &scenario, frame_rate, speed, paused));
                    }
                    Some(ControlCommand::SetParams(patch)) => {
                        patch.apply_to(&mut config);
                        engine.broadcast(OutboundFrame::Control(ControlFrame::Status(
                            StatusFrame::info("params_updated", "radar parameters updated"),
                        )));
                    }
                    Some(_) => {}
                }
            }
        }
    }
}

/// Live driver — steps the simulation forward in real time.
pub(super) async fn run_live(
    engine: Arc<SimEngine>,
    scenario: ScenarioSpec,
    session_id: u64,
    ctrl: mpsc::Receiver<ControlCommand>,
) {
    let source = DriverSource {
        label: "live",
        seed_base: scenario.base_seed,
        startup_message: format!("live simulation: {}", scenario.label),
    };
    drive(engine, scenario, session_id, ctrl, source).await;
}
