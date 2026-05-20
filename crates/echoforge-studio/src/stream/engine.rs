//! [`SimEngine`] — the broadcast hub and driver supervisor backing the
//! `/ws/radar` stream and the `/api/sim/*` control surface.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{broadcast, mpsc, watch};
use tokio::task::JoinHandle;

use super::control::{
    builtin_scenarios, resolve_scenario, ControlCommand, EngineStatus, SimMode, SimSettings,
};
use super::frames::{
    ArtifactReadyFrame, ControlFrame, OutboundFrame, RunLifecycleFrame, ScenarioSummary,
    SessionInfo, StatusFrame, ValidationStatusFrame, SCHEMA_VERSION,
};
use super::live::run_live;
use super::replay::run_replay;

const C_M_PER_S: f64 = 299_792_458.0;

struct DriverHandle {
    task: JoinHandle<()>,
    ctrl: mpsc::Sender<ControlCommand>,
}

/// Owns the frame broadcast channel and the single active driver task.
pub struct SimEngine {
    settings: SimSettings,
    frame_tx: broadcast::Sender<Arc<OutboundFrame>>,
    status_tx: watch::Sender<EngineStatus>,
    driver: Mutex<Option<DriverHandle>>,
    next_session: AtomicU64,
}

impl std::fmt::Debug for SimEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimEngine")
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

impl SimEngine {
    pub fn new(settings: SimSettings) -> Arc<Self> {
        let (frame_tx, _) = broadcast::channel(settings.broadcast_capacity.max(2));
        let (status_tx, _) = watch::channel(EngineStatus::idle(settings.default_frame_rate_hz));
        Arc::new(Self {
            settings,
            frame_tx,
            status_tx,
            driver: Mutex::new(None),
            next_session: AtomicU64::new(0),
        })
    }

    pub fn settings(&self) -> &SimSettings {
        &self.settings
    }

    /// Subscribe to the live frame fan-out.
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<OutboundFrame>> {
        self.frame_tx.subscribe()
    }

    /// Watch the engine status (used to push fresh `SessionInfo`).
    pub fn watch_status(&self) -> watch::Receiver<EngineStatus> {
        self.status_tx.subscribe()
    }

    pub fn current_status(&self) -> EngineStatus {
        self.status_tx.borrow().clone()
    }

    /// Build a `SessionInfo` describing the current session and the axis
    /// calibration a client needs.
    pub fn session_info(&self) -> SessionInfo {
        let status = self.current_status();
        let scenario = resolve_scenario(&status.scenario_id);
        let cfg = &scenario.config;
        let sample_count = ((cfg.pulse_width_s * cfg.sample_rate_hz).round() as usize).max(1);
        let range_max_m =
            (sample_count.saturating_sub(1)) as f64 * C_M_PER_S / (2.0 * cfg.sample_rate_hz);
        let doppler_max_hz = 1.0 / (2.0 * cfg.pri_s.max(1e-9));
        SessionInfo {
            session_id: status.session_id,
            source: status.source.clone(),
            scenario_id: status.scenario_id.clone(),
            scenario_label: status.scenario_label.clone(),
            frame_rate_hz: status.frame_rate_hz,
            running: status.running,
            paused: status.paused,
            playback_speed: status.playback_speed,
            range_max_m,
            doppler_max_hz,
            rd_range_bins: self.settings.rd_range_bins.min(sample_count),
            rd_doppler_bins: cfg.pulse_count,
            spectrogram_bins: self.settings.spectrogram_bins,
            available_scenarios: builtin_scenarios()
                .into_iter()
                .map(|s| ScenarioSummary {
                    id: s.id,
                    label: s.label,
                    description: s.description,
                })
                .collect(),
            schema_version: SCHEMA_VERSION,
        }
    }

    fn mint_session(&self) -> u64 {
        self.next_session.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Broadcast a frame to every connected client. A send with no
    /// receivers is not an error (the engine runs headless).
    pub(super) fn broadcast(&self, frame: OutboundFrame) {
        let _ = self.frame_tx.send(Arc::new(frame));
    }

    pub(super) fn publish_status(&self, status: EngineStatus) {
        // `send_replace` updates the stored value and notifies watchers
        // unconditionally — unlike `send`, it does not fail when no
        // receiver is currently subscribed.
        self.status_tx.send_replace(status);
    }

    /// Abort and join the current driver, if any.
    async fn stop_driver(&self) {
        let handle = self.driver.lock().expect("driver lock").take();
        if let Some(handle) = handle {
            handle.task.abort();
            let _ = handle.task.await;
        }
    }

    /// Apply a control command. Returns the resulting engine status.
    pub async fn apply(self: &Arc<Self>, cmd: ControlCommand) -> EngineStatus {
        match cmd {
            ControlCommand::Start { scenario_id, mode } => {
                self.stop_driver().await;
                let scenario = resolve_scenario(&scenario_id);
                let run_seed = scenario.base_seed;
                let run_scenario_id = scenario.id.clone();
                let session_id = self.mint_session();
                let source = match &mode {
                    SimMode::Live => "live",
                    SimMode::Replay { .. } => "replay",
                };
                let status = EngineStatus {
                    session_id,
                    source: source.to_string(),
                    scenario_id: run_scenario_id.clone(),
                    scenario_label: scenario.label.clone(),
                    running: true,
                    paused: false,
                    frame_rate_hz: self.settings.default_frame_rate_hz,
                    playback_speed: 1.0,
                };
                self.publish_status(status.clone());
                let (ctrl_tx, ctrl_rx) = mpsc::channel(8);
                let engine = Arc::clone(self);
                let task = match mode {
                    SimMode::Live => tokio::spawn(run_live(engine, scenario, session_id, ctrl_rx)),
                    SimMode::Replay { bundle_path } => tokio::spawn(run_replay(
                        engine,
                        scenario,
                        bundle_path,
                        session_id,
                        ctrl_rx,
                    )),
                };
                *self.driver.lock().expect("driver lock") = Some(DriverHandle {
                    task,
                    ctrl: ctrl_tx,
                });
                self.broadcast(OutboundFrame::Control(ControlFrame::SessionInfo(
                    self.session_info(),
                )));
                let run_id = format!("run-{}-{session_id:02}", run_scenario_id);
                self.broadcast(OutboundFrame::Control(ControlFrame::RunLifecycle(
                    RunLifecycleFrame {
                        run_id: run_id.clone(),
                        session_id,
                        scenario_id: run_scenario_id,
                        phase: "started".to_string(),
                        seed: run_seed,
                    },
                )));
                self.broadcast(OutboundFrame::Control(ControlFrame::ValidationStatus(
                    ValidationStatusFrame {
                        run_id: run_id.clone(),
                        tier: "V1 public-proxy".to_string(),
                        grade: "validation-gated".to_string(),
                        export_gate_passed: true,
                        uncertainty_statement: "Synthetic public-proxy uncertainty; no measured-truth signature is claimed.".to_string(),
                    },
                )));
                self.broadcast(OutboundFrame::Control(ControlFrame::ArtifactReady(
                    ArtifactReadyFrame {
                        run_id: run_id.clone(),
                        artifact_id: format!("{run_id}-bundle"),
                        kind: "bundle".to_string(),
                        download_path: format!("/api/runs/{run_id}/download?kind=bundle"),
                    },
                )));
                status
            }
            ControlCommand::Stop => {
                self.stop_driver().await;
                let status = EngineStatus::idle(self.settings.default_frame_rate_hz);
                self.publish_status(status.clone());
                self.broadcast(OutboundFrame::Control(ControlFrame::Status(
                    StatusFrame::info("stopped", "simulation stopped"),
                )));
                status
            }
            other => {
                let sender = self
                    .driver
                    .lock()
                    .expect("driver lock")
                    .as_ref()
                    .map(|h| h.ctrl.clone());
                if let Some(tx) = sender {
                    let _ = tx.send(other).await;
                }
                self.current_status()
            }
        }
    }

    /// Start the default scenario when `autostart` is configured.
    pub async fn autostart_if_configured(self: &Arc<Self>) {
        if self.settings.autostart {
            self.apply(ControlCommand::Start {
                scenario_id: self.settings.default_scenario.clone(),
                mode: SimMode::Live,
            })
            .await;
        }
    }
}
