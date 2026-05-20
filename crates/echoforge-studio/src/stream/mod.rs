//! Realtime radar-simulation streaming layer.
//!
//! Exposes a WebSocket at `/ws/radar` carrying live or replayed radar
//! scans, plus a REST control surface under `/api/sim/*`. See
//! [`frames`] for the wire contract and [`encode`] for the binary codec.

pub mod control;
pub mod encode;
pub mod engine;
pub mod frames;
mod live;
pub mod mapping;
mod replay;

#[cfg(test)]
#[path = "stream_tests.rs"]
mod stream_tests;

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::broadcast::error::RecvError;

use control::{builtin_scenarios, ControlCommand, RadarParamPatch, SimMode};
use encode::encode_scan_frame;
use engine::SimEngine;
use frames::{ControlFrame, OutboundFrame, ScenarioSummary, StatusFrame};

/// Default replay bundle when a `replay` start request omits a path.
const DEFAULT_REPLAY_BUNDLE: &str = "tests/science/fixtures/bundles/v1_pass";

/// Build the streaming router (`/ws/radar` + `/api/sim/*`).
pub fn stream_router(engine: Arc<SimEngine>) -> Router {
    Router::new()
        .route("/ws/radar", get(ws_handler))
        .route("/api/sim/status", get(status_handler))
        .route("/api/sim/scenarios", get(scenarios_handler))
        .route("/api/sim/start", post(start_handler))
        .route("/api/sim/stop", post(stop_handler))
        .route("/api/sim/pause", post(pause_handler))
        .route("/api/sim/resume", post(resume_handler))
        .route("/api/sim/speed", post(speed_handler))
        .route("/api/sim/params", post(params_handler))
        .with_state(engine)
}

// ---------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(engine): State<Arc<SimEngine>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, engine))
}

fn control_message(frame: &ControlFrame) -> Option<Message> {
    serde_json::to_string(frame).ok().map(|s| Message::Text(s.into()))
}

fn scan_message(frame: &frames::ScanFrame) -> Option<Message> {
    encode_scan_frame(frame).ok().map(|b| Message::Binary(b.into()))
}

/// Per-connection task: pushes broadcast frames, refreshes `SessionInfo`
/// on status changes, and tolerates a lagging consumer.
async fn handle_socket(socket: WebSocket, engine: Arc<SimEngine>) {
    let (mut sink, mut stream) = socket.split();
    let mut frames = engine.subscribe();
    let mut status = engine.watch_status();

    let hello = ControlFrame::SessionInfo(engine.session_info());
    if let Some(msg) = control_message(&hello) {
        if sink.send(msg).await.is_err() {
            return;
        }
    }

    loop {
        tokio::select! {
            received = frames.recv() => match received {
                Ok(frame) => {
                    let msg = match &*frame {
                        OutboundFrame::Control(c) => control_message(c),
                        OutboundFrame::Scan(s) => scan_message(s),
                    };
                    if let Some(msg) = msg {
                        if sink.send(msg).await.is_err() {
                            break;
                        }
                    }
                }
                Err(RecvError::Lagged(dropped)) => {
                    let mut warn = StatusFrame::warn(
                        "lagged",
                        format!("client fell behind; {dropped} frames dropped"),
                    );
                    warn.dropped_frames = Some(dropped);
                    if let Some(msg) = control_message(&ControlFrame::Status(warn)) {
                        if sink.send(msg).await.is_err() {
                            break;
                        }
                    }
                }
                Err(RecvError::Closed) => break,
            },
            changed = status.changed() => {
                if changed.is_err() {
                    break;
                }
                let info = ControlFrame::SessionInfo(engine.session_info());
                if let Some(msg) = control_message(&info) {
                    if sink.send(msg).await.is_err() {
                        break;
                    }
                }
            }
            inbound = stream.next() => match inbound {
                None | Some(Ok(Message::Close(_))) => break,
                Some(Err(_)) => break,
                // Control flows over REST; inbound data frames are
                // ignored and axum auto-responds to pings.
                Some(Ok(_)) => {}
            },
        }
    }
}

// ---------------------------------------------------------------------
// REST control surface
// ---------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct StartRequest {
    #[serde(default)]
    scenario_id: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    bundle_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SpeedRequest {
    speed: f64,
}

async fn status_handler(State(engine): State<Arc<SimEngine>>) -> impl IntoResponse {
    Json(engine.current_status())
}

async fn scenarios_handler(State(_engine): State<Arc<SimEngine>>) -> impl IntoResponse {
    let scenarios: Vec<ScenarioSummary> = builtin_scenarios()
        .into_iter()
        .map(|s| ScenarioSummary {
            id: s.id,
            label: s.label,
            description: s.description,
        })
        .collect();
    Json(scenarios)
}

async fn start_handler(
    State(engine): State<Arc<SimEngine>>,
    Json(req): Json<StartRequest>,
) -> impl IntoResponse {
    let scenario_id = req
        .scenario_id
        .unwrap_or_else(|| engine.settings().default_scenario.clone());
    let mode = match req.mode.as_deref() {
        Some("replay") => SimMode::Replay {
            bundle_path: PathBuf::from(
                req.bundle_path
                    .unwrap_or_else(|| DEFAULT_REPLAY_BUNDLE.to_string()),
            ),
        },
        _ => SimMode::Live,
    };
    let status = engine
        .apply(ControlCommand::Start { scenario_id, mode })
        .await;
    Json(status)
}

async fn stop_handler(State(engine): State<Arc<SimEngine>>) -> impl IntoResponse {
    Json(engine.apply(ControlCommand::Stop).await)
}

async fn pause_handler(State(engine): State<Arc<SimEngine>>) -> impl IntoResponse {
    Json(engine.apply(ControlCommand::Pause).await)
}

async fn resume_handler(State(engine): State<Arc<SimEngine>>) -> impl IntoResponse {
    Json(engine.apply(ControlCommand::Resume).await)
}

async fn speed_handler(
    State(engine): State<Arc<SimEngine>>,
    Json(req): Json<SpeedRequest>,
) -> impl IntoResponse {
    Json(engine.apply(ControlCommand::SetSpeed(req.speed)).await)
}

async fn params_handler(
    State(engine): State<Arc<SimEngine>>,
    Json(patch): Json<RadarParamPatch>,
) -> impl IntoResponse {
    Json(engine.apply(ControlCommand::SetParams(patch)).await)
}
