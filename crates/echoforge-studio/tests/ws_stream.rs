//! Integration tests for the `/ws/radar` stream and `/api/sim/*`
//! control surface — exercised over a real socket plus in-process
//! `tower` oneshot calls.

use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use echoforge_studio::stream::control::SimSettings;
use echoforge_studio::stream::encode::decode_scan_frame;
use echoforge_studio::stream::engine::SimEngine;
use echoforge_studio::stream::stream_router;
use futures_util::StreamExt;
use tokio::net::TcpListener;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tower::ServiceExt;

fn test_engine() -> std::sync::Arc<SimEngine> {
    SimEngine::new(SimSettings {
        default_frame_rate_hz: 60.0,
        autostart: false,
        ..SimSettings::default()
    })
}

async fn serve_router() -> (std::net::SocketAddr, axum::Router) {
    let engine = test_engine();
    let app = stream_router(engine);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let served = app.clone();
    tokio::spawn(async move {
        let _ = axum::serve(listener, served).await;
    });
    (addr, app)
}

async fn post(app: &axum::Router, uri: &str, body: &str) -> axum::http::StatusCode {
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("response")
        .status()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ws_streams_decodable_scan_frames() {
    let (addr, app) = serve_router().await;
    let url = format!("ws://{addr}/ws/radar");
    let (mut socket, _) = connect_async(&url).await.expect("ws connect");

    // First message is the SessionInfo control frame (text).
    let first = tokio::time::timeout(Duration::from_secs(2), socket.next())
        .await
        .expect("hello timeout")
        .expect("hello stream")
        .expect("hello message");
    assert!(
        matches!(first, Message::Text(_)),
        "expected SessionInfo text"
    );

    assert_eq!(post(&app, "/api/sim/start", "{}").await, 200);

    let mut scans = 0;
    let mut last_index = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
    while scans < 5 && tokio::time::Instant::now() < deadline {
        let msg = tokio::time::timeout(Duration::from_secs(3), socket.next()).await;
        if let Ok(Some(Ok(Message::Binary(bytes)))) = msg {
            let frame = decode_scan_frame(&bytes).expect("decode scan");
            if let Some(prev) = last_index {
                assert!(frame.meta.frame_index >= prev, "frame index regressed");
            }
            last_index = Some(frame.meta.frame_index);
            assert!(!frame.range_doppler.cells.is_empty());
            scans += 1;
        }
    }
    assert!(scans >= 5, "expected >=5 scan frames, got {scans}");

    assert_eq!(post(&app, "/api/sim/stop", "{}").await, 200);
}

type ClientSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn count_scan_frames(socket: &mut ClientSocket, want: usize) -> usize {
    let mut n = 0;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
    while n < want && tokio::time::Instant::now() < deadline {
        if let Ok(Some(Ok(Message::Binary(_)))) =
            tokio::time::timeout(Duration::from_secs(3), socket.next()).await
        {
            n += 1;
        }
    }
    n
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_clients_both_receive_frames() {
    let (addr, app) = serve_router().await;
    let url = format!("ws://{addr}/ws/radar");
    let (mut a, _) = connect_async(&url).await.expect("client a");
    let (mut b, _) = connect_async(&url).await.expect("client b");
    assert_eq!(post(&app, "/api/sim/start", "{}").await, 200);

    assert!(count_scan_frames(&mut a, 3).await >= 3, "client a starved");
    assert!(count_scan_frames(&mut b, 3).await >= 3, "client b starved");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rest_control_surface_responds() {
    let engine = test_engine();
    let app = stream_router(engine);

    for uri in ["/api/sim/status", "/api/sim/scenarios"] {
        let resp = app
            .clone()
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .expect("response");
        assert_eq!(resp.status(), 200, "GET {uri}");
    }

    let scenarios = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/sim/scenarios")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = axum::body::to_bytes(scenarios.into_body(), usize::MAX)
        .await
        .unwrap();
    let list: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(list.as_array().map(|a| a.len() >= 3).unwrap_or(false));

    assert_eq!(
        post(&app, "/api/sim/start", r#"{"mode":"live"}"#).await,
        200
    );
    assert_eq!(post(&app, "/api/sim/pause", "{}").await, 200);
    assert_eq!(post(&app, "/api/sim/resume", "{}").await, 200);
    assert_eq!(post(&app, "/api/sim/speed", r#"{"speed":2.0}"#).await, 200);
    assert_eq!(
        post(&app, "/api/sim/params", r#"{"transmit_power_w":2000000.0}"#).await,
        200
    );
    assert_eq!(post(&app, "/api/sim/stop", "{}").await, 200);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_mode_starts_without_panicking() {
    let engine = test_engine();
    let app = stream_router(engine);
    // Thin in-repo bundle — replay must fall back gracefully, not panic.
    let status = post(
        &app,
        "/api/sim/start",
        r#"{"mode":"replay","bundle_path":"tests/science/fixtures/bundles/v1_pass"}"#,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(post(&app, "/api/sim/stop", "{}").await, 200);
}
