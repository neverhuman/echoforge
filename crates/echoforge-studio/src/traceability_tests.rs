use super::*;
use serde_json::json;
use tempfile::tempdir;

fn write(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("mkdir");
    }
    fs::write(path, content).expect("write");
}

/// Build a minimal but realistic campaign output tree on disk and return
/// the (root, detection_id) pair pointing at the seeded event.
fn write_fixture_campaign() -> (tempfile::TempDir, String) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();

    let manifest = json!({
        "manifest_version": "1",
        "campaign_request_id": "shahed136-public-proxy-early-detection",
        "neutral_campaign_id": "owa-delta-pusher-public-proxy-early-detection",
        "generated_at": "2026-05-18T00:00:00Z",
        "root_seed": 20260518136u64,
        "records": [
            {
                "record_id": "record_000223",
                "record_index": 222,
                "target_family": "owa_delta_pusher_public_proxy",
                "class_id": "owa-delta-pusher-fixed-wing-public-proxy",
                "bucket": "positive_owa_delta_pusher",
                "is_shahed_public_proxy": true,
                "is_hard_negative": false,
                "hard_negative_family": "positive_public_proxy",
                "first_detectable_frame": 90,
                "first_model_trigger_frame": 102,
                "cpi_pulses": 64,
                "tensor_dir": "records/record_000223/products",
                "frame_labels_path": "records/record_000223/frame_labels.csv",
                "truth_metadata_path": "records/record_000223/truth_metadata.json",
                "model_predictions_path": "records/record_000223/model_predictions.csv",
                "detector_events_path": "records/record_000223/detector_events.json",
                "max_confidence_by_model": {
                    "cfar_tracker_baseline": 0.81115806f32,
                    "feature_tree_classifier": 0.5f32,
                    "temporal_tiny_model": 0.2f32
                },
                "first_trigger_by_model": {
                    "cfar_tracker_baseline": 102i64
                }
            }
        ]
    });
    write(&root.join("campaign_manifest.json"), &manifest.to_string());

    let dataset = json!({
        "id": "ef:dataset_card:owa-delta-pusher-fixed-wing-public-proxy:fixture:1",
        "splits": {"train": 700, "validation": 150, "test": 150}
    });
    write(&root.join("dataset_card.json"), &dataset.to_string());

    let truth = json!({
        "record_id": "record_000223",
        "neutral_object_id": "owa-delta-pusher-fixed-wing-public-proxy",
        "target_family": "owa_delta_pusher_public_proxy",
        "is_shahed_public_proxy": true,
        "is_hard_negative": false
    });
    write(
        &root.join("records/record_000223/truth_metadata.json"),
        &truth.to_string(),
    );

    let events = json!([
        {
            "model_id": "cfar_tracker_baseline",
            "frame_index": 102,
            "time_s": 51.0,
            "confidence": 0.81115806f64,
            "threshold": 0.8f64,
            "consecutive_frames": 2
        }
    ]);
    write(
        &root.join("records/record_000223/detector_events.json"),
        &events.to_string(),
    );

    write(
        &root.join("runtime_report.json"),
        &json!({"generated_at": "2026-05-18T12:20:20Z", "selected_backend": "gpu"}).to_string(),
    );

    let detection_id = "record_000223__cfar_tracker_baseline__frame_0102".to_string();
    (dir, detection_id)
}

#[test]
fn parses_well_formed_detection_id() {
    let parsed =
        parse_detection_id("record_000223__cfar_tracker_baseline__frame_0102").expect("parse");
    assert_eq!(parsed.record_id, "record_000223");
    assert_eq!(parsed.model_id, "cfar_tracker_baseline");
    assert_eq!(parsed.frame_index, 102);
}

#[test]
fn rejects_malformed_detection_id() {
    assert!(matches!(
        parse_detection_id("garbage"),
        Err(TraceabilityError::MalformedDetectionId(_))
    ));
    assert!(matches!(
        parse_detection_id("record_x__model_y__frame_notnum"),
        Err(TraceabilityError::MalformedDetectionId(_))
    ));
    assert!(matches!(
        parse_detection_id("__model_y__frame_1"),
        Err(TraceabilityError::MalformedDetectionId(_))
    ));
    assert!(matches!(
        parse_detection_id("record_x__model_y__bogus_1"),
        Err(TraceabilityError::MalformedDetectionId(_))
    ));
}

#[test]
fn lookup_returns_full_chain_for_known_detection() {
    let (dir, detection_id) = write_fixture_campaign();
    let config = TraceabilityConfig::with_root(dir.path());
    let response = lookup_detection(&detection_id, &config).expect("lookup");

    assert_eq!(response.detection_id, detection_id);
    assert_eq!(response.record_id, "record_000223");
    assert_eq!(response.scenario.seed, 20260518136);
    assert_eq!(
        response.scenario.preset,
        "shahed136-public-proxy-early-detection"
    );
    assert_eq!(
        response.target.public_proxy_id,
        "owa-delta-pusher-fixed-wing-public-proxy"
    );
    assert!(response.target.display_name.contains("Owa"));
    assert!(response
        .material
        .material_family
        .contains("owa_delta_pusher"));
    assert_eq!(response.sensor.band_name, "x_band_proxy");
    assert!(response
        .dataset
        .dataset_card_id
        .starts_with("ef:dataset_card:"));
    assert_eq!(response.dataset.splits.get("train").copied(), Some(700));
    assert_eq!(
        response.dataset.splits.get("validation").copied(),
        Some(150)
    );
    assert_eq!(response.dataset.splits.get("test").copied(), Some(150));
    assert_eq!(response.detection.frame_index, 102);
    assert_eq!(response.detection.model_id, "cfar_tracker_baseline");
    assert!((response.detection.confidence - 0.81115806f64).abs() < 1e-6);
    assert_eq!(response.detection.threshold, 0.8);
    assert_eq!(response.produced_at, "2026-05-18T12:20:20Z");
    assert_eq!(response.limitation, PUBLIC_PROXY_LIMITATION);
}

#[test]
fn lookup_returns_not_found_when_record_missing() {
    let (dir, _) = write_fixture_campaign();
    let config = TraceabilityConfig::with_root(dir.path());
    let err = lookup_detection("record_999999__cfar_tracker_baseline__frame_0001", &config)
        .expect_err("should miss");
    assert!(matches!(err, TraceabilityError::NotFound(_)));
}

#[test]
fn lookup_returns_detection_absent_when_event_did_not_fire() {
    let (dir, _) = write_fixture_campaign();
    let config = TraceabilityConfig::with_root(dir.path());
    let err = lookup_detection("record_000223__cfar_tracker_baseline__frame_0001", &config)
        .expect_err("event did not fire on frame 1");
    assert!(matches!(err, TraceabilityError::DetectionAbsent(_)));
}

#[test]
fn lookup_returns_not_found_when_no_roots_configured() {
    let config = TraceabilityConfig {
        campaign_roots: Vec::new(),
        object_packs_root: None,
    };
    let err = lookup_detection("record_000223__cfar_tracker_baseline__frame_0102", &config)
        .expect_err("no roots");
    assert!(matches!(err, TraceabilityError::NotFound(_)));
}

#[test]
fn response_has_stable_json_shape() {
    let (dir, detection_id) = write_fixture_campaign();
    let config = TraceabilityConfig::with_root(dir.path());
    let response = lookup_detection(&detection_id, &config).expect("lookup");
    let value = serde_json::to_value(&response).expect("serialize");

    let obj = value.as_object().expect("object");
    for key in [
        "detection_id",
        "record_id",
        "scenario",
        "target",
        "material",
        "sensor",
        "dataset",
        "detection",
        "produced_at",
        "limitation",
    ] {
        assert!(obj.contains_key(key), "missing key {key}");
    }
    assert!(value["scenario"]["seed"].is_u64());
    assert!(value["scenario"]["preset"].is_string());
    assert!(value["target"]["object_card_id"].is_string());
    assert!(value["dataset"]["splits"].is_object());
}

fn make_detection_app(dir: &tempfile::TempDir) -> axum::Router {
    let state = Arc::new(TraceabilityState {
        config: TraceabilityConfig::with_root(dir.path()),
    });
    axum::Router::new()
        .route(
            "/provenance/detection/{detection_id}",
            axum::routing::get(detection_handler),
        )
        .with_state(state)
}

fn oneshot_response(app: axum::Router, uri: &str) -> axum::http::Response<axum::body::Body> {
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt")
        .block_on(async {
            app.oneshot(
                Request::builder()
                    .uri(uri)
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response")
        })
}

#[test]
fn axum_endpoint_returns_200_and_json() {
    let (dir, detection_id) = write_fixture_campaign();
    let response = oneshot_response(
        make_detection_app(&dir),
        &format!("/provenance/detection/{detection_id}"),
    );
    assert_eq!(response.status(), StatusCode::OK);
}

#[test]
fn axum_endpoint_returns_404_for_unknown_detection() {
    let (dir, _) = write_fixture_campaign();
    let response = oneshot_response(
        make_detection_app(&dir),
        "/provenance/detection/record_999999__some_model__frame_0001",
    );
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}
