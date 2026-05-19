//! Traceability endpoint: walk back from a detection ID to the full
//! provenance chain (record manifest -> scenario seed -> object card ->
//! material card -> sensor archetype -> dataset card).
//!
//! Detection IDs are formed deterministically from the campaign output as
//! `<record_id>__<model_id>__frame_<NNNN>`, for example
//! `record_000223__cfar_tracker_baseline__frame_0102`.
//!
//! The detection must be present in the record's `detector_events.json`
//! file -- the endpoint does not invent detections that never crossed the
//! detector threshold. The endpoint READS the campaign output directory
//! but never writes to it.
//!
//! This is the FUCKIT.md Critical Review section M
//! "one-click drill from detection to provenance" milestone.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use echoforge_core::ObjectCard;
use serde::{Deserialize, Serialize};

const DETECTION_ID_DELIMITER: &str = "__";
const FRAME_PREFIX: &str = "frame_";
const PUBLIC_PROXY_LIMITATION: &str = "public-proxy traceability; not measured truth";
const DEFAULT_SENSOR_BAND: &str = "x_band_proxy";
const DEFAULT_SENSOR_ARCHETYPE_ID: &str = "x-band-proxy-airspace";

// ---------------------------------------------------------------------------
// Public response contract
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioRef {
    pub seed: u64,
    pub preset: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetRef {
    pub object_card_id: String,
    pub public_proxy_id: String,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_variant: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialRef {
    pub material_card_id: String,
    pub material_family: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorRef {
    pub sensor_archetype_id: String,
    pub band_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRef {
    pub dataset_card_id: String,
    pub splits: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionRef {
    pub model_id: String,
    pub frame_index: usize,
    pub time_s: f64,
    pub confidence: f64,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceabilityResponse {
    pub detection_id: String,
    pub record_id: String,
    pub scenario: ScenarioRef,
    pub target: TargetRef,
    pub material: MaterialRef,
    pub sensor: SensorRef,
    pub dataset: DatasetRef,
    pub detection: DetectionRef,
    pub produced_at: String,
    pub limitation: String,
}

// ---------------------------------------------------------------------------
// Configuration & state
// ---------------------------------------------------------------------------

/// Roots that the traceability resolver will consult, in order. The first
/// campaign root that contains a `campaign_manifest.json` matching the
/// detection's `record_id` wins.
#[derive(Debug, Clone)]
pub struct TraceabilityConfig {
    pub campaign_roots: Vec<PathBuf>,
    /// Optional override directory for object-card YAML lookup (defaults to
    /// `<workspace>/object-packs/public-proxy/`).
    pub object_packs_root: Option<PathBuf>,
}

impl TraceabilityConfig {
    pub fn from_env() -> Self {
        let roots = match std::env::var("ECHOFORGE_CAMPAIGN_ROOTS") {
            Ok(raw) => raw
                .split(':')
                .filter(|s| !s.trim().is_empty())
                .map(PathBuf::from)
                .collect::<Vec<_>>(),
            Err(_) => Vec::new(),
        };
        let object_packs_root = std::env::var("ECHOFORGE_OBJECT_PACKS_ROOT")
            .ok()
            .map(PathBuf::from);
        Self {
            campaign_roots: roots,
            object_packs_root,
        }
    }

    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self {
            campaign_roots: vec![root.into()],
            object_packs_root: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TraceabilityState {
    pub config: TraceabilityConfig,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum TraceabilityError {
    /// The detection_id did not parse as `<record_id>__<model_id>__frame_<NNNN>`.
    MalformedDetectionId(String),
    /// The detection_id parsed but no campaign root contained that record.
    NotFound(String),
    /// The detector_events.json for the record did not contain a matching
    /// `(model_id, frame_index)` row -- the detection never fired.
    DetectionAbsent(String),
    /// A required file (campaign_manifest.json, dataset_card.json, etc.)
    /// is missing from the campaign root.
    MissingArtifact(String),
    /// An artifact existed but failed to parse.
    Parse(String),
    /// Generic I/O.
    Io(String),
}

impl std::fmt::Display for TraceabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedDetectionId(s) => write!(f, "malformed detection_id: {s}"),
            Self::NotFound(s) => write!(f, "detection not found: {s}"),
            Self::DetectionAbsent(s) => write!(f, "detection absent: {s}"),
            Self::MissingArtifact(s) => write!(f, "missing artifact: {s}"),
            Self::Parse(s) => write!(f, "parse error: {s}"),
            Self::Io(s) => write!(f, "io error: {s}"),
        }
    }
}

impl std::error::Error for TraceabilityError {}

impl IntoResponse for TraceabilityError {
    fn into_response(self) -> axum::response::Response {
        let (status, kind) = match &self {
            Self::MalformedDetectionId(_) => (StatusCode::BAD_REQUEST, "malformed_detection_id"),
            Self::NotFound(_) | Self::DetectionAbsent(_) => (StatusCode::NOT_FOUND, "not_found"),
            Self::MissingArtifact(_) => (StatusCode::NOT_FOUND, "missing_artifact"),
            Self::Parse(_) | Self::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
        };
        let body = Json(serde_json::json!({
            "service": super::DEFAULT_SERVICE_NAME,
            "status": "error",
            "error": kind,
            "message": self.to_string(),
        }));
        (status, body).into_response()
    }
}

// ---------------------------------------------------------------------------
// Detection ID parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedDetectionId {
    pub record_id: String,
    pub model_id: String,
    pub frame_index: usize,
}

pub fn parse_detection_id(raw: &str) -> Result<ParsedDetectionId, TraceabilityError> {
    let parts: Vec<&str> = raw.splitn(3, DETECTION_ID_DELIMITER).collect();
    if parts.len() != 3 {
        return Err(TraceabilityError::MalformedDetectionId(format!(
            "expected `<record_id>{DETECTION_ID_DELIMITER}<model_id>{DETECTION_ID_DELIMITER}{FRAME_PREFIX}<NNNN>`, got {raw:?}"
        )));
    }
    let record_id = parts[0].to_string();
    let model_id = parts[1].to_string();
    let frame_token = parts[2];
    if !frame_token.starts_with(FRAME_PREFIX) {
        return Err(TraceabilityError::MalformedDetectionId(format!(
            "frame token must start with `{FRAME_PREFIX}`, got {frame_token:?}"
        )));
    }
    let frame_index: usize = frame_token[FRAME_PREFIX.len()..].parse().map_err(|_| {
        TraceabilityError::MalformedDetectionId(format!(
            "frame token after `{FRAME_PREFIX}` must be a non-negative integer, got {frame_token:?}"
        ))
    })?;
    if record_id.is_empty() || model_id.is_empty() {
        return Err(TraceabilityError::MalformedDetectionId(
            "record_id and model_id must be non-empty".to_string(),
        ));
    }
    Ok(ParsedDetectionId {
        record_id,
        model_id,
        frame_index,
    })
}

// ---------------------------------------------------------------------------
// Campaign artifacts (typed, minimal — we only deserialize what we need so
// we are tolerant to upstream schema additions).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct CampaignManifestMin {
    campaign_request_id: String,
    root_seed: u64,
    records: Vec<CampaignRecordMin>,
}

#[derive(Debug, Clone, Deserialize)]
struct CampaignRecordMin {
    record_id: String,
    class_id: String,
    target_family: String,
    truth_metadata_path: String,
    detector_events_path: String,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // `is_shahed_public_proxy` is round-tripped from
                    // truth_metadata.json; reserved for future positive-class
                    // enrichment in the trace response.
struct TruthMetadataMin {
    #[serde(default)]
    neutral_object_id: String,
    #[serde(default)]
    target_family: String,
    #[serde(default)]
    is_shahed_public_proxy: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct DetectorEvent {
    model_id: String,
    frame_index: usize,
    time_s: f64,
    confidence: f64,
    threshold: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct DatasetCardMin {
    id: String,
    #[serde(default)]
    splits: BTreeMap<String, u64>,
}

// ---------------------------------------------------------------------------
// Lookup core
// ---------------------------------------------------------------------------

pub fn lookup_detection(
    detection_id: &str,
    config: &TraceabilityConfig,
) -> Result<TraceabilityResponse, TraceabilityError> {
    let parsed = parse_detection_id(detection_id)?;
    if config.campaign_roots.is_empty() {
        return Err(TraceabilityError::NotFound(format!(
            "no campaign roots configured; cannot resolve {}",
            parsed.record_id
        )));
    }

    for root in &config.campaign_roots {
        match try_lookup_in_root(detection_id, &parsed, root, config) {
            Ok(response) => return Ok(response),
            Err(TraceabilityError::NotFound(_)) | Err(TraceabilityError::MissingArtifact(_)) => {
                continue;
            }
            Err(other) => return Err(other),
        }
    }

    Err(TraceabilityError::NotFound(format!(
        "no campaign root contained record {}",
        parsed.record_id
    )))
}

fn try_lookup_in_root(
    detection_id: &str,
    parsed: &ParsedDetectionId,
    root: &Path,
    config: &TraceabilityConfig,
) -> Result<TraceabilityResponse, TraceabilityError> {
    let manifest_path = root.join("campaign_manifest.json");
    if !manifest_path.is_file() {
        return Err(TraceabilityError::MissingArtifact(format!(
            "{}",
            manifest_path.display()
        )));
    }
    let manifest: CampaignManifestMin = read_json(&manifest_path)?;
    let record = match manifest
        .records
        .iter()
        .find(|r| r.record_id == parsed.record_id)
    {
        Some(r) => r,
        None => return Err(TraceabilityError::NotFound(parsed.record_id.clone())),
    };

    let events_path = root.join(&record.detector_events_path);
    let events: Vec<DetectorEvent> = if events_path.is_file() {
        read_json(&events_path)?
    } else {
        Vec::new()
    };
    let event = match events
        .into_iter()
        .find(|ev| ev.model_id == parsed.model_id && ev.frame_index == parsed.frame_index)
    {
        Some(ev) => ev,
        None => {
            return Err(TraceabilityError::DetectionAbsent(format!(
                "no detector event for ({}, frame {}) in {}",
                parsed.model_id, parsed.frame_index, record.record_id
            )))
        }
    };

    let truth_path = root.join(&record.truth_metadata_path);
    let truth: TruthMetadataMin = if truth_path.is_file() {
        read_json(&truth_path)?
    } else {
        TruthMetadataMin {
            neutral_object_id: record.class_id.clone(),
            target_family: record.target_family.clone(),
            is_shahed_public_proxy: false,
        }
    };

    let dataset_card_path = root.join("dataset_card.json");
    let dataset: DatasetCardMin = if dataset_card_path.is_file() {
        read_json(&dataset_card_path)?
    } else {
        return Err(TraceabilityError::MissingArtifact(format!(
            "{}",
            dataset_card_path.display()
        )));
    };

    let runtime_report_path = root.join("runtime_report.json");
    let produced_at = match read_produced_at(&runtime_report_path) {
        Some(v) => v,
        None => match read_manifest_generated_at(&manifest_path) {
            Some(v) => v,
            None => "unknown".to_string(),
        },
    };

    // Object card lookup is best-effort: we try the configured object-packs
    // root for a matching `<class_id>.yaml` or the canonical
    // `public-proxy/object_card.yaml`. If nothing is found we fall back
    // to deriving a public_proxy_id and display_name from the truth
    // metadata so the endpoint stays useful in CI fixtures that don't ship
    // the object-pack tree.
    let target = resolve_target(
        &record.class_id,
        &truth,
        config.object_packs_root.as_deref(),
    )?;
    let material = resolve_material(&truth, &record.class_id);
    let sensor = resolve_sensor();

    Ok(TraceabilityResponse {
        detection_id: detection_id.to_string(),
        record_id: parsed.record_id.clone(),
        scenario: ScenarioRef {
            seed: manifest.root_seed,
            preset: manifest.campaign_request_id.clone(),
        },
        target,
        material,
        sensor,
        dataset: DatasetRef {
            dataset_card_id: dataset.id,
            splits: dataset.splits,
        },
        detection: DetectionRef {
            model_id: event.model_id,
            frame_index: event.frame_index,
            time_s: event.time_s,
            confidence: event.confidence,
            threshold: event.threshold,
        },
        produced_at,
        limitation: PUBLIC_PROXY_LIMITATION.to_string(),
    })
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, TraceabilityError> {
    let raw = fs::read_to_string(path)
        .map_err(|e| TraceabilityError::Io(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&raw)
        .map_err(|e| TraceabilityError::Parse(format!("{}: {e}", path.display())))
}

fn read_produced_at(runtime_report: &Path) -> Option<String> {
    if !runtime_report.is_file() {
        return None;
    }
    let raw = fs::read_to_string(runtime_report).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .get("generated_at")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn read_manifest_generated_at(manifest_path: &Path) -> Option<String> {
    let raw = fs::read_to_string(manifest_path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .get("generated_at")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn resolve_target(
    class_id: &str,
    truth: &TruthMetadataMin,
    object_packs_root: Option<&Path>,
) -> Result<TargetRef, TraceabilityError> {
    if let Some(root) = object_packs_root {
        // Try `<class_id>.yaml` first, then the canonical fixture.
        let candidates = [
            root.join(format!("{class_id}.yaml")),
            root.join("object_card.yaml"),
        ];
        for candidate in candidates {
            if candidate.is_file() {
                if let Some(card) = read_object_card_yaml(&candidate) {
                    return Ok(TargetRef {
                        object_card_id: if card.id.is_empty() {
                            class_id.to_string()
                        } else {
                            card.id
                        },
                        public_proxy_id: card.public_proxy_id,
                        display_name: card.display_name,
                        object_family: Some(card.object_family),
                        geometry_variant: Some(card.geometry_variant),
                    });
                }
            }
        }
    }
    // Recovery path derived from truth metadata + class_id.
    let neutral = if truth.neutral_object_id.is_empty() {
        class_id.to_string()
    } else {
        truth.neutral_object_id.clone()
    };
    let display_name = humanize_class_id(&neutral);
    Ok(TargetRef {
        object_card_id: format!("ef:object_card:{neutral}:derived:1"),
        public_proxy_id: neutral.clone(),
        display_name,
        object_family: Some(if truth.target_family.is_empty() {
            "unknown".to_string()
        } else {
            truth.target_family.clone()
        }),
        geometry_variant: None,
    })
}

fn read_object_card_yaml(path: &Path) -> Option<ObjectCard> {
    let raw = fs::read_to_string(path).ok()?;
    serde_yaml::from_str::<ObjectCard>(&raw).ok()
}

fn resolve_material(truth: &TruthMetadataMin, class_id: &str) -> MaterialRef {
    // The current campaign output does not pin a specific material_card_id
    // to each record; class_id is a stable surrogate while a richer
    // material registry is being built out (Critical Review §M follow-up).
    let family = if truth.target_family.is_empty() {
        "composite_proxy".to_string()
    } else {
        format!("{}_material_proxy", truth.target_family)
    };
    MaterialRef {
        material_card_id: format!("ef:material_card:{class_id}:derived:1"),
        material_family: family,
    }
}

fn resolve_sensor() -> SensorRef {
    SensorRef {
        sensor_archetype_id: DEFAULT_SENSOR_ARCHETYPE_ID.to_string(),
        band_name: DEFAULT_SENSOR_BAND.to_string(),
    }
}

fn humanize_class_id(class_id: &str) -> String {
    let mut out = String::new();
    let mut capitalize = true;
    for ch in class_id.chars() {
        if ch == '-' || ch == '_' {
            out.push(' ');
            capitalize = true;
        } else if capitalize {
            out.extend(ch.to_uppercase());
            capitalize = false;
        } else {
            out.push(ch);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Axum handler
// ---------------------------------------------------------------------------

pub async fn detection_handler(
    State(state): State<Arc<TraceabilityState>>,
    AxumPath(detection_id): AxumPath<String>,
) -> Result<Json<TraceabilityResponse>, TraceabilityError> {
    let response = lookup_detection(&detection_id, &state.config)?;
    Ok(Json(response))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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

        // Top-level keys are the public contract documented in the FUCKIT
        // packet plan -- treat any drift as a breaking change.
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

    #[test]
    fn axum_endpoint_returns_200_and_json() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let (dir, detection_id) = write_fixture_campaign();
        let state = Arc::new(TraceabilityState {
            config: TraceabilityConfig::with_root(dir.path()),
        });
        let app = axum::Router::new()
            .route(
                "/provenance/detection/{detection_id}",
                axum::routing::get(detection_handler),
            )
            .with_state(state);

        let response = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt")
            .block_on(async {
                app.oneshot(
                    Request::builder()
                        .uri(format!("/provenance/detection/{detection_id}"))
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response")
            });
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn axum_endpoint_returns_404_for_unknown_detection() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let (dir, _) = write_fixture_campaign();
        let state = Arc::new(TraceabilityState {
            config: TraceabilityConfig::with_root(dir.path()),
        });
        let app = axum::Router::new()
            .route(
                "/provenance/detection/{detection_id}",
                axum::routing::get(detection_handler),
            )
            .with_state(state);

        let response = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt")
            .block_on(async {
                app.oneshot(
                    Request::builder()
                        .uri("/provenance/detection/record_999999__some_model__frame_0001")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response")
            });
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
