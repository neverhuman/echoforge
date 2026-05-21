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

    #[allow(dead_code)]
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
// Lookup core (implementation in traceability_lookup.rs)
// ---------------------------------------------------------------------------

#[path = "traceability_lookup.rs"]
mod traceability_lookup;
pub use traceability_lookup::lookup_detection;

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
#[path = "traceability_tests.rs"]
mod tests;
