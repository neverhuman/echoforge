//! Wire contract for the realtime radar stream.
//!
//! Two message families travel over `/ws/radar`:
//!
//!   * [`ControlFrame`] — low-rate lifecycle/session messages, sent as
//!     JSON **text** WebSocket frames.
//!   * [`ScanFrame`] — one radar scan, sent as a compact **binary**
//!     WebSocket frame (see [`super::encode`]). The structured part is
//!     JSON; the two heavy 2-D arrays (range-Doppler grid, micro-Doppler
//!     column) ride as quantised `u8` blocks.
//!
//! Splitting text vs binary lets the browser dispatch purely on the
//! `MessageEvent.data` type (`string` vs `ArrayBuffer`) with no envelope
//! tagging overhead, and keeps the heavy grids out of JSON entirely.

use serde::{Deserialize, Serialize};

/// Binary container magic — `EC40` (EchoForge) + format revision `01`.
pub const SCAN_FRAME_MAGIC: u32 = 0xEC40_0001;

/// Contract revision. Bump on any breaking change to a frame struct so
/// a stale client can detect the mismatch from [`SessionInfo`].
pub const SCHEMA_VERSION: u32 = 1;

/// Fixed binary header length, in bytes, of an encoded [`ScanFrame`].
pub const SCAN_HEADER_LEN: usize = 34;

// ---------------------------------------------------------------------
// Control-plane frames (JSON text)
// ---------------------------------------------------------------------

/// A control-plane message — JSON, sent as a text WebSocket frame.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlFrame {
    /// Sent once on connect and again whenever the engine session
    /// changes (scenario switch, start/stop, parameter patch).
    SessionInfo(SessionInfo),
    /// Lifecycle / error notice (paused, ended, lagged client, …).
    Status(StatusFrame),
}

/// Describes the active simulation session and the axis calibration a
/// client needs to render incoming [`ScanFrame`]s.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionInfo {
    pub session_id: u64,
    /// `"live"`, `"replay"`, or `"idle"`.
    pub source: String,
    pub scenario_id: String,
    pub scenario_label: String,
    pub frame_rate_hz: f64,
    pub running: bool,
    pub paused: bool,
    pub playback_speed: f64,
    /// Maximum unambiguous range of the range axis (m).
    pub range_max_m: f64,
    /// Half-width of the Doppler axis (Hz); the axis spans ±this value.
    pub doppler_max_hz: f64,
    pub rd_range_bins: usize,
    pub rd_doppler_bins: usize,
    pub spectrogram_bins: usize,
    pub available_scenarios: Vec<ScenarioSummary>,
    pub schema_version: u32,
}

/// One selectable built-in scenario.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScenarioSummary {
    pub id: String,
    pub label: String,
    pub description: String,
}

/// A lifecycle or error notice.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StatusFrame {
    /// `"info"`, `"warn"`, or `"error"`.
    pub level: String,
    /// Stable machine code, e.g. `"started"`, `"stopped"`, `"lagged"`.
    pub code: String,
    pub message: String,
    /// Number of frames a slow consumer missed — set only for `lagged`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub dropped_frames: Option<u64>,
}

impl StatusFrame {
    pub fn info(code: &str, message: impl Into<String>) -> Self {
        Self {
            level: "info".to_string(),
            code: code.to_string(),
            message: message.into(),
            dropped_frames: None,
        }
    }

    pub fn warn(code: &str, message: impl Into<String>) -> Self {
        Self {
            level: "warn".to_string(),
            code: code.to_string(),
            message: message.into(),
            dropped_frames: None,
        }
    }

    pub fn error(code: &str, message: impl Into<String>) -> Self {
        Self {
            level: "error".to_string(),
            code: code.to_string(),
            message: message.into(),
            dropped_frames: None,
        }
    }
}

// ---------------------------------------------------------------------
// Scan frame (binary)
// ---------------------------------------------------------------------

/// One complete radar scan. Encoded to a binary blob by
/// [`super::encode::encode_scan_frame`].
#[derive(Debug, Clone, PartialEq)]
pub struct ScanFrame {
    pub meta: ScanMeta,
    pub range_doppler: RangeDopplerGrid,
    pub micro_doppler: MicroDopplerColumn,
}

/// The structured (JSON-encoded) portion of a [`ScanFrame`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScanMeta {
    pub frame_index: u64,
    /// Simulation clock (s) — monotonic within a session.
    pub sim_time_s: f64,
    /// Wall-clock emission time (ms since the UNIX epoch).
    pub wall_time_ms: u64,
    /// Current rotating-beam azimuth (deg, 0 = North, clockwise).
    pub beam_azimuth_deg: f64,
    /// Truth-derived blips, one per scene entity.
    pub ppi: Vec<PpiBlip>,
    /// CFAR detections from the signal-processing chain (range-only).
    pub detections: Vec<RdDetection>,
    /// Track table rows from the tracking/fusion adapter.
    pub tracks: Vec<TrackRow>,
    pub telemetry: Telemetry,
}

/// A PPI blip — a scene entity at its truth range and bearing, with the
/// signal-processing detection verdict overlaid.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpiBlip {
    pub entity_id: usize,
    pub range_m: f64,
    pub azimuth_deg: f64,
    pub amplitude_db: f64,
    pub snr_db: f64,
    /// True when a CFAR detection associates with this entity's range.
    pub detected: bool,
    pub class_label: String,
}

/// A CFAR detection located on the range-Doppler plane.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RdDetection {
    pub range_bin: usize,
    pub range_m: f64,
    pub doppler_bin: usize,
    pub magnitude_db: f64,
    pub snr_db: f64,
}

/// One row of the operator track table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrackRow {
    pub track_id: usize,
    pub range_m: f64,
    pub azimuth_deg: f64,
    pub radial_velocity_mps: f64,
    pub snr_db: f64,
    pub confidence: f32,
    pub class_label: String,
    pub age_frames: u32,
}

/// Link-budget and per-frame diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Telemetry {
    pub snr_db: f64,
    pub received_power_dbw: f64,
    pub noise_power_dbw: f64,
    pub free_space_path_loss_db: f64,
    pub atmospheric_loss_db: f64,
    pub rain_loss_db: f64,
    pub propagation_factor_db: f64,
    pub coherent_integration_gain_db: f64,
    pub above_horizon: bool,
    pub detections_this_frame: usize,
    /// Wall-clock cost of synthesising this frame (ms).
    pub frame_compute_ms: f64,
    /// Achieved scan rate (Hz) — may be below the requested rate when
    /// synthesis is slower than the frame interval.
    pub scan_rate_hz: f64,
}

/// A quantised range-Doppler magnitude image. `cells` is row-major
/// `[doppler_bin][range_bin]`; each `u8` dequantises to dB via
/// `db = db_min + (cell / 255) * (db_max - db_min)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeDopplerGrid {
    pub range_bins: usize,
    pub doppler_bins: usize,
    pub db_min: f32,
    pub db_max: f32,
    pub cells: Vec<u8>,
}

/// A single quantised micro-Doppler spectrum column (one scan dwell).
/// The browser owns the scrolling waterfall history.
#[derive(Debug, Clone, PartialEq)]
pub struct MicroDopplerColumn {
    pub bins: usize,
    pub db_min: f32,
    pub db_max: f32,
    /// Doppler half-width (Hz); the column spans ±this value.
    pub doppler_max_hz: f64,
    pub column: Vec<u8>,
}

/// A frame queued for fan-out on the engine broadcast channel.
#[derive(Debug, Clone)]
pub enum OutboundFrame {
    Control(ControlFrame),
    Scan(ScanFrame),
}
