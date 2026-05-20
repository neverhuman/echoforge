// TypeScript mirror of the Rust `echoforge-studio` radar stream contract
// (`crates/echoforge-studio/src/stream/frames.rs`). Field names match the
// serde snake_case wire format exactly.

export const SCAN_FRAME_MAGIC = 0xec40_0001;
export const SCAN_HEADER_LEN = 34;

export interface ScenarioSummary {
  id: string;
  label: string;
  description: string;
}

export interface SessionInfo {
  session_id: number;
  source: string;
  scenario_id: string;
  scenario_label: string;
  frame_rate_hz: number;
  running: boolean;
  paused: boolean;
  playback_speed: number;
  range_max_m: number;
  doppler_max_hz: number;
  rd_range_bins: number;
  rd_doppler_bins: number;
  spectrogram_bins: number;
  available_scenarios: ScenarioSummary[];
  schema_version: number;
}

export interface StatusFrame {
  level: string;
  code: string;
  message: string;
  dropped_frames?: number | null;
}

/** A control-plane message — arrives as a JSON text WebSocket frame. */
export type ControlFrame =
  | ({ type: 'session_info' } & SessionInfo)
  | ({ type: 'status' } & StatusFrame);

export interface PpiBlip {
  entity_id: number;
  range_m: number;
  azimuth_deg: number;
  amplitude_db: number;
  snr_db: number;
  detected: boolean;
  class_label: string;
}

export interface RdDetection {
  range_bin: number;
  range_m: number;
  doppler_bin: number;
  magnitude_db: number;
  snr_db: number;
}

export interface TrackRow {
  track_id: number;
  range_m: number;
  azimuth_deg: number;
  radial_velocity_mps: number;
  snr_db: number;
  confidence: number;
  class_label: string;
  age_frames: number;
}

export interface Telemetry {
  snr_db: number;
  received_power_dbw: number;
  noise_power_dbw: number;
  free_space_path_loss_db: number;
  atmospheric_loss_db: number;
  rain_loss_db: number;
  propagation_factor_db: number;
  coherent_integration_gain_db: number;
  above_horizon: boolean;
  detections_this_frame: number;
  frame_compute_ms: number;
  scan_rate_hz: number;
}

export interface ScanMeta {
  frame_index: number;
  sim_time_s: number;
  wall_time_ms: number;
  beam_azimuth_deg: number;
  ppi: PpiBlip[];
  detections: RdDetection[];
  tracks: TrackRow[];
  telemetry: Telemetry;
}

/** A quantised range-Doppler image; `cells` is row-major [doppler][range]. */
export interface RangeDopplerGrid {
  range_bins: number;
  doppler_bins: number;
  db_min: number;
  db_max: number;
  cells: Uint8Array;
}

/** One quantised micro-Doppler spectrum column. */
export interface MicroDopplerColumn {
  bins: number;
  db_min: number;
  db_max: number;
  doppler_max_hz: number;
  column: Uint8Array;
}

/** One complete radar scan — arrives as a binary WebSocket frame. */
export interface ScanFrame {
  meta: ScanMeta;
  range_doppler: RangeDopplerGrid;
  micro_doppler: MicroDopplerColumn;
}

/** Outbound control commands posted to the `/api/sim/*` REST surface. */
export interface RadarParamPatch {
  transmit_power_w?: number;
  tx_gain_dbi?: number;
  rx_gain_dbi?: number;
  noise_figure_db?: number;
  cfar_pfa?: number;
  cfar_training_cells?: number;
  cfar_guard_cells?: number;
  rain_rate_mm_per_h?: number;
  atmospheric_one_way_db_per_km?: number;
}
