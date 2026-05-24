// Decoders for the two radar wire message families: JSON text control
// frames and binary scan frames. See `radarContract.ts` and the Rust
// encoder in `crates/echoforge-studio/src/stream/encode.rs`.

import {
  SCAN_FRAME_MAGIC,
  SCAN_HEADER_LEN,
  type ControlFrame,
  type ScanFrame,
  type ScanMeta,
} from './radarContract';

export class FrameDecodeError extends Error {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isBoolean(value: unknown): value is boolean {
  return typeof value === 'boolean';
}

function isString(value: unknown): value is string {
  return typeof value === 'string';
}

function isPpiBlip(value: unknown): value is ScanMeta['ppi'][number] {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isFiniteNumber(value.entity_id) &&
    isFiniteNumber(value.range_m) &&
    isFiniteNumber(value.azimuth_deg) &&
    isFiniteNumber(value.amplitude_db) &&
    isFiniteNumber(value.snr_db) &&
    isBoolean(value.detected) &&
    isString(value.class_label)
  );
}

function isRdDetection(value: unknown): value is ScanMeta['detections'][number] {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isFiniteNumber(value.range_bin) &&
    isFiniteNumber(value.range_m) &&
    isFiniteNumber(value.doppler_bin) &&
    isFiniteNumber(value.magnitude_db) &&
    isFiniteNumber(value.snr_db)
  );
}

function isTrackRow(value: unknown): value is ScanMeta['tracks'][number] {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isFiniteNumber(value.track_id) &&
    isFiniteNumber(value.range_m) &&
    isFiniteNumber(value.azimuth_deg) &&
    isFiniteNumber(value.radial_velocity_mps) &&
    isFiniteNumber(value.snr_db) &&
    isFiniteNumber(value.confidence) &&
    isString(value.class_label) &&
    isFiniteNumber(value.age_frames)
  );
}

function isTelemetry(value: unknown): value is ScanMeta['telemetry'] {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isFiniteNumber(value.snr_db) &&
    isFiniteNumber(value.received_power_dbw) &&
    isFiniteNumber(value.noise_power_dbw) &&
    isFiniteNumber(value.free_space_path_loss_db) &&
    isFiniteNumber(value.atmospheric_loss_db) &&
    isFiniteNumber(value.rain_loss_db) &&
    isFiniteNumber(value.propagation_factor_db) &&
    isFiniteNumber(value.coherent_integration_gain_db) &&
    isBoolean(value.above_horizon) &&
    isFiniteNumber(value.detections_this_frame) &&
    isFiniteNumber(value.frame_compute_ms) &&
    isFiniteNumber(value.scan_rate_hz)
  );
}

function isScanMeta(value: unknown): value is ScanMeta {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isFiniteNumber(value.frame_index) &&
    isFiniteNumber(value.sim_time_s) &&
    isFiniteNumber(value.wall_time_ms) &&
    isFiniteNumber(value.beam_azimuth_deg) &&
    Array.isArray(value.ppi) &&
    value.ppi.every(isPpiBlip) &&
    Array.isArray(value.detections) &&
    value.detections.every(isRdDetection) &&
    Array.isArray(value.tracks) &&
    value.tracks.every(isTrackRow) &&
    isTelemetry(value.telemetry)
  );
}

function parseScanMeta(value: unknown): ScanMeta {
  if (!isScanMeta(value)) {
    throw new FrameDecodeError('scan meta shape does not match ScanMeta');
  }
  return value;
}

/** Parse a JSON text WebSocket frame into a `ControlFrame`. */
export function decodeControlFrame(text: string): ControlFrame {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new FrameDecodeError(`control frame is not valid JSON: ${String(err)}`);
  }
  if (
    typeof parsed !== 'object' ||
    parsed === null ||
    typeof (parsed as { type?: unknown }).type !== 'string'
  ) {
    throw new FrameDecodeError('control frame is missing a string `type` tag');
  }
  const tag = (parsed as { type: string }).type;
  if (
    tag !== 'session_info' &&
    tag !== 'status' &&
    tag !== 'run_lifecycle' &&
    tag !== 'artifact_ready' &&
    tag !== 'validation_status' &&
    tag !== 'backpressure'
  ) {
    throw new FrameDecodeError(`unknown control frame type: ${tag}`);
  }
  return parsed as ControlFrame;
}

/** Parse a binary WebSocket frame into a `ScanFrame`. */
export function decodeScanFrame(buffer: ArrayBuffer): ScanFrame {
  if (buffer.byteLength < SCAN_HEADER_LEN) {
    throw new FrameDecodeError(
      `scan frame too short: ${buffer.byteLength} < ${SCAN_HEADER_LEN}`,
    );
  }
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== SCAN_FRAME_MAGIC) {
    throw new FrameDecodeError(`bad scan frame magic: 0x${magic.toString(16)}`);
  }
  const jsonLen = view.getUint32(4, true);
  const rdRange = view.getUint16(8, true);
  const rdDoppler = view.getUint16(10, true);
  const rdDbMin = view.getFloat32(12, true);
  const rdDbMax = view.getFloat32(16, true);
  const specBins = view.getUint16(20, true);
  const specDbMin = view.getFloat32(22, true);
  const specDbMax = view.getFloat32(26, true);
  const specDopplerHz = view.getFloat32(30, true);

  const rdLen = rdRange * rdDoppler;
  const total = SCAN_HEADER_LEN + jsonLen + rdLen + specBins;
  if (buffer.byteLength < total) {
    throw new FrameDecodeError(
      `scan frame truncated: ${buffer.byteLength} < ${total}`,
    );
  }

  const jsonStart = SCAN_HEADER_LEN;
  const rdStart = jsonStart + jsonLen;
  const specStart = rdStart + rdLen;

  const jsonText = new TextDecoder().decode(
    new Uint8Array(buffer, jsonStart, jsonLen),
  );
  let parsedMeta: unknown;
  try {
    parsedMeta = JSON.parse(jsonText);
  } catch (err) {
    throw new FrameDecodeError(`scan meta is not valid JSON: ${String(err)}`);
  }
  const meta = parseScanMeta(parsedMeta);

  return {
    meta,
    range_doppler: {
      range_bins: rdRange,
      doppler_bins: rdDoppler,
      db_min: rdDbMin,
      db_max: rdDbMax,
      cells: new Uint8Array(buffer.slice(rdStart, rdStart + rdLen)),
    },
    micro_doppler: {
      bins: specBins,
      db_min: specDbMin,
      db_max: specDbMax,
      doppler_max_hz: specDopplerHz,
      column: new Uint8Array(buffer.slice(specStart, specStart + specBins)),
    },
  };
}
