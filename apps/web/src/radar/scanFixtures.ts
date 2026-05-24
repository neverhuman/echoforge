// Test fixtures: build binary scan-frame buffers matching the Rust
// encoder layout in `crates/echoforge-studio/src/stream/encode.rs`.

import type { ScanMeta } from './radarContract';

export interface ScanBufferParts {
  meta: ScanMeta;
  rd: { range: number; doppler: number; dbMin: number; dbMax: number; cells: number[] };
  spec: { bins: number; dbMin: number; dbMax: number; dopplerHz: number; column: number[] };
}

/** Build a binary scan-frame buffer in the on-wire layout. */
export function buildScanBuffer(parts: ScanBufferParts): ArrayBuffer {
  const json = new TextEncoder().encode(JSON.stringify(parts.meta));
  const total = 34 + json.length + parts.rd.cells.length + parts.spec.column.length;
  const buffer = new ArrayBuffer(total);
  const view = new DataView(buffer);
  view.setUint32(0, 0xec40_0001, true);
  view.setUint32(4, json.length, true);
  view.setUint16(8, parts.rd.range, true);
  view.setUint16(10, parts.rd.doppler, true);
  view.setFloat32(12, parts.rd.dbMin, true);
  view.setFloat32(16, parts.rd.dbMax, true);
  view.setUint16(20, parts.spec.bins, true);
  view.setFloat32(22, parts.spec.dbMin, true);
  view.setFloat32(26, parts.spec.dbMax, true);
  view.setFloat32(30, parts.spec.dopplerHz, true);
  const bytes = new Uint8Array(buffer);
  bytes.set(json, 34);
  bytes.set(parts.rd.cells, 34 + json.length);
  bytes.set(parts.spec.column, 34 + json.length + parts.rd.cells.length);
  return buffer;
}

/** A representative `ScanMeta` for tests. */
export function sampleMeta(overrides: Partial<ScanMeta> = {}): ScanMeta {
  return {
    frame_index: 12,
    sim_time_s: 0.8,
    wall_time_ms: 1_700_000_000_000,
    beam_azimuth_deg: 135,
    ppi: [],
    detections: [],
    tracks: [],
    telemetry: {
      snr_db: 18,
      received_power_dbw: -92,
      noise_power_dbw: -130,
      free_space_path_loss_db: 140,
      atmospheric_loss_db: 0.5,
      rain_loss_db: 0,
      propagation_factor_db: 0,
      coherent_integration_gain_db: 15,
      above_horizon: true,
      detections_this_frame: 0,
      frame_compute_ms: 1.5,
      scan_rate_hz: 15,
    },
    ...overrides,
  };
}

/** A small but complete scan buffer for store / socket tests. */
export function sampleScanBuffer(metaOverrides: Partial<ScanMeta> = {}): ArrayBuffer {
  return buildScanBuffer({
    meta: sampleMeta(metaOverrides),
    rd: { range: 4, doppler: 2, dbMin: -60, dbMax: 0, cells: [1, 2, 3, 4, 5, 6, 7, 8] },
    spec: { bins: 4, dbMin: -60, dbMax: 0, dopplerHz: 500, column: [9, 10, 11, 12] },
  });
}
