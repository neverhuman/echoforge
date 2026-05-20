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
  if (tag !== 'session_info' && tag !== 'status') {
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
  let meta: ScanMeta;
  try {
    meta = JSON.parse(jsonText) as ScanMeta;
  } catch (err) {
    throw new FrameDecodeError(`scan meta is not valid JSON: ${String(err)}`);
  }

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
