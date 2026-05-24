import { describe, expect, test } from 'vitest';
import { decodeControlFrame, decodeScanFrame, FrameDecodeError } from './frameCodec';
import { buildScanBuffer, sampleMeta } from './scanFixtures';

describe('decodeControlFrame', () => {
  test('parses a session_info frame', () => {
    const frame = decodeControlFrame('{"type":"session_info","scenario_id":"x"}');
    expect(frame.type).toBe('session_info');
  });

  test('parses a status frame', () => {
    const frame = decodeControlFrame('{"type":"status","code":"started"}');
    expect(frame.type).toBe('status');
  });

  test('rejects invalid JSON', () => {
    expect(() => decodeControlFrame('{not json')).toThrow(FrameDecodeError);
  });

  test('rejects a missing type tag', () => {
    expect(() => decodeControlFrame('{"scenario_id":"x"}')).toThrow(FrameDecodeError);
  });

  test('rejects an unknown type tag', () => {
    expect(() => decodeControlFrame('{"type":"mystery"}')).toThrow(FrameDecodeError);
  });
});

describe('decodeScanFrame', () => {
  test('round-trips a scan frame from the wire layout', () => {
    const buffer = buildScanBuffer({
      meta: sampleMeta(),
      rd: { range: 3, doppler: 2, dbMin: -60, dbMax: 0, cells: [1, 2, 3, 4, 5, 6] },
      spec: { bins: 4, dbMin: -60, dbMax: 0, dopplerHz: 500, column: [7, 8, 9, 10] },
    });
    const scan = decodeScanFrame(buffer);
    expect(scan.meta.frame_index).toBe(12);
    expect(scan.range_doppler.range_bins).toBe(3);
    expect(scan.range_doppler.doppler_bins).toBe(2);
    expect(Array.from(scan.range_doppler.cells)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(scan.micro_doppler.bins).toBe(4);
    expect(Array.from(scan.micro_doppler.column)).toEqual([7, 8, 9, 10]);
    expect(scan.micro_doppler.doppler_max_hz).toBeCloseTo(500, 3);
  });

  test('rejects a bad magic number', () => {
    const buffer = buildScanBuffer({
      meta: sampleMeta(),
      rd: { range: 1, doppler: 1, dbMin: 0, dbMax: 1, cells: [9] },
      spec: { bins: 1, dbMin: 0, dbMax: 1, dopplerHz: 1, column: [9] },
    });
    new DataView(buffer).setUint32(0, 0xdead_beef, true);
    expect(() => decodeScanFrame(buffer)).toThrow(FrameDecodeError);
  });

  test('rejects a truncated buffer', () => {
    expect(() => decodeScanFrame(new ArrayBuffer(8))).toThrow(FrameDecodeError);
  });
});
