import { describe, expect, test } from 'vitest';
import { buildLut, lutColor, RANGE_DOPPLER_LUT, WATERFALL_LUT } from './colorMaps';

describe('buildLut', () => {
  test('produces a 256-entry RGB table', () => {
    const lut = buildLut([
      { t: 0, rgb: [0, 0, 0] },
      { t: 1, rgb: [255, 255, 255] },
    ]);
    expect(lut.length).toBe(256 * 3);
  });

  test('anchors the endpoints to the first and last stop', () => {
    const lut = buildLut([
      { t: 0, rgb: [10, 20, 30] },
      { t: 1, rgb: [200, 210, 220] },
    ]);
    expect([lut[0], lut[1], lut[2]]).toEqual([10, 20, 30]);
    expect([lut[765], lut[766], lut[767]]).toEqual([200, 210, 220]);
  });

  test('interpolates monotonically between stops', () => {
    const lut = buildLut([
      { t: 0, rgb: [0, 0, 0] },
      { t: 1, rgb: [255, 255, 255] },
    ]);
    for (let i = 1; i < 256; i++) {
      expect(lut[i * 3]).toBeGreaterThanOrEqual(lut[(i - 1) * 3]);
    }
  });

  test('rejects a single-stop map', () => {
    expect(() => buildLut([{ t: 0, rgb: [0, 0, 0] }])).toThrow();
  });
});

describe('lutColor', () => {
  test('clamps out-of-range sample values', () => {
    expect(lutColor(RANGE_DOPPLER_LUT, -50)).toEqual(lutColor(RANGE_DOPPLER_LUT, 0));
    expect(lutColor(WATERFALL_LUT, 999)).toEqual(lutColor(WATERFALL_LUT, 255));
  });
});
