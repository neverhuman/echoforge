// jankurai:allow HLT-006-DIRECT-DB-WRONG-LAYER this file IS the data-access service test suite
import { describe, expect, test } from 'vitest';
import {
  meshListUrl,
  meshStlUrl,
  parseBinaryStl,
} from './meshService';

/**
 * Build a minimal binary STL payload for parseBinaryStl tests.
 */
function buildStl(triangles: number[][][]): ArrayBuffer {
  const count = triangles.length;
  const buf = new ArrayBuffer(84 + count * 50);
  const dv = new DataView(buf);
  dv.setUint32(80, count, true);
  for (let i = 0; i < count; i++) {
    const tri = triangles[i];
    const base = 84 + i * 50;
    for (let v = 0; v < 3; v++) {
      const vOff = base + 12 + v * 12;
      dv.setFloat32(vOff + 0, tri[v][0], true);
      dv.setFloat32(vOff + 4, tri[v][1], true);
      dv.setFloat32(vOff + 8, tri[v][2], true);
    }
  }
  return buf;
}

describe('meshService URL helpers', () => {
  test('meshListUrl points at /api/mesh/list', () => {
    expect(meshListUrl()).toBe('/api/mesh/list');
  });

  test('meshStlUrl URL-encodes the primitive slug', () => {
    expect(meshStlUrl('sphere')).toBe('/api/mesh/sphere');
    expect(meshStlUrl('odd name')).toBe('/api/mesh/odd%20name');
  });
});

describe('parseBinaryStl', () => {
  test('returns triangle count and bounding box for a non-empty mesh', () => {
    const buf = buildStl([
      [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
      ],
      [
        [0, 0, 0],
        [0, 0, 1],
        [-1, -1, 0],
      ],
    ]);
    const stats = parseBinaryStl(buf);
    expect(stats.triangleCount).toBe(2);
    expect(stats.boundingBox.min).toEqual([-1, -1, 0]);
    expect(stats.boundingBox.max).toEqual([1, 1, 1]);
  });

  test('rejects truncated payloads', () => {
    expect(() => parseBinaryStl(new ArrayBuffer(10))).toThrow(/too small/);
  });

  test('rejects header / payload length mismatch', () => {
    const bad = new ArrayBuffer(84);
    new DataView(bad).setUint32(80, 1, true);
    expect(() => parseBinaryStl(bad)).toThrow(/length mismatch/);
  });
});
