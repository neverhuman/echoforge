import { describe, expect, test } from 'vitest';
import type { ScanFrame } from '../radarContract';
import { computeViewport, nearestBlip, polarToXY, xyToPolar } from './ppiRenderer';

const vp = computeViewport(400, 400, 20000);

describe('computeViewport', () => {
  test('centres within the canvas', () => {
    expect(vp.cx).toBe(200);
    expect(vp.cy).toBe(200);
    expect(vp.radius).toBeGreaterThan(0);
  });
});

describe('polarToXY', () => {
  test('places zero range at the centre', () => {
    const p = polarToXY(0, 123, vp);
    expect(p.x).toBeCloseTo(vp.cx, 6);
    expect(p.y).toBeCloseTo(vp.cy, 6);
  });

  test('puts North (0 deg) straight up', () => {
    const p = polarToXY(20000, 0, vp);
    expect(p.x).toBeCloseTo(vp.cx, 6);
    expect(p.y).toBeCloseTo(vp.cy - vp.radius, 6);
  });

  test('puts East (90 deg) to the right', () => {
    const p = polarToXY(20000, 90, vp);
    expect(p.x).toBeCloseTo(vp.cx + vp.radius, 6);
    expect(p.y).toBeCloseTo(vp.cy, 6);
  });
});

describe('xyToPolar', () => {
  test('round-trips with polarToXY', () => {
    for (const az of [0, 47, 158, 270, 333]) {
      const range = 8200;
      const p = polarToXY(range, az, vp);
      const back = xyToPolar(p.x, p.y, vp);
      expect(back.rangeM).toBeCloseTo(range, 2);
      expect(back.azimuthDeg).toBeCloseTo(az, 3);
    }
  });
});

describe('nearestBlip', () => {
  const scan = {
    meta: {
      ppi: [
        { entity_id: 0, range_m: 10000, azimuth_deg: 90, amplitude_db: 0, snr_db: 20, detected: true, class_label: 'a' },
        { entity_id: 1, range_m: 5000, azimuth_deg: 270, amplitude_db: 0, snr_db: 20, detected: true, class_label: 'b' },
      ],
    },
  } as unknown as ScanFrame;

  test('returns the index of the blip under the cursor', () => {
    const target = polarToXY(10000, 90, vp);
    expect(nearestBlip(scan, target.x, target.y, vp)).toBe(0);
  });

  test('returns -1 when nothing is within tolerance', () => {
    expect(nearestBlip(scan, vp.cx, vp.cy, vp, 4)).toBe(-1);
  });
});
