// Range-Doppler heat-map renderer. The grid is small (<=256x256), so a
// Canvas-2D `putImageData` upscale holds 60fps without a WebGL path.

import { lutColor } from './colorMaps';
import type { RangeDopplerGrid, RdDetection } from '../radarContract';

/**
 * Draw the range-Doppler grid scaled into the display context.
 * `scratch` is a reusable offscreen canvas resized to the native grid.
 */
export function drawRangeDoppler(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  grid: RangeDopplerGrid,
  detections: RdDetection[],
  scratch: HTMLCanvasElement,
  lut: Uint8ClampedArray,
): void {
  const rb = grid.range_bins;
  const db = grid.doppler_bins;
  if (rb < 1 || db < 1) {
    ctx.clearRect(0, 0, w, h);
    return;
  }
  scratch.width = rb;
  scratch.height = db;
  const sctx = scratch.getContext('2d');
  if (!sctx) return;

  const img = sctx.createImageData(rb, db);
  for (let d = 0; d < db; d++) {
    // Doppler bin 0 (most negative) at the bottom row.
    const rowBase = (db - 1 - d) * rb;
    for (let r = 0; r < rb; r++) {
      const [cr, cg, cb] = lutColor(lut, grid.cells[d * rb + r]);
      const p = (rowBase + r) * 4;
      img.data[p] = cr;
      img.data[p + 1] = cg;
      img.data[p + 2] = cb;
      img.data[p + 3] = 255;
    }
  }
  sctx.putImageData(img, 0, 0);

  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(scratch, 0, 0, rb, db, 0, 0, w, h);

  // CFAR detection markers.
  ctx.strokeStyle = '#ff5d5d';
  ctx.lineWidth = 1.5;
  for (const det of detections) {
    const x = ((det.range_bin + 0.5) / rb) * w;
    const y = ((db - 1 - det.doppler_bin + 0.5) / db) * h;
    ctx.strokeRect(x - 6, y - 6, 12, 12);
  }
}
