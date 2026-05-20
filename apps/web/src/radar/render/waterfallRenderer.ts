// Micro-Doppler waterfall renderer — a self-blitting scrolling spectrogram.

import { lutColor } from './colorMaps';
import type { ColumnRing } from './ringBuffer';

function paintColumn(
  ctx: CanvasRenderingContext2D,
  x: number,
  h: number,
  column: Uint8Array,
  lut: Uint8ClampedArray,
): void {
  const img = ctx.createImageData(1, h);
  const n = column.length;
  for (let y = 0; y < h; y++) {
    // Canvas top = high (positive) Doppler, bottom = negative.
    const bin = n > 1 ? Math.round((1 - y / (h - 1)) * (n - 1)) : 0;
    const [r, g, b] = lutColor(lut, column[bin] ?? 0);
    const p = y * 4;
    img.data[p] = r;
    img.data[p + 1] = g;
    img.data[p + 2] = b;
    img.data[p + 3] = 255;
  }
  ctx.putImageData(img, x, 0);
}

/** Scroll the waterfall left one pixel and paint the newest column. */
export function scrollWaterfall(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  column: Uint8Array,
  lut: Uint8ClampedArray,
): void {
  if (w < 2 || h < 1) return;
  ctx.drawImage(ctx.canvas, 1, 0, w - 1, h, 0, 0, w - 1, h);
  paintColumn(ctx, w - 1, h, column, lut);
}

/** Repaint the whole waterfall from ring history (e.g. after a resize). */
export function repaintWaterfall(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  ring: ColumnRing,
  lut: Uint8ClampedArray,
): void {
  ctx.clearRect(0, 0, w, h);
  const columns = Math.min(w, ring.length);
  for (let age = 0; age < columns; age++) {
    const col = ring.at(age);
    if (col) {
      paintColumn(ctx, w - 1 - age, h, col, lut);
    }
  }
}
