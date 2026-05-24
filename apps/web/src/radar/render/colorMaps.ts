// Colour-map lookup tables for the radar heat displays. Pure functions —
// unit-tested under the node test environment.

export type Rgb = [number, number, number];

interface Stop {
  t: number;
  rgb: Rgb;
}

/** Build a 256-entry RGB lookup table by linear interpolation of stops. */
export function buildLut(stops: Stop[]): Uint8ClampedArray {
  if (stops.length < 2) {
    throw new Error('a colour map needs at least two stops');
  }
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let lo = stops[0];
    let hi = stops[stops.length - 1];
    for (let s = 0; s < stops.length - 1; s++) {
      if (t >= stops[s].t && t <= stops[s + 1].t) {
        lo = stops[s];
        hi = stops[s + 1];
        break;
      }
    }
    const span = hi.t - lo.t || 1;
    const f = (t - lo.t) / span;
    lut[i * 3 + 0] = lo.rgb[0] + f * (hi.rgb[0] - lo.rgb[0]);
    lut[i * 3 + 1] = lo.rgb[1] + f * (hi.rgb[1] - lo.rgb[1]);
    lut[i * 3 + 2] = lo.rgb[2] + f * (hi.rgb[2] - lo.rgb[2]);
  }
  return lut;
}

/** Read a colour from a LUT for a 0..255 sample value. */
export function lutColor(lut: Uint8ClampedArray, value: number): Rgb {
  const idx = (value < 0 ? 0 : value > 255 ? 255 : value | 0) * 3;
  return [lut[idx], lut[idx + 1], lut[idx + 2]];
}

/** Range-Doppler heat map — deep navy through cyan to white. */
export const RANGE_DOPPLER_LUT = buildLut([
  { t: 0.0, rgb: [4, 8, 14] },
  { t: 0.35, rgb: [12, 46, 92] },
  { t: 0.6, rgb: [22, 132, 170] },
  { t: 0.82, rgb: [60, 224, 200] },
  { t: 1.0, rgb: [236, 255, 252] },
]);

/** Micro-Doppler waterfall — black through magenta and amber to white. */
export const WATERFALL_LUT = buildLut([
  { t: 0.0, rgb: [3, 4, 10] },
  { t: 0.35, rgb: [62, 18, 96] },
  { t: 0.62, rgb: [190, 54, 96] },
  { t: 0.83, rgb: [248, 162, 58] },
  { t: 1.0, rgb: [255, 248, 222] },
]);
