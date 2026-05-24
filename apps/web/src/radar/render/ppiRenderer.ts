// Plan-Position-Indicator renderer. The coordinate transforms are pure
// and unit-tested; `drawPpi` is browser-only canvas drawing.

import type { ScanFrame, SessionInfo } from '../radarContract';

export interface PpiViewport {
  cx: number;
  cy: number;
  radius: number;
  rangeMax: number;
}

/** Screen viewport for a square-ish PPI within a `w`×`h` canvas. */
export function computeViewport(w: number, h: number, rangeMax: number): PpiViewport {
  const radius = Math.max(8, Math.min(w, h) / 2 - 26);
  return { cx: w / 2, cy: h / 2, radius, rangeMax: Math.max(1, rangeMax) };
}

/** Polar (range m, azimuth deg, 0 = North, clockwise) → screen pixels. */
export function polarToXY(rangeM: number, azimuthDeg: number, vp: PpiViewport): {
  x: number;
  y: number;
} {
  const rr = Math.min(rangeM / vp.rangeMax, 1) * vp.radius;
  const a = (azimuthDeg * Math.PI) / 180;
  return { x: vp.cx + rr * Math.sin(a), y: vp.cy - rr * Math.cos(a) };
}

/** Screen pixels → polar; azimuth normalised to 0..360. */
export function xyToPolar(x: number, y: number, vp: PpiViewport): {
  rangeM: number;
  azimuthDeg: number;
} {
  const dx = x - vp.cx;
  const dy = y - vp.cy;
  const r = Math.sqrt(dx * dx + dy * dy);
  let az = (Math.atan2(dx, -dy) * 180) / Math.PI;
  if (az < 0) az += 360;
  return { rangeM: (r / vp.radius) * vp.rangeMax, azimuthDeg: az };
}

/** Index of the PPI blip nearest a screen point, or -1 if none within tol. */
export function nearestBlip(
  scan: ScanFrame,
  x: number,
  y: number,
  vp: PpiViewport,
  tolerancePx = 18,
): number {
  let best = -1;
  let bestDist = tolerancePx;
  scan.meta.ppi.forEach((blip, i) => {
    const p = polarToXY(blip.range_m, blip.azimuth_deg, vp);
    const d = Math.hypot(p.x - x, p.y - y);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  });
  return best;
}

export interface PpiTheme {
  grid: string;
  sweep: string;
  blip: string;
  blipMiss: string;
  selected: string;
  label: string;
}

export const PHOSPHOR_THEME: PpiTheme = {
  grid: 'rgba(40, 224, 200, 0.30)',
  sweep: 'rgba(40, 224, 200, 0.55)',
  blip: '#3ce0c8',
  blipMiss: '#ffb44a',
  selected: '#ff5d5d',
  label: 'rgba(150, 198, 200, 0.7)',
};

/** Render one PPI frame with phosphor afterglow. */
export function drawPpi(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  scan: ScanFrame | null,
  session: SessionInfo | null,
  selectedTrackId: number | null,
  theme: PpiTheme = PHOSPHOR_THEME,
): void {
  // Phosphor afterglow: dim the previous frame instead of clearing.
  ctx.fillStyle = 'rgba(5, 9, 13, 0.22)';
  ctx.fillRect(0, 0, w, h);

  const rangeMax = session?.range_max_m ?? 20000;
  const vp = computeViewport(w, h, rangeMax);

  // Range rings + range labels.
  ctx.lineWidth = 1;
  ctx.strokeStyle = theme.grid;
  ctx.fillStyle = theme.label;
  ctx.font = '10px ui-monospace, monospace';
  ctx.textAlign = 'center';
  for (let ring = 1; ring <= 4; ring++) {
    const rr = (vp.radius * ring) / 4;
    ctx.beginPath();
    ctx.arc(vp.cx, vp.cy, rr, 0, Math.PI * 2);
    ctx.stroke();
    const km = ((rangeMax * ring) / 4 / 1000).toFixed(1);
    ctx.fillText(`${km} km`, vp.cx, vp.cy - rr - 3);
  }

  // Azimuth spokes every 30°.
  for (let deg = 0; deg < 360; deg += 30) {
    const outer = polarToXY(rangeMax, deg, vp);
    ctx.beginPath();
    ctx.moveTo(vp.cx, vp.cy);
    ctx.lineTo(outer.x, outer.y);
    ctx.stroke();
  }

  // Rotating sweep wedge.
  const beam = scan?.meta.beam_azimuth_deg ?? 0;
  const beamRad = (beam * Math.PI) / 180 - Math.PI / 2;
  const grad = ctx.createConicGradient(beamRad, vp.cx, vp.cy);
  grad.addColorStop(0, theme.sweep);
  grad.addColorStop(0.08, 'rgba(40, 224, 200, 0)');
  grad.addColorStop(1, 'rgba(40, 224, 200, 0)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(vp.cx, vp.cy, vp.radius, 0, Math.PI * 2);
  ctx.fill();

  // Target blips.
  if (scan) {
    for (const blip of scan.meta.ppi) {
      const p = polarToXY(blip.range_m, blip.azimuth_deg, vp);
      const strength = Math.max(0.25, Math.min(1, (blip.snr_db + 10) / 40));
      const size = 3 + strength * 5;
      ctx.beginPath();
      ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
      ctx.fillStyle = blip.detected ? theme.blip : theme.blipMiss;
      ctx.globalAlpha = blip.detected ? 1 : 0.7;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    for (const track of scan.meta.tracks) {
      if (track.track_id !== selectedTrackId) continue;
      const p = polarToXY(track.range_m, track.azimuth_deg, vp);
      ctx.strokeStyle = theme.selected;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 13, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}
