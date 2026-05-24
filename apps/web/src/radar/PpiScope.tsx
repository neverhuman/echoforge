import { useCallback, useRef } from 'react';
import { radarStore } from './radarStore';
import { computeViewport, drawPpi, polarToXY } from './render/ppiRenderer';
import { useAnimationFrame } from './render/useAnimationFrame';
import { useCanvasSize } from './render/useCanvasSize';

/** Plan-Position-Indicator scope — rotating sweep, range rings, blips. */
export default function PpiScope() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useCanvasSize(canvasRef);

  useAnimationFrame(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const snap = radarStore.getSnapshot();
    drawPpi(
      ctx,
      canvas.width,
      canvas.height,
      radarStore.latestScan,
      snap.session,
      snap.selectedTrackId,
    );
  });

  const handleClick = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const scan = radarStore.latestScan;
    if (!canvas || !scan) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / Math.max(1, rect.width);
    const scaleY = canvas.height / Math.max(1, rect.height);
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    const vp = computeViewport(
      canvas.width,
      canvas.height,
      radarStore.getSnapshot().session?.range_max_m ?? 20000,
    );
    let bestId: number | null = null;
    let bestDist = 28;
    for (const track of scan.meta.tracks) {
      const p = polarToXY(track.range_m, track.azimuth_deg, vp);
      const dist = Math.hypot(p.x - x, p.y - y);
      if (dist < bestDist) {
        bestDist = dist;
        bestId = track.track_id;
      }
    }
    radarStore.setSelectedTrack(bestId);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="radar-canvas"
      data-testid="ppi-scope"
      onClick={handleClick}
    />
  );
}
