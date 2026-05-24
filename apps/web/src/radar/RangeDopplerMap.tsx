import { useRef } from 'react';
import { radarStore } from './radarStore';
import { RANGE_DOPPLER_LUT } from './render/colorMaps';
import { drawRangeDoppler } from './render/rangeDopplerRenderer';
import { useAnimationFrame } from './render/useAnimationFrame';
import { useCanvasSize } from './render/useCanvasSize';

/** Range-Doppler heat map with CFAR detections overlaid. */
export default function RangeDopplerMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scratchRef = useRef<HTMLCanvasElement | null>(null);
  const lastSeqRef = useRef(-1);
  useCanvasSize(canvasRef);

  useAnimationFrame(() => {
    const canvas = canvasRef.current;
    const scan = radarStore.latestScan;
    if (!canvas || !scan) return;
    if (radarStore.scanSeq === lastSeqRef.current) return;
    lastSeqRef.current = radarStore.scanSeq;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (!scratchRef.current) {
      scratchRef.current = document.createElement('canvas');
    }
    drawRangeDoppler(
      ctx,
      canvas.width,
      canvas.height,
      scan.range_doppler,
      scan.meta.detections,
      scratchRef.current,
      RANGE_DOPPLER_LUT,
    );
  });

  return (
    <canvas
      ref={canvasRef}
      className="radar-canvas"
      data-testid="range-doppler-map"
    />
  );
}
