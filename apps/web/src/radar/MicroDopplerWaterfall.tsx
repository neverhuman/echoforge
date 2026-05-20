import { useRef } from 'react';
import { radarStore } from './radarStore';
import { WATERFALL_LUT } from './render/colorMaps';
import { useAnimationFrame } from './render/useAnimationFrame';
import { useCanvasSize } from './render/useCanvasSize';
import { repaintWaterfall, scrollWaterfall } from './render/waterfallRenderer';

/** Scrolling micro-Doppler spectrogram (time-frequency waterfall). */
export default function MicroDopplerWaterfall() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastSeqRef = useRef(-1);
  const sizeRef = useRef({ w: 0, h: 0 });
  useCanvasSize(canvasRef);

  useAnimationFrame(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // A resize clears the backing store — repaint history from the ring.
    if (sizeRef.current.w !== canvas.width || sizeRef.current.h !== canvas.height) {
      sizeRef.current = { w: canvas.width, h: canvas.height };
      repaintWaterfall(ctx, canvas.width, canvas.height, radarStore.waterfall, WATERFALL_LUT);
      lastSeqRef.current = radarStore.scanSeq;
      return;
    }

    if (radarStore.scanSeq === lastSeqRef.current) return;
    lastSeqRef.current = radarStore.scanSeq;
    const scan = radarStore.latestScan;
    if (!scan) return;
    scrollWaterfall(
      ctx,
      canvas.width,
      canvas.height,
      scan.micro_doppler.column,
      WATERFALL_LUT,
    );
  });

  return (
    <canvas
      ref={canvasRef}
      className="radar-canvas"
      data-testid="micro-doppler-waterfall"
    />
  );
}
