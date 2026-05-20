// Keep a canvas backing store sized to its CSS box at device-pixel
// resolution (DPR capped at 2 to bound fill cost on hi-DPI screens).

import { useEffect, type RefObject } from 'react';

export function useCanvasSize(ref: RefObject<HTMLCanvasElement | null>): void {
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;

    const apply = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cssW = canvas.clientWidth || 1;
      const cssH = canvas.clientHeight || 1;
      const w = Math.max(1, Math.round(cssW * dpr));
      const h = Math.max(1, Math.round(cssH * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    };

    apply();
    const observer = new ResizeObserver(apply);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [ref]);
}
