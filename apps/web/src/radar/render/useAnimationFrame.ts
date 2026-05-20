// requestAnimationFrame loop hook. Pauses while the tab is hidden so a
// backgrounded console does not burn CPU on canvases nobody can see.

import { useEffect, useRef } from 'react';

export function useAnimationFrame(callback: (deltaMs: number) => void): void {
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    let running = true;

    const loop = (now: number) => {
      if (!running) return;
      const delta = now - last;
      last = now;
      if (typeof document === 'undefined' || !document.hidden) {
        callbackRef.current(delta);
      }
      raf = requestAnimationFrame(loop);
    };

    raf = requestAnimationFrame(loop);
    return () => {
      running = false;
      cancelAnimationFrame(raf);
    };
  }, []);
}
