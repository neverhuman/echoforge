import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// The Rust `echoforge-studio` service serves the API + WebSocket. In dev
// the Vite server proxies to it; override the origin with
// ECHOFORGE_STUDIO_ORIGIN. In production the studio serves the built SPA
// itself, so these relative paths resolve same-origin with no proxy.
const studioOrigin = process.env.ECHOFORGE_STUDIO_ORIGIN ?? 'http://127.0.0.1:8080';
const studioWsOrigin = studioOrigin.replace(/^http/, 'ws');

export default defineConfig({
  root: new URL('.', import.meta.url).pathname,
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/ws/radar': { target: studioWsOrigin, ws: true, changeOrigin: true },
      '/api': { target: studioOrigin, changeOrigin: true },
      '/healthz': { target: studioOrigin, changeOrigin: true },
      '/provenance': { target: studioOrigin, changeOrigin: true },
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  },
});
