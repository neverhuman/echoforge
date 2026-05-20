// REST control surface for the radar simulation (`/api/sim/*`). The
// control plane is request/response; the live data plane is the
// WebSocket in `radarSocket.ts`.

import type { RadarParamPatch, ScenarioSummary } from './radarContract';

async function postSim(path: string, body?: unknown): Promise<unknown> {
  const res = await fetch(`/api/sim/${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) {
    throw new Error(`POST /api/sim/${path} failed: ${res.status}`);
  }
  return res.json();
}

export interface StartOptions {
  scenarioId?: string;
  mode?: 'live' | 'replay';
  bundlePath?: string;
}

export function startSim(opts: StartOptions = {}): Promise<unknown> {
  return postSim('start', {
    scenario_id: opts.scenarioId,
    mode: opts.mode ?? 'live',
    bundle_path: opts.bundlePath,
  });
}

export const stopSim = (): Promise<unknown> => postSim('stop');
export const pauseSim = (): Promise<unknown> => postSim('pause');
export const resumeSim = (): Promise<unknown> => postSim('resume');
export const setPlaybackSpeed = (speed: number): Promise<unknown> =>
  postSim('speed', { speed });
export const setRadarParams = (patch: RadarParamPatch): Promise<unknown> =>
  postSim('params', patch);

export async function fetchScenarios(): Promise<ScenarioSummary[]> {
  const res = await fetch('/api/sim/scenarios');
  if (!res.ok) {
    throw new Error(`GET /api/sim/scenarios failed: ${res.status}`);
  }
  return (await res.json()) as ScenarioSummary[];
}
