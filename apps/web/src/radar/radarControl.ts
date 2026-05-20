// REST control surface for the radar simulation (`/api/sim/*`). The
// control plane is request/response; the live data plane is the
// WebSocket in `radarSocket.ts`.

import type {
  JobComposeRequest,
  JobSummary,
  RadarParamPatch,
  RunArtifact,
  RunSummary,
  ScenarioSummary,
} from './radarContract';

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

export async function fetchRuns(): Promise<RunSummary[]> {
  const res = await fetch('/api/runs');
  if (!res.ok) {
    throw new Error(`GET /api/runs failed: ${res.status}`);
  }
  return (await res.json()) as RunSummary[];
}

export async function fetchRunArtifacts(runId: string): Promise<RunArtifact[]> {
  const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/artifacts`);
  if (!res.ok) {
    throw new Error(`GET /api/runs/${runId}/artifacts failed: ${res.status}`);
  }
  return (await res.json()) as RunArtifact[];
}

export async function replayRun(runId: string): Promise<unknown> {
  const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/replay`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ mode: 'exact_seed' }),
  });
  if (!res.ok) {
    throw new Error(`POST /api/runs/${runId}/replay failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchJobs(): Promise<JobSummary[]> {
  const res = await fetch('/api/jobs');
  if (!res.ok) {
    throw new Error(`GET /api/jobs failed: ${res.status}`);
  }
  return (await res.json()) as JobSummary[];
}

export async function createJob(request: JobComposeRequest): Promise<JobSummary> {
  const res = await fetch('/api/jobs', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    throw new Error(`POST /api/jobs failed: ${res.status}`);
  }
  return (await res.json()) as JobSummary;
}

export async function cancelJob(jobId: string): Promise<JobSummary> {
  const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`POST /api/jobs/${jobId}/cancel failed: ${res.status}`);
  }
  return (await res.json()) as JobSummary;
}
