import { useCallback, useEffect, useState } from 'react';
import { fetchRuns, replayRun } from './radarControl';
import type { RunSummary } from './radarContract';

type LoadState =
  | { phase: 'loading' }
  | { phase: 'ready'; runs: RunSummary[] }
  | { phase: 'error'; message: string };

function RunsEmptyState() {
  return <p className="studio-empty">No run receipts are available yet.</p>;
}

function RunsErrorState({ message }: { message: string }) {
  return <p className="studio-error">Run history unavailable: {message}</p>;
}

export default function RunsPanel() {
  const [state, setState] = useState<LoadState>({ phase: 'loading' });
  const [busyRunId, setBusyRunId] = useState<string | null>(null);

  const loadRuns = useCallback(() => {
    setState({ phase: 'loading' });
    fetchRuns()
      .then((runs) => setState({ phase: 'ready', runs }))
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err);
        setState({ phase: 'error', message });
      });
  }, []);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  const onReplay = useCallback((runId: string) => {
    setBusyRunId(runId);
    replayRun(runId)
      .catch(() => undefined)
      .finally(() => setBusyRunId(null));
  }, []);

  return (
    <section className="studio-view" data-testid="runs-view">
      <div className="studio-view__header">
        <div>
          <h2>Runs</h2>
          <p>Stable run IDs, seeds, validation gates, artifact manifests.</p>
        </div>
        <button type="button" className="radar-btn" onClick={loadRuns}>
          Refresh
        </button>
      </div>

      {state.phase === 'loading' ? <p className="studio-empty">Loading run history...</p> : null}
      {state.phase === 'error' ? <RunsErrorState message={state.message} /> : null}
      {state.phase === 'ready' && state.runs.length === 0 ? <RunsEmptyState /> : null}
      {state.phase === 'ready' && state.runs.length > 0 ? (
        <div className="run-table-wrap">
          <table className="radar-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Scenario</th>
                <th>Seed</th>
                <th>Validation</th>
                <th>Artifacts</th>
                <th>Replay</th>
              </tr>
            </thead>
            <tbody>
              {state.runs.map((run) => (
                <tr key={run.run_id}>
                  <td>{run.run_id}</td>
                  <td>{run.config.scenario_label}</td>
                  <td>{run.config.seed}</td>
                  <td>
                    {run.validation.tier} / {run.validation.grade}
                  </td>
                  <td>
                    {run.artifacts.map((artifact) => (
                      <a
                        key={artifact.id}
                        className="run-download"
                        href={artifact.download_path}
                      >
                        {artifact.kind}
                      </a>
                    ))}
                  </td>
                  <td>
                    <button
                      type="button"
                      className="radar-btn"
                      disabled={busyRunId === run.run_id}
                      onClick={() => onReplay(run.run_id)}
                      data-testid={`replay-${run.run_id}`}
                    >
                      Replay
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
