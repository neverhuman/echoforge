import { useCallback, useEffect, useMemo, useState } from 'react';
import { cancelJob, createJob, fetchJobs } from './radarControl';
import type { JobComposeRequest, JobSummary } from './radarContract';

const PIPELINES: Array<{
  id: string;
  label: string;
}> = [
  {
    id: 'physics_cfar_track_fusion_v1',
    label: 'Physics CFAR Track Fusion',
  },
  {
    id: 'tensor_microdoppler_fusion_v1',
    label: 'Tensor Micro-Doppler Fusion',
  },
  {
    id: 'raw_iq_ssl_research_v1',
    label: 'Raw IQ SSL Research',
  },
];

const VALIDATION_TIER = 'evidence_ladder_v1';
const DEFAULT_SUITE_ID = 'evidence-ladder-v1';
const DATA_ROOT = 'outputs/training-data/best-final-scenario-v1';
const OUT_ROOT = 'outputs/ml-pipelines';

type LoadState =
  | { phase: 'loading' }
  | { phase: 'ready'; jobs: JobSummary[] }
  | { phase: 'error'; message: string };

function metricRow(label: string, value: string | number) {
  return (
    <tr>
      <td>{label}</td>
      <td>{value}</td>
    </tr>
  );
}

function pipelineLabel(pipelineId: string) {
  return PIPELINES.find((pipeline) => pipeline.id === pipelineId)?.label ?? pipelineId;
}

function requestLabel(request: JobSummary['request']) {
  return request.selection === 'pipeline' ? pipelineLabel(request.pipeline_id) : request.suite_id;
}

function chooseSelectedJobId(jobs: JobSummary[], currentJobId: string | null) {
  if (currentJobId && jobs.some((job) => job.job_id === currentJobId)) {
    return currentJobId;
  }
  return jobs[0]?.job_id ?? null;
}

function sortJobs(jobs: JobSummary[]) {
  return [...jobs].sort((left, right) => {
    if (left.created_utc === right.created_utc) {
      return right.job_id.localeCompare(left.job_id);
    }
    return right.created_utc.localeCompare(left.created_utc);
  });
}

export default function JobsPanel() {
  const [state, setState] = useState<LoadState>({ phase: 'loading' });
  const [busy, setBusy] = useState(false);
  const [selection, setSelection] = useState<'pipeline' | 'suite'>('pipeline');
  const [pipelineId, setPipelineId] = useState(PIPELINES[0].id);
  const [suiteId, setSuiteId] = useState(DEFAULT_SUITE_ID);
  const [dataRoot, setDataRoot] = useState(DATA_ROOT);
  const [outRoot, setOutRoot] = useState(OUT_ROOT);
  const [workersPerPipeline, setWorkersPerPipeline] = useState(20);
  const [maxConcurrent, setMaxConcurrent] = useState(3);
  const [seed, setSeed] = useState(20260520390001);
  const [smoke, setSmoke] = useState(true);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  const request = useMemo<JobComposeRequest>(
    () => ({
      job_type: 'ml_processing',
      selection,
      pipeline_id: pipelineId,
      suite_id: suiteId,
      data_root: dataRoot,
      out_root: outRoot,
      workers_per_pipeline: workersPerPipeline,
      max_concurrent: maxConcurrent,
      seed,
      smoke,
      validation_tier: VALIDATION_TIER,
    }),
    [dataRoot, maxConcurrent, outRoot, pipelineId, seed, selection, smoke, suiteId, workersPerPipeline],
  );

  const loadJobs = useCallback(() => {
    setState({ phase: 'loading' });
    fetchJobs()
      .then((jobs) => {
        const sorted = sortJobs(jobs);
        setState({ phase: 'ready', jobs: sorted });
        setSelectedJobId((current) => chooseSelectedJobId(sorted, current));
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err);
        setState({ phase: 'error', message });
      });
  }, []);

  useEffect(() => {
    loadJobs();
  }, [loadJobs]);

  const launchJob = useCallback(() => {
    setBusy(true);
    createJob(request)
      .then((job) => {
        setSelectedJobId(job.job_id);
        loadJobs();
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err);
        setState({ phase: 'error', message });
      })
      .finally(() => setBusy(false));
  }, [loadJobs, request]);

  const jobs = state.phase === 'ready' ? state.jobs : [];
  const selectedJob = useMemo(
    () => jobs.find((job) => job.job_id === selectedJobId) ?? jobs[0] ?? null,
    [jobs, selectedJobId],
  );
  const activePipeline = PIPELINES.find((pipeline) => pipeline.id === pipelineId) ?? PIPELINES[0];
  const selectedJobCanCancel = selectedJob ? selectedJob.status === 'queued' || selectedJob.status === 'running' : false;

  const cancelSelectedJob = useCallback(() => {
    if (!selectedJob || !selectedJobCanCancel) {
      return;
    }
    setBusy(true);
    cancelJob(selectedJob.job_id)
      .then((job) => {
        setSelectedJobId(job.job_id);
        loadJobs();
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err);
        setState({ phase: 'error', message });
      })
      .finally(() => setBusy(false));
  }, [loadJobs, selectedJob, selectedJobCanCancel]);

  return (
    <section className="studio-view" data-testid="jobs-view">
      <div className="studio-view__header">
        <div>
          <h2>ML Processing Jobs</h2>
          <p>Compose, launch, and inspect the evidence-ladder ML pipeline suite.</p>
        </div>
        <button type="button" className="radar-btn" onClick={loadJobs}>
          Refresh
        </button>
      </div>

      <div className="job-layout">
        <section className="radar-panel" data-testid="job-composer">
          <h3 className="radar-panel__title">Composer</h3>
          <label className="radar-field">
            <span>Selection</span>
            <select value={selection} onChange={(event) => setSelection(event.target.value as 'pipeline' | 'suite')}>
              <option value="pipeline">Single pipeline</option>
              <option value="suite">Evidence ladder suite</option>
            </select>
          </label>
          {selection === 'pipeline' ? (
            <label className="radar-field">
              <span>Pipeline</span>
              <select value={pipelineId} onChange={(event) => setPipelineId(event.target.value)}>
                {PIPELINES.map((pipeline) => (
                  <option key={pipeline.id} value={pipeline.id}>
                    {pipeline.label}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <label className="radar-field">
              <span>Suite</span>
              <input type="text" value={suiteId} onChange={(event) => setSuiteId(event.target.value)} />
            </label>
          )}
          <label className="radar-field">
            <span>Data root</span>
            <input type="text" value={dataRoot} onChange={(event) => setDataRoot(event.target.value)} />
          </label>
          <label className="radar-field">
            <span>Output root</span>
            <input type="text" value={outRoot} onChange={(event) => setOutRoot(event.target.value)} />
          </label>
          <label className="radar-field">
            <span>Workers per pipeline - {workersPerPipeline}</span>
            <input
              type="range"
              min="1"
              max="20"
              step="1"
              value={workersPerPipeline}
              onChange={(event) => setWorkersPerPipeline(Number(event.target.value))}
            />
          </label>
          <label className="radar-field">
            <span>Max concurrent - {maxConcurrent}</span>
            <input
              type="range"
              min="1"
              max="3"
              step="1"
              value={maxConcurrent}
              onChange={(event) => setMaxConcurrent(Number(event.target.value))}
            />
          </label>
          <label className="radar-field">
            <span>Seed</span>
            <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
          </label>
          <label className="radar-field radar-field--inline">
            <input type="checkbox" checked={smoke} onChange={(event) => setSmoke(event.target.checked)} />
            <span>Smoke run</span>
          </label>
          <div className="job-chip-row">
            <span className="job-chip job-chip--gold">{request.job_type}</span>
            <span className="job-chip">{selection}</span>
            <span className="job-chip">{request.validation_tier}</span>
            <span className="job-chip">{selection === 'pipeline' ? activePipeline.label : suiteId}</span>
          </div>
          <div className="job-total">
            {selection === 'pipeline'
              ? `Workers: ${workersPerPipeline}`
              : `Workers: ${workersPerPipeline} x max ${maxConcurrent}`}
          </div>
          <button
            type="button"
            className="radar-btn radar-btn--go"
            onClick={launchJob}
            disabled={busy}
            data-testid="job-launch"
          >
            Launch job
          </button>
        </section>

        <div className="job-stack">
          <section className="radar-panel" data-testid="job-board">
            <h3 className="radar-panel__title">Job Queue</h3>
            {state.phase === 'loading' ? <p className="studio-empty">Loading jobs...</p> : null}
            {state.phase === 'error' ? (
              <p className="studio-error">Job API unavailable: {state.message}</p>
            ) : null}
            {jobs.length ? (
              <div className="job-list">
                {jobs.map((job) => (
                  <button
                    type="button"
                    key={job.job_id}
                    className={`job-card${job.job_id === selectedJob?.job_id ? ' is-selected' : ''}`}
                    onClick={() => setSelectedJobId(job.job_id)}
                  >
                    <div className="job-card__head">
                      <strong>{job.job_id}</strong>
                      <span className="job-status-badge" data-status={job.status}>
                        {job.status}
                      </span>
                    </div>
                    <div className="job-card__meta">
                      <span>{job.created_utc}</span>
                      <span>{job.request.selection}</span>
                      <span>{requestLabel(job.request)}</span>
                      <span>{job.progress_percent}%</span>
                    </div>
                    <p>{job.message}</p>
                  </button>
                ))}
              </div>
            ) : (
              <p className="studio-empty">No ML jobs have been launched yet.</p>
            )}
          </section>

          {selectedJob ? (
            <section className="radar-panel" data-testid="job-detail">
              <div className="job-detail__head">
                <div>
                  <h3 className="radar-panel__title">Progress</h3>
                  <p className="job-detail__subtitle">{selectedJob.job_id}</p>
                </div>
                <div className="job-detail__actions">
                  <span className="job-status-badge" data-status={selectedJob.status}>
                    {selectedJob.status}
                  </span>
                  {selectedJobCanCancel ? (
                    <button
                      type="button"
                      className="radar-btn"
                      onClick={cancelSelectedJob}
                      disabled={busy}
                      data-testid="job-cancel"
                    >
                      Cancel job
                    </button>
                  ) : null}
                </div>
              </div>
              <div className="job-progress">
                <div className="job-progress__bar">
                  <span style={{ width: `${selectedJob.progress_percent}%` }} />
                </div>
                <div className="job-progress__meta">
                  <span>{selectedJob.progress_percent}%</span>
                  <span>{selectedJob.request.selection}</span>
                  <span>{selectedJob.request.validation_tier}</span>
                  <span>{selectedJob.results.export_readiness}</span>
                </div>
                <p>{selectedJob.message}</p>
              </div>

              <div className="job-chip-row">
                <span className="job-chip job-chip--gold">{selectedJob.request.job_type}</span>
                <span className="job-chip">{selectedJob.request.selection}</span>
                <span className="job-chip">{requestLabel(selectedJob.request)}</span>
                <span className="job-chip">{selectedJob.results.export_readiness}</span>
              </div>

              <table className="radar-table">
                <tbody>
                  {metricRow('Primary pipeline', selectedJob.results.primary_pipeline_id)}
                  {metricRow('Suite', selectedJob.results.suite_id ?? 'n/a')}
                  {metricRow('Validation tier', selectedJob.request.validation_tier)}
                  {metricRow('PR-AUC', selectedJob.results.pr_auc.toFixed(3))}
                  {metricRow('Export readiness', selectedJob.results.export_readiness)}
                  {metricRow('Artifact root', selectedJob.output_dir)}
                  {metricRow('Artifacts', selectedJob.artifacts.length)}
                  {metricRow('Rows', selectedJob.record_count)}
                </tbody>
              </table>
            </section>
          ) : null}

          {selectedJob ? (
            <section className="radar-panel">
              <h3 className="radar-panel__title">Pipeline Summaries</h3>
              <table className="radar-table">
                <tbody>
                  {selectedJob.results.pipeline_summaries.map((summary) => (
                    <tr key={summary.pipeline_id}>
                      <td>{summary.pipeline_id}</td>
                      <td>{summary.status}</td>
                      <td>
                        {summary.roc_auc.toFixed(3)} / {summary.pr_auc.toFixed(3)}
                      </td>
                      <td>{summary.missing_input_kind ?? 'n/a'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          ) : null}

          {selectedJob ? (
            <section className="radar-panel">
              <h3 className="radar-panel__title">Validation Gates</h3>
              <table className="radar-table">
                <tbody>
                  {selectedJob.results.leakage_gates.map((gate) => (
                    <tr key={gate.gate}>
                      <td>{gate.gate}</td>
                      <td>{gate.status}</td>
                      <td>{gate.detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          ) : null}

          {selectedJob ? (
            <section className="radar-panel">
              <h3 className="radar-panel__title">Phase and Holdout Metrics</h3>
              <table className="radar-table">
                <tbody>
                  {selectedJob.results.phase_auc.map((row) => (
                    <tr key={`phase-${row.label}`}>
                      <td>{row.label}</td>
                      <td>{row.auc.toFixed(3)}</td>
                    </tr>
                  ))}
                  {selectedJob.results.sensor_holdout_auc.map((row) => (
                    <tr key={`sensor-${row.label}`}>
                      <td>{row.label}</td>
                      <td>{row.auc.toFixed(3)}</td>
                    </tr>
                  ))}
                  {selectedJob.results.class_holdout_auc.map((row) => (
                    <tr key={`class-${row.label}`}>
                      <td>{row.label}</td>
                      <td>{row.auc.toFixed(3)}</td>
                    </tr>
                  ))}
                  {selectedJob.results.hard_negative_breakdown.map((row) => (
                    <tr key={`hard-${row.label}`}>
                      <td>{row.label}</td>
                      <td>{row.auc.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          ) : null}

          {selectedJob ? (
            <section className="radar-panel">
              <h3 className="radar-panel__title">Calibration</h3>
              <table className="radar-table">
                <tbody>
                  {selectedJob.results.calibration_bins.map((bin) => (
                    <tr key={bin.bin}>
                      <td>{bin.bin}</td>
                      <td>
                        {bin.lower.toFixed(2)}-{bin.upper.toFixed(2)}
                      </td>
                      <td>{bin.positive_rate.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          ) : null}

          {selectedJob ? (
            <section className="radar-panel">
              <h3 className="radar-panel__title">Artifacts</h3>
              <table className="radar-table">
                <tbody>
                  {selectedJob.artifacts.map((artifact) => (
                    <tr key={artifact.id}>
                      <td>{artifact.kind}</td>
                      <td>{artifact.ready ? 'ready' : 'pending'}</td>
                      <td>{artifact.incomplete ? 'partial' : 'complete'}</td>
                      <td>{artifact.path}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          ) : null}
        </div>
      </div>
    </section>
  );
}
