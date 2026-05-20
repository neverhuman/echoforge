import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  cancelJob,
  createJob,
  fetchJobDefaults,
  fetchJobs,
  staticPreviewJobDefaults,
} from './radarControl';
import JobComposer from './JobComposer';
import JobDetailSections from './JobDetailSections';
import { coerceSelection, requestLabel, type JobSelection } from './jobLabels';
import type { JobComposeRequest, JobDefaultsResponse, JobSummary } from './radarContract';

type LoadState =
  | { phase: 'loading' }
  | { phase: 'ready'; jobs: JobSummary[] }
  | { phase: 'error'; message: string };

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
  const [defaults, setDefaults] = useState<JobDefaultsResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [selection, setSelection] = useState<JobSelection>('pipeline');
  const [pipelineId, setPipelineId] = useState('');
  const [suiteId, setSuiteId] = useState('');
  const [dataRoot, setDataRoot] = useState('');
  const [outRoot, setOutRoot] = useState('');
  const [workersPerPipeline, setWorkersPerPipeline] = useState(1);
  const [maxConcurrent, setMaxConcurrent] = useState(1);
  const [seed, setSeed] = useState(0);
  const [smoke, setSmoke] = useState(true);
  const [validationTier, setValidationTier] = useState('');
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const pipelines = defaults?.pipelines ?? [];
  const suites = defaults?.suites ?? [];
  const composerReady = defaults !== null;

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
      validation_tier: validationTier,
    }),
    [dataRoot, maxConcurrent, outRoot, pipelineId, seed, selection, smoke, suiteId, validationTier, workersPerPipeline],
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

  useEffect(() => {
    let active = true;
    fetchJobDefaults()
      .catch(() => staticPreviewJobDefaults())
      .then((response) => {
        if (!active) {
          return;
        }
        const next = response.request;
        setDefaults(response);
        setSelection(coerceSelection(next.selection));
        setPipelineId(next.pipeline_id);
        setSuiteId(next.suite_id);
        setDataRoot(next.data_root);
        setOutRoot(next.out_root);
        setWorkersPerPipeline(next.workers_per_pipeline);
        setMaxConcurrent(next.max_concurrent);
        setSeed(next.seed);
        setSmoke(next.smoke);
        setValidationTier(next.validation_tier);
      });
    return () => {
      active = false;
    };
  }, []);

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
        <JobComposer
          busy={busy}
          composerReady={composerReady}
          dataRoot={dataRoot}
          launchJob={launchJob}
          maxConcurrent={maxConcurrent}
          outRoot={outRoot}
          pipelineId={pipelineId}
          pipelines={pipelines}
          request={request}
          seed={seed}
          selection={selection}
          setDataRoot={setDataRoot}
          setMaxConcurrent={setMaxConcurrent}
          setOutRoot={setOutRoot}
          setPipelineId={setPipelineId}
          setSeed={setSeed}
          setSelection={setSelection}
          setSmoke={setSmoke}
          setSuiteId={setSuiteId}
          setWorkersPerPipeline={setWorkersPerPipeline}
          smoke={smoke}
          suiteId={suiteId}
          suites={suites}
          workersPerPipeline={workersPerPipeline}
        />

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
                      <span>{requestLabel(job.request, pipelines)}</span>
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
            <JobDetailSections
              busy={busy}
              cancelSelectedJob={cancelSelectedJob}
              pipelines={pipelines}
              selectedJob={selectedJob}
              selectedJobCanCancel={selectedJobCanCancel}
            />
          ) : null}
        </div>
      </div>
    </section>
  );
}
