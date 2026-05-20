import type { Dispatch, SetStateAction } from 'react';
import { coerceSelection, pipelineLabel, type JobSelection } from './jobLabels';
import type { JobComposeRequest, JobOption } from './radarContract';

interface JobComposerProps {
  busy: boolean;
  composerReady: boolean;
  dataRoot: string;
  launchJob: () => void;
  maxConcurrent: number;
  outRoot: string;
  pipelineId: string;
  pipelines: JobOption[];
  request: JobComposeRequest;
  seed: number;
  selection: JobSelection;
  setDataRoot: Dispatch<SetStateAction<string>>;
  setMaxConcurrent: Dispatch<SetStateAction<number>>;
  setOutRoot: Dispatch<SetStateAction<string>>;
  setPipelineId: Dispatch<SetStateAction<string>>;
  setSeed: Dispatch<SetStateAction<number>>;
  setSelection: Dispatch<SetStateAction<JobSelection>>;
  setSmoke: Dispatch<SetStateAction<boolean>>;
  setSuiteId: Dispatch<SetStateAction<string>>;
  setWorkersPerPipeline: Dispatch<SetStateAction<number>>;
  smoke: boolean;
  suiteId: string;
  suites: JobOption[];
  workersPerPipeline: number;
}

export default function JobComposer(props: JobComposerProps) {
  const activeLabel = pipelineLabel(props.pipelineId, props.pipelines);
  return (
    <section className="radar-panel" data-testid="job-composer">
      <h3 className="radar-panel__title">Composer</h3>
      <label className="radar-field">
        <span>Selection</span>
        <select
          value={props.selection}
          onChange={(event) => props.setSelection(coerceSelection(event.target.value))}
        >
          <option value="pipeline">Single pipeline</option>
          <option value="suite">Evidence ladder suite</option>
        </select>
      </label>
      {props.selection === 'pipeline' ? (
        <label className="radar-field">
          <span>Pipeline</span>
          <select
            value={props.pipelineId}
            onChange={(event) => props.setPipelineId(event.target.value)}
          >
            {props.pipelines.map((pipeline) => (
              <option key={pipeline.id} value={pipeline.id}>
                {pipeline.label}
              </option>
            ))}
          </select>
        </label>
      ) : (
        <label className="radar-field">
          <span>Suite</span>
          <select
            value={props.suiteId}
            onChange={(event) => props.setSuiteId(event.target.value)}
          >
            {props.suites.map((suite) => (
              <option key={suite.id} value={suite.id}>
                {suite.label}
              </option>
            ))}
          </select>
        </label>
      )}
      <label className="radar-field">
        <span>Data root</span>
        <input type="text" value={props.dataRoot} onChange={(event) => props.setDataRoot(event.target.value)} />
      </label>
      <label className="radar-field">
        <span>Output root</span>
        <input type="text" value={props.outRoot} onChange={(event) => props.setOutRoot(event.target.value)} />
      </label>
      <label className="radar-field">
        <span>Workers per pipeline - {props.workersPerPipeline}</span>
        <input
          type="range"
          min="1"
          max="20"
          step="1"
          value={props.workersPerPipeline}
          onChange={(event) => props.setWorkersPerPipeline(Number(event.target.value))}
        />
      </label>
      <label className="radar-field">
        <span>Max concurrent - {props.maxConcurrent}</span>
        <input
          type="range"
          min="1"
          max="3"
          step="1"
          value={props.maxConcurrent}
          onChange={(event) => props.setMaxConcurrent(Number(event.target.value))}
        />
      </label>
      <label className="radar-field">
        <span>Seed</span>
        <input type="number" value={props.seed} onChange={(event) => props.setSeed(Number(event.target.value))} />
      </label>
      <label className="radar-field radar-field--inline">
        <input type="checkbox" checked={props.smoke} onChange={(event) => props.setSmoke(event.target.checked)} />
        <span>Smoke run</span>
      </label>
      <div className="job-chip-row">
        <span className="job-chip job-chip--gold">{props.request.job_type}</span>
        <span className="job-chip">{props.selection}</span>
        <span className="job-chip">{props.request.validation_tier}</span>
        <span className="job-chip">{props.selection === 'pipeline' ? activeLabel : props.suiteId}</span>
      </div>
      <div className="job-total">
        {props.selection === 'pipeline'
          ? `Workers: ${props.workersPerPipeline}`
          : `Workers: ${props.workersPerPipeline} x max ${props.maxConcurrent}`}
      </div>
      <button
        type="button"
        className="radar-btn radar-btn--go"
        onClick={props.launchJob}
        disabled={props.busy || !props.composerReady}
        data-testid="job-launch"
      >
        Launch job
      </button>
    </section>
  );
}
