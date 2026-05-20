import { requestLabel } from './jobLabels';
import type { JobOption, JobSummary } from './radarContract';

interface JobDetailSectionsProps {
  busy: boolean;
  cancelSelectedJob: () => void;
  pipelines: JobOption[];
  selectedJob: JobSummary;
  selectedJobCanCancel: boolean;
}

function metricRow(label: string, value: string | number) {
  return (
    <tr>
      <td>{label}</td>
      <td>{value}</td>
    </tr>
  );
}

export default function JobDetailSections({
  busy,
  cancelSelectedJob,
  pipelines,
  selectedJob,
  selectedJobCanCancel,
}: JobDetailSectionsProps) {
  return (
    <>
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
          <span className="job-chip">{requestLabel(selectedJob.request, pipelines)}</span>
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
    </>
  );
}
