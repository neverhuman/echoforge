const GATES = [
  ['Validation summary', 'pass'],
  ['Object source card', 'pass'],
  ['Material assumption card', 'pass'],
  ['Leakage guard', 'pass'],
  ['Seed and scenario hash', 'pass'],
];

const NEGATIVES = [
  'bird',
  'ground vehicle',
  'helicopter',
  'wind turbine',
  'rain clutter',
  'RF interference',
];

export default function DatasetExportPanel() {
  return (
    <section className="studio-view" data-testid="dataset-export-view">
      <div className="studio-view__header">
        <div>
          <h2>Dataset Export</h2>
          <p>Downloads stay locked behind provenance, reproducibility, and leakage gates.</p>
        </div>
        <a className="radar-btn radar-btn--go" href="/api/runs/run-shahed-ingress-00/download?kind=dataset">
          Dataset bundle
        </a>
      </div>

      <div className="export-layout">
        <section className="radar-panel">
          <h3 className="radar-panel__title">Export gates</h3>
          <div className="gate-list">
            {GATES.map(([label, status]) => (
              <div className="gate-row" key={label}>
                <span>{label}</span>
                <strong>{status}</strong>
              </div>
            ))}
          </div>
        </section>

        <section className="radar-panel">
          <h3 className="radar-panel__title">Hard negatives</h3>
          <div className="negative-list">
            {NEGATIVES.map((negative) => (
              <span key={negative}>{negative}</span>
            ))}
          </div>
        </section>

        <section className="radar-panel">
          <h3 className="radar-panel__title">Split policy</h3>
          <label className="radar-field">
            <span>Train</span>
            <input type="range" min="50" max="80" defaultValue="70" />
          </label>
          <label className="radar-field">
            <span>Validation</span>
            <input type="range" min="10" max="30" defaultValue="15" />
          </label>
          <label className="radar-field">
            <span>Holdout</span>
            <input type="range" min="10" max="30" defaultValue="15" />
          </label>
        </section>
      </div>
    </section>
  );
}
