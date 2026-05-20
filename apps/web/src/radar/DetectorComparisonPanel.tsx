const DETECTORS = [
  {
    name: 'CFAR',
    latency: 'low',
    evidence: 'range-Doppler threshold baseline',
    status: 'validation-gated',
  },
  {
    name: 'MTI',
    latency: 'low',
    evidence: 'stationary clutter rejection',
    status: 'comparison-ready',
  },
  {
    name: 'Micro-Doppler',
    latency: 'medium',
    evidence: 'propulsor and blade-rate proxy features',
    status: 'uncertainty-labeled',
  },
  {
    name: 'Track-before-detect',
    latency: 'high',
    evidence: 'multi-frame weak signal accumulation',
    status: 'campaign-ready',
  },
];

export default function DetectorComparisonPanel() {
  return (
    <section className="studio-view" data-testid="detectors-view">
      <div className="studio-view__header">
        <div>
          <h2>Detectors</h2>
          <p>Compare classical and multi-frame detectors without hiding validation limits.</p>
        </div>
        <div className="studio-segment" aria-label="detector mode">
          <button type="button" className="is-active">
            Live
          </button>
          <button type="button">Replay</button>
          <button type="button">Campaign</button>
        </div>
      </div>

      <div className="detector-grid">
        {DETECTORS.map((detector) => (
          <article className="detector-card" key={detector.name}>
            <div className="detector-card__head">
              <h3>{detector.name}</h3>
              <span>{detector.latency}</span>
            </div>
            <p>{detector.evidence}</p>
            <div className="detector-meter" aria-label={`${detector.name} confidence`}>
              <span />
            </div>
            <strong>{detector.status}</strong>
          </article>
        ))}
      </div>
    </section>
  );
}
