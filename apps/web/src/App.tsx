import { useEffect, useState } from 'react';
import { loadStudioContracts, type EchoForgeCatalogEntry, type EchoForgeContracts } from './runtime';

function stat(label: string, value: string) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className="stat__value">{value}</span>
    </div>
  );
}

function CatalogItem({ schema }: { schema: EchoForgeCatalogEntry }) {
  return (
    <li className="catalog-item">
      <div className="catalog-item__head">
        <h3>{schema.name}</h3>
        <span className="badge badge--neutral">canonical</span>
      </div>
      <p className="catalog-item__detail">{schema.schema_file}</p>
      <dl className="catalog-item__meta">
        <dt>Rust</dt>
        <dd>{schema.rust_type}</dd>
        <dt>Python</dt>
        <dd>{schema.python_type}</dd>
      </dl>
    </li>
  );
}

export function CatalogSurface({ snapshot }: { snapshot: EchoForgeContracts }) {
  const { health, catalog, validation } = snapshot;
  return (
    <article className="status-card">
      <div className="status-card__header">
        <div>
          <h2>Live contract surface</h2>
          <p className="status-summary">
            Same-origin Rust service, canonical schema catalog, and demo bundle
            validation from {health.bundle_path}.
          </p>
        </div>
        <div className="status-pills" aria-label="status badges">
          <span className="badge badge--healthy">{health.status}</span>
          <span className="badge badge--mode">{health.mode}</span>
          <span className={`badge badge--${validation.overall_status}`}>{validation.overall_status}</span>
        </div>
      </div>

      <div className="stat-grid">
        {stat('Service', health.service)}
        {stat('Catalog source', catalog.catalog_source)}
        {stat('Bundle', health.bundle_path)}
        {stat('Validation tier', validation.tier)}
        {stat('Catalog count', String(catalog.schema_count))}
        {stat('Validation status', validation.overall_status)}
      </div>

      <p className="status-summary status-summary--wide">
        The catalog is loaded from {catalog.catalog_source} and the latest report
        shows {validation.n_pass} passing checks, {validation.n_warn} warnings,
        and {validation.n_fail} failures.
      </p>

      <ul className="catalog-list">
        {catalog.schemas.map((schema) => (
          <CatalogItem key={schema.name} schema={schema} />
        ))}
      </ul>

      {validation.notes.length > 0 ? (
        <div className="notes-panel">
          <h3>Validation notes</h3>
          <ul className="notes-list">
            {validation.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </article>
  );
}

type LoadState =
  | { phase: 'loading' }
  | { phase: 'ready'; snapshot: EchoForgeContracts }
  | { phase: 'error'; message: string };

export default function App() {
  const [state, setState] = useState<LoadState>({ phase: 'loading' });

  useEffect(() => {
    let active = true;

    loadStudioContracts()
      .then((snapshot) => {
        if (active) {
          setState({ phase: 'ready', snapshot });
        }
      })
      .catch((err: unknown) => {
        if (!active) return;
        const message = err instanceof Error ? err.message : String(err);
        setState({ phase: 'error', message });
      });

    return () => {
      active = false;
    };
  }, []);

  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">EchoForge Studio</p>
        <h1>Forge public-proxy radar artifacts from one Rust origin.</h1>
        <p className="lede">
          The studio UI fetches live health, the canonical catalog, and the latest
          validation report from the same service that serves the Vite build.
        </p>
      </section>

      <section className="panel">
        <div className="app-state" aria-live="polite">
          {state.phase === 'loading' ? (
            <article className="status-card status-card--loading">
              <h2>Connecting to the studio service</h2>
              <p className="status-summary">
                Loading the same-origin contract snapshot and demo bundle report.
              </p>
            </article>
          ) : null}

          {state.phase === 'error' ? (
            <article className="status-card status-card--error">
              <h2>Unable to load the studio snapshot</h2>
              <p className="status-summary">{state.message}</p>
            </article>
          ) : null}

          {state.phase === 'ready' ? <CatalogSurface snapshot={state.snapshot} /> : null}
        </div>
      </section>
    </main>
  );
}
