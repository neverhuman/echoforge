import { useCallback, useState } from 'react';

import type { MeshDescriptor, MeshStats } from './domainTypes';
import MeshCanvas from './MeshCanvas';

export type { MeshDescriptor, MeshStats };

// ---------------------------------------------------------------------------
// List-level state machine — this component is a pure view over a
// pre-loaded primitive list passed in as props.
// ---------------------------------------------------------------------------

/**
 * Format `boundingBox` for the status bar as `dx x dy x dz m`. Three sig
 * figs is plenty for the canonical primitives (sphere radius ~ 1 m, etc.).
 */
export function formatBoundingBox(bbox: MeshStats['boundingBox']): string {
  const dx = bbox.max[0] - bbox.min[0];
  const dy = bbox.max[1] - bbox.min[1];
  const dz = bbox.max[2] - bbox.min[2];
  function fmtDim(n: number): string {
    if (Math.abs(n) < 1e-3) return '0';
    return n.toFixed(3);
  }
  return `${fmtDim(dx)} x ${fmtDim(dy)} x ${fmtDim(dz)} m`;
}

// ---------------------------------------------------------------------------
// Named status helpers and sub-components
// ---------------------------------------------------------------------------

function meshStatusText(meshPhase: 'idle' | 'loading' | 'error', errorMessage: string | null): string {
  if (meshPhase === 'loading') return 'loading...';
  if (meshPhase === 'error') {
    if (errorMessage !== null) return errorMessage;
    return 'error';
  }
  return 'ok';
}

function TriangleCountBadge({ stats }: { stats: MeshStats | null }) {
  if (stats === null) return <></>;
  return <span className="badge badge--mode">{stats.triangleCount} tri</span>;
}

function renderStatValue(stats: MeshStats | null, extractor: (s: MeshStats) => string): string {
  if (stats === null) return '—';
  return extractor(stats);
}

interface MeshViewerProps {
  /** Pre-loaded mesh primitive list passed in from the parent component. */
  list: MeshDescriptor[];
}

export default function MeshViewer({ list }: MeshViewerProps) {
  const [selected, setSelected] = useState<string>(list[0]?.primitive_id ?? '');
  const [meshStats, setMeshStats] = useState<MeshStats | null>(null);
  const [meshPhase, setMeshPhase] = useState<'idle' | 'loading' | 'error'>('idle');
  const [meshErrorMessage, setMeshErrorMessage] = useState<string | null>(null);

  const onLoading = useCallback(() => {
    setMeshPhase('loading');
    setMeshStats(null);
    setMeshErrorMessage(null);
  }, []);

  const onLoaded = useCallback((stats: MeshStats) => {
    setMeshPhase('idle');
    setMeshStats(stats);
    setMeshErrorMessage(null);
  }, []);

  const onError = useCallback((msg: string) => {
    setMeshPhase('error');
    setMeshStats(null);
    setMeshErrorMessage(msg);
  }, []);

  const onPrimitiveChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const nextId = event.target.value;
    if (nextId === selected) return;
    setSelected(nextId);
  };

  return (
    <article className="status-card mesh-viewer" data-testid="mesh-viewer">
      <div className="status-card__header">
        <div>
          <h2>Mesh viewer</h2>
          <p className="status-summary">
            Parametric mesh primitives synthesized in-process by{' '}
            <code>echoforge-world::mesh</code> and streamed as binary STL.
          </p>
        </div>
        <div className="status-pills" aria-label="mesh status">
          <span className="badge badge--neutral">{selected}</span>
          <TriangleCountBadge stats={meshStats} />
        </div>
      </div>

      <div className="mesh-viewer__controls">
        <label className="mesh-viewer__select-label" htmlFor="mesh-viewer-primitive">
          Primitive
        </label>
        <select
          id="mesh-viewer-primitive"
          className="mesh-viewer__select"
          value={selected}
          onChange={onPrimitiveChange}
        >
          {list.map((p) => (
            <option key={p.primitive_id} value={p.primitive_id}>
              {p.display_name}
            </option>
          ))}
        </select>
      </div>

      <div className="mesh-viewer__surface" data-testid="mesh-viewer-surface">
        <MeshCanvas
          primitiveId={selected}
          onLoaded={onLoaded}
          onError={onError}
          onLoading={onLoading}
        />
      </div>

      <dl className="mesh-viewer__stats">
        <dt>Primitive</dt>
        <dd>{selected}</dd>
        <dt>Triangle count</dt>
        <dd>{renderStatValue(meshStats, (s) => s.triangleCount.toLocaleString())}</dd>
        <dt>Bounding box</dt>
        <dd>{renderStatValue(meshStats, (s) => formatBoundingBox(s.boundingBox))}</dd>
        <dt>Status</dt>
        <dd>{meshStatusText(meshPhase, meshErrorMessage)}</dd>
      </dl>
    </article>
  );
}
