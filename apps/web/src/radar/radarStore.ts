// A tiny custom store that decouples the high-rate radar stream from
// React. Canvas displays read the mutable `latestScan` / `waterfall`
// fields directly in their rAF loop; React components subscribe to the
// coalesced snapshot via `useSyncExternalStore` (a React built-in — no
// external store library, per the web app conventions).

import { decodeControlFrame, decodeScanFrame } from './frameCodec';
import { SCAN_HEADER_LEN } from './radarContract';
import type {
  ArtifactReadyFrame,
  BackpressureFrame,
  RunLifecycleFrame,
  ScanFrame,
  SessionInfo,
  StatusFrame,
  ValidationStatusFrame,
} from './radarContract';
import { ColumnRing } from './render/ringBuffer';

export type ConnectionState = 'connecting' | 'open' | 'reconnecting' | 'closed';

/** The React-visible, low-rate slice of store state. */
export interface RadarSnapshot {
  connection: ConnectionState;
  session: SessionInfo | null;
  status: StatusFrame | null;
  lifecycle: RunLifecycleFrame | null;
  validation: ValidationStatusFrame | null;
  artifact: ArtifactReadyFrame | null;
  backpressure: BackpressureFrame | null;
  scanSeq: number;
  selectedTrackId: number | null;
  lastError: string | null;
}

const WATERFALL_HISTORY = 480;
const WATERFALL_COLUMN_LEN = 256;
const NOTIFY_INTERVAL_MS = 140;

export class RadarStore {
  /** Newest scan — read imperatively by canvas renderers. */
  latestScan: ScanFrame | null = null;
  /** Rolling micro-Doppler column history for the waterfall. */
  readonly waterfall = new ColumnRing(WATERFALL_HISTORY, WATERFALL_COLUMN_LEN);
  /** Monotonic scan counter — lets renderers detect a fresh frame. */
  scanSeq = 0;

  private snapshot: RadarSnapshot = {
    connection: 'connecting',
    session: null,
    status: null,
    lifecycle: null,
    validation: null,
    artifact: null,
    backpressure: null,
    scanSeq: 0,
    selectedTrackId: null,
    lastError: null,
  };
  private readonly listeners = new Set<() => void>();
  private notifyTimer: ReturnType<typeof setTimeout> | null = null;
  private lastNotifyMs = 0;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = (): RadarSnapshot => this.snapshot;

  setConnection(state: ConnectionState): void {
    this.snapshot = { ...this.snapshot, connection: state };
    this.emit();
  }

  setSelectedTrack(trackId: number | null): void {
    if (this.snapshot.selectedTrackId === trackId) return;
    this.snapshot = { ...this.snapshot, selectedTrackId: trackId };
    this.emit();
  }

  setError(message: string): void {
    this.snapshot = { ...this.snapshot, lastError: message };
    this.emit();
  }

  /** Ingest a JSON text control frame. */
  ingestText(text: string): void {
    const frame = decodeControlFrame(text);
    if (frame.type === 'session_info') {
      const mismatch =
        typeof frame.schema_version === 'number' &&
        frame.schema_version !== 1;
      this.snapshot = {
        ...this.snapshot,
        session: frame,
        lastError: mismatch
          ? `stream schema mismatch: client 1, server ${frame.schema_version}`
          : this.snapshot.lastError,
      };
    } else if (frame.type === 'status') {
      this.snapshot = { ...this.snapshot, status: frame };
    } else if (frame.type === 'run_lifecycle') {
      this.snapshot = { ...this.snapshot, lifecycle: frame };
    } else if (frame.type === 'validation_status') {
      this.snapshot = { ...this.snapshot, validation: frame };
    } else if (frame.type === 'artifact_ready') {
      this.snapshot = { ...this.snapshot, artifact: frame };
    } else {
      this.snapshot = { ...this.snapshot, backpressure: frame };
    }
    this.emit();
  }

  /** Ingest a binary scan frame. */
  ingestBinary(buffer: ArrayBuffer): void {
    if (buffer.byteLength < SCAN_HEADER_LEN) {
      this.setError(`scan frame too short: ${buffer.byteLength}`);
      return;
    }
    const scan = decodeScanFrame(buffer);
    this.latestScan = scan;
    this.waterfall.push(scan.micro_doppler.column);
    this.scanSeq += 1;
    this.scheduleScanNotify();
  }

  // Coalesce high-rate scan notifications to ~7 Hz for React.
  private scheduleScanNotify(): void {
    const now = Date.now();
    if (now - this.lastNotifyMs >= NOTIFY_INTERVAL_MS) {
      this.flushScanNotify();
      return;
    }
    if (this.notifyTimer === null) {
      this.notifyTimer = setTimeout(() => {
        this.notifyTimer = null;
        this.flushScanNotify();
      }, NOTIFY_INTERVAL_MS);
    }
  }

  private flushScanNotify(): void {
    this.lastNotifyMs = Date.now();
    this.snapshot = { ...this.snapshot, scanSeq: this.scanSeq };
    this.emit();
  }

  private emit(): void {
    for (const listener of this.listeners) {
      listener();
    }
  }
}

export const radarStore = new RadarStore();
