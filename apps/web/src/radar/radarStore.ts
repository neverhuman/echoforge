// A tiny custom store that decouples the high-rate radar stream from
// React. Canvas displays read the mutable `latestScan` / `waterfall`
// fields directly in their rAF loop; React components subscribe to the
// coalesced snapshot via `useSyncExternalStore` (a React built-in — no
// external store library, per the web app conventions).

import { decodeControlFrame, decodeScanFrame } from './frameCodec';
import type { ScanFrame, SessionInfo, StatusFrame } from './radarContract';
import { ColumnRing } from './render/ringBuffer';

export type ConnectionState = 'connecting' | 'open' | 'reconnecting' | 'closed';

/** The React-visible, low-rate slice of store state. */
export interface RadarSnapshot {
  connection: ConnectionState;
  session: SessionInfo | null;
  status: StatusFrame | null;
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
      this.snapshot = { ...this.snapshot, session: frame };
    } else {
      this.snapshot = { ...this.snapshot, status: frame };
    }
    this.emit();
  }

  /** Ingest a binary scan frame. */
  ingestBinary(buffer: ArrayBuffer): void {
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
