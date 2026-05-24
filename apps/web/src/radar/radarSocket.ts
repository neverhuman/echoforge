// WebSocket client for the `/ws/radar` stream. Auto-reconnects with
// exponential backoff + jitter and feeds frames into a `RadarStore`.

import { radarStore, type RadarStore } from './radarStore';

/** Minimal WebSocket surface — lets tests inject a fake socket. */
export interface SocketLike {
  binaryType: string;
  onopen: (() => void) | null;
  onclose: (() => void) | null;
  onerror: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  close(): void;
}

export type SocketFactory = (url: string) => SocketLike;

const MIN_BACKOFF_MS = 250;
const MAX_BACKOFF_MS = 8000;

/** Resolve the `/ws/radar` URL for the current page origin. */
export function defaultRadarUrl(): string {
  if (typeof location === 'undefined') return 'ws://localhost:8080/ws/radar';
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${scheme}://${location.host}/ws/radar`;
}

export class RadarSocket {
  private socket: SocketLike | null = null;
  private backoffMs = MIN_BACKOFF_MS;
  private closedByUser = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly url: string = defaultRadarUrl(),
    private readonly store: RadarStore = radarStore,
    private readonly factory: SocketFactory = (u) =>
      new WebSocket(u) as unknown as SocketLike,
  ) {}

  connect(): void {
    this.closedByUser = false;
    this.open();
  }

  close(): void {
    this.closedByUser = true;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.socket?.close();
    this.store.setConnection('closed');
  }

  private open(): void {
    this.store.setConnection(this.socket ? 'reconnecting' : 'connecting');
    let socket: SocketLike;
    try {
      socket = this.factory(this.url);
    } catch (err) {
      this.store.setError(`socket open failed: ${String(err)}`);
      this.scheduleReconnect();
      return;
    }
    socket.binaryType = 'arraybuffer';
    this.socket = socket;

    socket.onopen = () => {
      this.backoffMs = MIN_BACKOFF_MS;
      this.store.setConnection('open');
    };
    socket.onmessage = (ev) => {
      try {
        if (typeof ev.data === 'string') {
          this.store.ingestText(ev.data);
        } else if (ev.data instanceof ArrayBuffer) {
          this.store.ingestBinary(ev.data);
        }
      } catch (err) {
        this.store.setError(`frame decode failed: ${String(err)}`);
      }
    };
    socket.onerror = () => {
      // `onclose` always follows — reconnect is handled there.
    };
    socket.onclose = () => {
      this.socket = null;
      if (this.closedByUser) {
        this.store.setConnection('closed');
      } else {
        this.scheduleReconnect();
      }
    };
  }

  private scheduleReconnect(): void {
    this.store.setConnection('reconnecting');
    const jitter = Math.random() * 200;
    const delay = Math.min(this.backoffMs, MAX_BACKOFF_MS) + jitter;
    this.backoffMs = Math.min(this.backoffMs * 2, MAX_BACKOFF_MS);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.closedByUser) {
        this.open();
      }
    }, delay);
  }
}
