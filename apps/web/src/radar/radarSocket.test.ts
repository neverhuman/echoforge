import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { RadarSocket, type SocketLike } from './radarSocket';
import { RadarStore } from './radarStore';

class FakeSocket implements SocketLike {
  binaryType = '';
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  closed = false;

  close(): void {
    this.closed = true;
  }
}

describe('RadarSocket', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  function harness() {
    const store = new RadarStore();
    const sockets: FakeSocket[] = [];
    const socket = new RadarSocket('ws://test/ws/radar', store, () => {
      const fake = new FakeSocket();
      sockets.push(fake);
      return fake;
    });
    return { store, sockets, socket };
  }

  test('opens a socket and reports the connected state', () => {
    const { store, sockets, socket } = harness();
    socket.connect();
    expect(sockets).toHaveLength(1);
    expect(store.getSnapshot().connection).toBe('connecting');
    sockets[0].onopen?.();
    expect(store.getSnapshot().connection).toBe('open');
  });

  test('routes inbound text frames into the store', () => {
    const { store, sockets, socket } = harness();
    socket.connect();
    sockets[0].onopen?.();
    sockets[0].onmessage?.({ data: '{"type":"status","code":"started"}' });
    expect(store.getSnapshot().status?.code).toBe('started');
  });

  test('reconnects with backoff after an unexpected close', () => {
    const { store, sockets, socket } = harness();
    socket.connect();
    sockets[0].onopen?.();
    sockets[0].onclose?.();
    expect(store.getSnapshot().connection).toBe('reconnecting');
    vi.advanceTimersByTime(9000);
    expect(sockets.length).toBe(2);
  });

  test('a user-requested close does not reconnect', () => {
    const { store, sockets, socket } = harness();
    socket.connect();
    sockets[0].onopen?.();
    socket.close();
    expect(store.getSnapshot().connection).toBe('closed');
    vi.advanceTimersByTime(20000);
    expect(sockets.length).toBe(1);
  });
});
