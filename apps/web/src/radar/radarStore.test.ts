import { describe, expect, test } from 'vitest';
import { RadarStore } from './radarStore';
import { sampleScanBuffer } from './scanFixtures';

describe('RadarStore', () => {
  test('ingests a session_info control frame', () => {
    const store = new RadarStore();
    store.ingestText('{"type":"session_info","scenario_id":"shahed-ingress"}');
    expect(store.getSnapshot().session?.scenario_id).toBe('shahed-ingress');
  });

  test('ingests a status control frame', () => {
    const store = new RadarStore();
    store.ingestText('{"type":"status","code":"started","message":"go"}');
    expect(store.getSnapshot().status?.code).toBe('started');
  });

  test('ingests a binary scan frame', () => {
    const store = new RadarStore();
    store.ingestBinary(sampleScanBuffer());
    expect(store.latestScan?.meta.frame_index).toBe(12);
    expect(store.scanSeq).toBe(1);
    expect(store.waterfall.length).toBe(1);
  });

  test('advances scanSeq on each frame', () => {
    const store = new RadarStore();
    store.ingestBinary(sampleScanBuffer({ frame_index: 1 }));
    store.ingestBinary(sampleScanBuffer({ frame_index: 2 }));
    expect(store.scanSeq).toBe(2);
    expect(store.latestScan?.meta.frame_index).toBe(2);
  });

  test('notifies subscribers and tracks selection', () => {
    const store = new RadarStore();
    let notifications = 0;
    const unsubscribe = store.subscribe(() => {
      notifications += 1;
    });
    store.setConnection('open');
    store.setSelectedTrack(7);
    expect(store.getSnapshot().connection).toBe('open');
    expect(store.getSnapshot().selectedTrackId).toBe(7);
    expect(notifications).toBeGreaterThanOrEqual(2);
    unsubscribe();
    store.setConnection('closed');
    expect(notifications).toBe(notifications); // no further calls after unsubscribe
  });

  test('keeps a stable snapshot reference until something changes', () => {
    const store = new RadarStore();
    const first = store.getSnapshot();
    expect(store.getSnapshot()).toBe(first);
    store.setConnection('open');
    expect(store.getSnapshot()).not.toBe(first);
  });
});
