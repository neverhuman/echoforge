import { describe, expect, test } from 'vitest';
import { ColumnRing } from './ringBuffer';

describe('ColumnRing', () => {
  test('reports length up to capacity', () => {
    const ring = new ColumnRing(3, 2);
    expect(ring.length).toBe(0);
    ring.push(new Uint8Array([1, 1]));
    ring.push(new Uint8Array([2, 2]));
    expect(ring.length).toBe(2);
  });

  test('evicts the oldest column past capacity', () => {
    const ring = new ColumnRing(3, 2);
    for (let i = 1; i <= 5; i++) {
      ring.push(new Uint8Array([i, i]));
    }
    expect(ring.length).toBe(3);
    expect(Array.from(ring.at(0) ?? [])).toEqual([5, 5]);
    expect(Array.from(ring.at(2) ?? [])).toEqual([3, 3]);
    expect(ring.at(3)).toBeNull();
  });

  test('resamples a mismatched column length', () => {
    const ring = new ColumnRing(2, 4);
    ring.push(new Uint8Array([10, 20]));
    const col = ring.at(0);
    expect(col?.length).toBe(4);
    expect(col?.[0]).toBe(10);
    expect(col?.[3]).toBe(20);
  });

  test('clear resets the ring', () => {
    const ring = new ColumnRing(2, 2);
    ring.push(new Uint8Array([1, 1]));
    ring.clear();
    expect(ring.length).toBe(0);
    expect(ring.at(0)).toBeNull();
  });

  test('rejects non-positive dimensions', () => {
    expect(() => new ColumnRing(0, 4)).toThrow();
    expect(() => new ColumnRing(4, 0)).toThrow();
  });
});
