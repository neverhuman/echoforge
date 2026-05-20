import { describe, expect, test } from 'vitest';
import { formatBoundingBox } from './MeshViewer';

describe('formatBoundingBox', () => {
  test('formats sphere-sized boxes with three decimals', () => {
    expect(
      formatBoundingBox({ min: [-1, -1, -1], max: [1, 1, 1] }),
    ).toBe('2.000 x 2.000 x 2.000 m');
  });

  test('collapses near-zero dimensions to 0', () => {
    expect(
      formatBoundingBox({ min: [-1.5, -2, 0], max: [1.5, 2, 0] }),
    ).toBe('3.000 x 4.000 x 0 m');
  });
});
