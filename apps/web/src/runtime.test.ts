import { expect, test } from 'vitest';
import { provenanceLinkForDetection } from './runtime';

test('provenance link uses the same-origin /provenance/detection/ path', () => {
  expect(provenanceLinkForDetection('record_000223__cfar_tracker_baseline__frame_0102')).toBe(
    '/provenance/detection/record_000223__cfar_tracker_baseline__frame_0102',
  );
});

test('provenance link percent-encodes unsafe URL characters in detection_id', () => {
  expect(provenanceLinkForDetection('rec/oddly named__model y__frame_0001')).toBe(
    '/provenance/detection/rec%2Foddly%20named__model%20y__frame_0001',
  );
});
