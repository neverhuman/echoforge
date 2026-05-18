import assert from 'node:assert/strict';
import test from 'node:test';
import { buildGpuDoctorReport } from './gpu-doctor.mjs';

test('gpu doctor report is explicitly placeholder-only', () => {
  const report = buildGpuDoctorReport();

  assert.equal(report.kind, 'echoforge.gpu_doctor');
  assert.equal(report.status, 'placeholder');
  assert.equal(report.mode, 'scaffold');
  assert.ok(Array.isArray(report.checks));
  assert.ok(report.notes.some((note) => note.includes('placeholder')));
});

