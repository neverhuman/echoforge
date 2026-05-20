import { expect, test } from 'vitest';
import catalogRoot from '../../../contracts/schema_catalog.json';

const schemaCatalog = (catalogRoot as { schemas: Array<{ name: string; schema_file: string; rust_type: string; python_type: string }> }).schemas;

test('schema catalog stays canonical and stable', () => {
  expect(schemaCatalog).toHaveLength(12);
  expect(schemaCatalog.map((entry) => entry.name)).toEqual([
    'object_card',
    'material_card',
    'mesh_manifest',
    'solver_card',
    'rcs_campaign',
    'echosig_manifest',
    'sensor_archetype',
    'scenario',
    'radar_episode',
    'detector_graph',
    'dataset_card',
    'validation_report',
  ]);

  for (const entry of schemaCatalog) {
    expect(entry.schema_file).toMatch(/^schemas\/.+\.schema\.json$/);
    expect(entry.rust_type).toContain('echoforge_core::');
    expect(entry.python_type).toContain('echoforge_core.models.');
  }
});
