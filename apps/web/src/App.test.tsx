import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import App, { CatalogSurface } from './App';
import type { EchoForgeContracts } from './runtime';

const snapshot: EchoForgeContracts = {
  service: 'echoforge-studio',
  status: 'ready',
  public_base_url: '/',
  catalog_source: 'contracts/schema_catalog.json',
  bundle_path: 'tests/science/fixtures/bundles/v1_pass',
  health: {
    service: 'echoforge-studio',
    status: 'ok',
    mode: 'rust-studio',
    public_base_url: '/',
    catalog_source: 'contracts/schema_catalog.json',
    schema_count: 2,
    bundle_path: 'tests/science/fixtures/bundles/v1_pass',
    validation_status: 'pass',
    validation_tier: 'V1',
  },
  catalog: {
    service: 'echoforge-studio',
    status: 'ready',
    public_base_url: '/',
    catalog_source: 'contracts/schema_catalog.json',
    schema_count: 2,
    schemas: [
      {
        name: 'object_card',
        schema_file: 'schemas/object_card.schema.json',
        rust_type: 'echoforge_core::ObjectCard',
        python_type: 'echoforge_core.models.ObjectCard',
      },
      {
        name: 'validation_report',
        schema_file: 'schemas/validation_report.schema.json',
        rust_type: 'echoforge_core::ValidationReport',
        python_type: 'echoforge_core.models.ValidationReport',
      },
    ],
  },
  validation: {
    tier: 'V1',
    overall_status: 'pass',
    primitives_checked: ['sphere'],
    n_pass: 4,
    n_warn: 0,
    n_fail: 0,
    error_budget: {
      analytic_db: 0,
      numeric_db: 0,
      method_db: 0,
      total_db: 0,
    },
    checks: {
      canonical_validation_present: true,
      canonical_overall_pass: true,
      polarization_complete: true,
      determinism_pass: true,
      units_frame_pass: true,
      cross_solver_present: false,
      cross_solver_pass: false,
      convergence_present: false,
      convergence_pass: false,
    },
    notes: ['synthetic fixture'],
  },
};

test('catalog surface renders the live contract snapshot', () => {
  const html = renderToStaticMarkup(<CatalogSurface snapshot={snapshot} />);

  expect(html).toContain('Live contract surface');
  expect(html).toContain('contracts/schema_catalog.json');
  expect(html).toContain('object_card');
  expect(html).toContain('validation_report');
  expect(html).toContain('pass');
});

test('app renders the loading shell before data arrives', () => {
  const html = renderToStaticMarkup(<App />);

  expect(html).toContain('Connecting to the studio service');
  expect(html).toContain('Forge public-proxy radar artifacts from one Rust origin.');
});
