import { createEchoForgeClient } from './generated/contracts-client.generated.mjs';

export interface EchoForgeCatalogEntry {
  name: string;
  schema_file: string;
  rust_type: string;
  python_type: string;
}

export interface EchoForgeHealth {
  service: string;
  status: string;
  mode: string;
  public_base_url: string;
  catalog_source: string;
  schema_count: number;
  bundle_path: string;
  validation_status: string;
  validation_tier: string;
}

export interface EchoForgeCatalog {
  service: string;
  status: string;
  public_base_url: string;
  catalog_source: string;
  schema_count: number;
  schemas: EchoForgeCatalogEntry[];
}

export interface EchoForgeValidationErrorBudget {
  analytic_db: number;
  numeric_db: number;
  method_db: number;
  total_db: number;
}

export interface EchoForgeValidationChecks {
  canonical_validation_present: boolean;
  canonical_overall_pass: boolean;
  polarization_complete: boolean;
  determinism_pass: boolean;
  units_frame_pass: boolean;
  cross_solver_present: boolean;
  cross_solver_pass: boolean;
  convergence_present: boolean;
  convergence_pass: boolean;
}

export interface EchoForgeValidationReport {
  tier: string;
  overall_status: string;
  primitives_checked: string[];
  n_pass: number;
  n_warn: number;
  n_fail: number;
  error_budget: EchoForgeValidationErrorBudget;
  checks: EchoForgeValidationChecks;
  notes: string[];
}

export interface EchoForgeContracts {
  service: string;
  status: string;
  public_base_url: string;
  catalog_source: string;
  bundle_path: string;
  health: EchoForgeHealth;
  catalog: EchoForgeCatalog;
  validation: EchoForgeValidationReport;
}

export async function loadStudioContracts(): Promise<EchoForgeContracts> {
  return createEchoForgeClient().getContracts() as Promise<EchoForgeContracts>;
}
