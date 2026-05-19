//! EchoForge solver adapter contract.
//!
//! Defines a formal trait + value-types that heterogeneous RCS solvers
//! (SagittaSBR, openEMS, SCUFF-EM, Bempp, Palace, Puma-EM, etc.) must
//! implement to plug into the EchoForge planner/runner. The shapes
//! mirror the optional additive surface added to
//! `schemas/solver_card.schema.json` by the `solver-adapter-contract`
//! packet so a registered adapter can be reconciled against its on-disk
//! solver_card.
//!
//! The contract is intentionally small: planners declare a request,
//! the adapter produces a plan, then a plan is executed and returns a
//! run report. Everything else (mesh production, material binding,
//! container fan-out) is the runner's concern.
//!
//! See FUCKIT.md.done P3 cross-tip resolution for the
//! `core_open | optional_open | restricted_plugin` distinction:
//! strict-open default ships only `core_open`; `optional_open`
//! plugins are opt-in builds from open-source code; `restricted_plugin`
//! adapters never ship in the strict-open default but may be loaded
//! by downstream consumers under their own license obligations.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use echoforge_core::NumericRange;

/// Polarization labels accepted on the wire by `solver_card`
/// (`supported_polarizations`). Mirrors the schema enum
/// `{H, V, L, R}`. Re-defined here rather than imported from
/// `echoforge-validate` so this contract crate stays dependency-light
/// and free of science-stack imports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Polarization {
    H,
    V,
    L,
    R,
}

/// License-class declaration controlling whether a solver may ship in
/// the strict-open default packaging.
///
/// * `CoreOpen` — solver is part of the strict-open default; always
///   built.
/// * `OptionalOpen` — solver is built from open-source code but kept
///   behind a feature gate / opt-in build.
/// * `RestrictedPlugin` — solver is loaded behind a restricted license
///   and the strict-open default never ships it. Downstream consumers
///   may load it under their own license terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LicenseClass {
    CoreOpen,
    OptionalOpen,
    RestrictedPlugin,
}

/// Coarse parallel-execution class declared by a solver implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelismClass {
    SingleCore,
    MultiCore,
    Gpu,
    Distributed,
}

/// Mirrors the `capabilities` object on `solver_card.schema.json`.
///
/// All fields are `Option` so an adapter can decline to assert a
/// capability; the planner treats absent fields as "undeclared / not
/// asserted", never as "unsupported".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SolverCapabilities {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_range_hz: Option<NumericRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solves_pec: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solves_dielectric: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solves_layered_dielectric: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solves_anisotropy: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solves_bistatic: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solves_monostatic: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_far_field: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_near_field: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_electrical_size_lambda: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallelism_class: Option<ParallelismClass>,
}

/// Mirrors the `inputs.frequency_sampling` enum on the schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrequencySampling {
    UserSpecified,
    Adaptive,
}

/// Mirrors the `inputs` object on `solver_card.schema.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SolverInputs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_mesh: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_material_card: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_polarization: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_sampling: Option<FrequencySampling>,
}

/// Mirrors the `outputs` object on `solver_card.schema.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SolverOutputs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub produces_rcs_cube: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub produces_currents: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub produces_near_field_volume: Option<bool>,
}

/// Mirrors the `convergence` object on `solver_card.schema.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SolverConvergence {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub produces_richardson_estimate: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub produces_cross_solver_delta: Option<bool>,
}

/// What a planner hands to a solver adapter when asking "can you do
/// this job and how big will it be?". Intentionally minimal — heavy
/// payloads (mesh tensors, dielectric stacks, etc.) are referenced by
/// id strings the adapter can resolve out-of-band.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverPlanRequest {
    /// Deterministic EchoForge id of the target object_card.
    pub object_card_id: String,
    /// Deterministic id of the chosen mesh_manifest, if a mesh is in
    /// play (PEC-only point-target solvers may skip).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_manifest_id: Option<String>,
    /// Deterministic id of the material_card, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material_card_id: Option<String>,
    /// Frequency band (closed interval, Hz) for the request.
    pub frequency_range_hz: NumericRange,
    /// TX polarization the planner intends to drive.
    pub tx_polarization: Polarization,
    /// RX polarization the planner intends to read.
    pub rx_polarization: Polarization,
    /// Whether the request is monostatic. `false` implies bistatic.
    pub monostatic: bool,
}

/// What a solver adapter returns when asked to plan a job. Captures
/// the bookkeeping the runner needs to schedule and the cost it should
/// expect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverPlan {
    /// Adapter-supplied opaque id used to thread plan→run.
    pub plan_id: String,
    /// Estimated frequency point count.
    pub estimated_frequency_points: u64,
    /// Estimated angular sample count (across both azimuth and
    /// elevation, as a single coarse total).
    pub estimated_angle_samples: u64,
    /// Estimated peak host or device memory, in megabytes.
    pub estimated_peak_memory_mb: u64,
    /// Adapter-supplied human-readable rationale / warnings emitted
    /// during planning.
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Per-product status emitted by a `run`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Ok,
    PartialFailure,
    Failed,
}

/// What a solver adapter returns from `run`. Captures only the
/// envelope: actual product tensors live on disk, referenced by
/// `output_paths`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverRunReport {
    /// Echo of the plan_id this run consumed.
    pub plan_id: String,
    /// Overall run status.
    pub status: RunStatus,
    /// Wallclock runtime in milliseconds. Adapters that cannot
    /// measure this may report 0.
    pub elapsed_ms: u64,
    /// Adapter-relative paths or URIs to produced artifacts, in
    /// emission order.
    #[serde(default)]
    pub output_paths: Vec<String>,
    /// Optional Richardson-extrapolation error estimate (dB) if the
    /// adapter declared `convergence.produces_richardson_estimate`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub richardson_error_db: Option<f64>,
    /// Adapter-emitted warnings / progress notes.
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Errors a `SolverAdapter` may report. Kept coarse on purpose;
/// per-adapter detail belongs in `notes` on the returned report.
#[derive(Debug, Error)]
pub enum SolverError {
    /// The solver does not support the requested job. Planners should
    /// treat this as "try another solver".
    #[error("solver does not support requested job: {0}")]
    Unsupported(String),
    /// The request was malformed or referenced unknown ids.
    #[error("invalid solver request: {0}")]
    InvalidRequest(String),
    /// The plan was malformed or not produced by this adapter.
    #[error("invalid solver plan: {0}")]
    InvalidPlan(String),
    /// The adapter ran but the underlying solver failed.
    #[error("solver runtime failure: {0}")]
    Runtime(String),
    /// Catch-all for adapter-internal errors that do not map cleanly
    /// to the above categories.
    #[error("solver adapter internal error: {0}")]
    Internal(String),
}

/// The plug-in trait every concrete EchoForge solver adapter
/// implements. Implementations live in their own crates (e.g. a
/// future `echoforge-solver-sagittasbr`), depend on this crate, and
/// register themselves into the runner via the adapter-registry
/// (out of scope for this packet).
pub trait SolverAdapter {
    /// Stable solver slug, e.g. `"sagittasbr"`. Should match the
    /// associated `solver_card.public_proxy_id`.
    fn name(&self) -> &str;

    /// Solver version string (semver or build tag). Should match the
    /// associated `solver_card.version`.
    fn version(&self) -> &str;

    /// License class for strict-open default packaging.
    fn license_class(&self) -> LicenseClass;

    /// Declared capabilities, mirroring the solver_card schema
    /// surface. The runner reconciles these against the on-disk
    /// solver_card before scheduling.
    fn capabilities(&self) -> &SolverCapabilities;

    /// Plan a job. Returns an estimate the runner uses for scheduling
    /// without committing to execution. Adapters that cannot service
    /// the request should return `SolverError::Unsupported`.
    fn plan(&self, request: &SolverPlanRequest) -> Result<SolverPlan, SolverError>;

    /// Execute a previously planned job. Adapters should validate the
    /// `plan_id` belongs to them and reject foreign plans.
    fn run(&self, plan: &SolverPlan) -> Result<SolverRunReport, SolverError>;
}

/// A no-op adapter used as a pending stand-in and for test scaffolding.
/// All capability flags are absent, all calls return
/// `SolverError::Unsupported`. License class is `CoreOpen` because
/// the null adapter ships with the strict-open default.
#[derive(Debug, Clone)]
pub struct NullSolver {
    name: String,
    version: String,
    capabilities: SolverCapabilities,
}

impl NullSolver {
    /// Construct a `NullSolver` with default identity strings.
    pub fn new() -> Self {
        Self {
            name: "null".to_string(),
            version: "0.0.0".to_string(),
            capabilities: SolverCapabilities::default(),
        }
    }

    /// Construct a `NullSolver` with caller-supplied identity strings
    /// (useful for tests that want a recognizable name).
    pub fn with_identity(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            capabilities: SolverCapabilities::default(),
        }
    }
}

impl Default for NullSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SolverAdapter for NullSolver {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn license_class(&self) -> LicenseClass {
        LicenseClass::CoreOpen
    }

    fn capabilities(&self) -> &SolverCapabilities {
        &self.capabilities
    }

    fn plan(&self, _request: &SolverPlanRequest) -> Result<SolverPlan, SolverError> {
        Err(SolverError::Unsupported(
            "NullSolver does not service any solver job".to_string(),
        ))
    }

    fn run(&self, _plan: &SolverPlan) -> Result<SolverRunReport, SolverError> {
        Err(SolverError::Unsupported(
            "NullSolver does not execute any solver plan".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> SolverPlanRequest {
        SolverPlanRequest {
            object_card_id: "ef:object_card:proxy-1:0123456789abcdef:1".to_string(),
            mesh_manifest_id: None,
            material_card_id: None,
            frequency_range_hz: NumericRange {
                min: 8.0e9,
                max: 12.0e9,
            },
            tx_polarization: Polarization::H,
            rx_polarization: Polarization::H,
            monostatic: true,
        }
    }

    fn sample_plan() -> SolverPlan {
        SolverPlan {
            plan_id: "null-plan-1".to_string(),
            estimated_frequency_points: 0,
            estimated_angle_samples: 0,
            estimated_peak_memory_mb: 0,
            notes: Vec::new(),
        }
    }

    #[test]
    fn null_solver_reports_expected_identity_and_license() {
        let s = NullSolver::new();
        assert_eq!(s.name(), "null");
        assert_eq!(s.version(), "0.0.0");
        assert_eq!(s.license_class(), LicenseClass::CoreOpen);
        // No capability flags asserted by default.
        assert!(s.capabilities().solves_pec.is_none());
        assert!(s.capabilities().solves_monostatic.is_none());
        assert!(s.capabilities().parallelism_class.is_none());
    }

    #[test]
    fn null_solver_returns_unsupported_on_plan_and_run() {
        let s = NullSolver::new();
        match s.plan(&sample_request()) {
            Err(SolverError::Unsupported(_)) => {}
            other => panic!("expected Unsupported from plan, got {other:?}"),
        }
        match s.run(&sample_plan()) {
            Err(SolverError::Unsupported(_)) => {}
            other => panic!("expected Unsupported from run, got {other:?}"),
        }
    }

    #[test]
    fn null_solver_with_identity_overrides_name_and_version() {
        let s = NullSolver::with_identity("test-null", "9.9.9");
        assert_eq!(s.name(), "test-null");
        assert_eq!(s.version(), "9.9.9");
        // License class is still CoreOpen even with custom identity.
        assert_eq!(s.license_class(), LicenseClass::CoreOpen);
    }

    #[test]
    fn license_class_round_trips_through_serde() {
        for (variant, wire) in [
            (LicenseClass::CoreOpen, "\"core_open\""),
            (LicenseClass::OptionalOpen, "\"optional_open\""),
            (LicenseClass::RestrictedPlugin, "\"restricted_plugin\""),
        ] {
            let encoded = serde_json::to_string(&variant).expect("encode");
            assert_eq!(encoded, wire, "wire form for {variant:?}");
            let decoded: LicenseClass = serde_json::from_str(&encoded).expect("decode");
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn parallelism_class_round_trips_through_serde() {
        for (variant, wire) in [
            (ParallelismClass::SingleCore, "\"single_core\""),
            (ParallelismClass::MultiCore, "\"multi_core\""),
            (ParallelismClass::Gpu, "\"gpu\""),
            (ParallelismClass::Distributed, "\"distributed\""),
        ] {
            let encoded = serde_json::to_string(&variant).expect("encode");
            assert_eq!(encoded, wire);
            let decoded: ParallelismClass = serde_json::from_str(&encoded).expect("decode");
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn capabilities_round_trip_skips_absent_fields() {
        // Fully populated: round-trip preserves every field.
        let full = SolverCapabilities {
            frequency_range_hz: Some(NumericRange {
                min: 1.0e9,
                max: 18.0e9,
            }),
            solves_pec: Some(true),
            solves_dielectric: Some(false),
            solves_layered_dielectric: Some(false),
            solves_anisotropy: Some(false),
            solves_bistatic: Some(true),
            solves_monostatic: Some(true),
            supports_far_field: Some(true),
            supports_near_field: Some(false),
            max_electrical_size_lambda: Some(64.0),
            parallelism_class: Some(ParallelismClass::Gpu),
        };
        let encoded = serde_json::to_string(&full).expect("encode full");
        let decoded: SolverCapabilities = serde_json::from_str(&encoded).expect("decode full");
        assert_eq!(decoded, full);

        // Empty capabilities serialize to an empty object (all fields
        // skipped); decode round-trips back to default.
        let empty = SolverCapabilities::default();
        let encoded_empty = serde_json::to_string(&empty).expect("encode empty");
        assert_eq!(encoded_empty, "{}");
        let decoded_empty: SolverCapabilities =
            serde_json::from_str(&encoded_empty).expect("decode empty");
        assert_eq!(decoded_empty, empty);
    }

    #[test]
    fn inputs_outputs_convergence_round_trip() {
        let inputs = SolverInputs {
            requires_mesh: Some(true),
            requires_material_card: Some(true),
            requires_polarization: Some(true),
            frequency_sampling: Some(FrequencySampling::Adaptive),
        };
        let outputs = SolverOutputs {
            produces_rcs_cube: Some(true),
            produces_currents: Some(false),
            produces_near_field_volume: Some(false),
        };
        let convergence = SolverConvergence {
            produces_richardson_estimate: Some(true),
            produces_cross_solver_delta: Some(true),
        };

        for (name, value) in [
            (
                "inputs",
                serde_json::to_string(&inputs).expect("encode inputs"),
            ),
            (
                "outputs",
                serde_json::to_string(&outputs).expect("encode outputs"),
            ),
            (
                "convergence",
                serde_json::to_string(&convergence).expect("encode convergence"),
            ),
        ] {
            // Each encoded form should be a JSON object.
            assert!(value.starts_with('{'), "{name} did not encode as object");
        }

        let inputs_back: SolverInputs =
            serde_json::from_str(&serde_json::to_string(&inputs).unwrap()).unwrap();
        assert_eq!(inputs_back, inputs);
        let outputs_back: SolverOutputs =
            serde_json::from_str(&serde_json::to_string(&outputs).unwrap()).unwrap();
        assert_eq!(outputs_back, outputs);
        let convergence_back: SolverConvergence =
            serde_json::from_str(&serde_json::to_string(&convergence).unwrap()).unwrap();
        assert_eq!(convergence_back, convergence);
    }

    #[test]
    fn solver_error_messages_are_distinct_per_variant() {
        let unsupported = SolverError::Unsupported("no can do".to_string()).to_string();
        let invalid_req = SolverError::InvalidRequest("missing mesh".to_string()).to_string();
        let invalid_plan = SolverError::InvalidPlan("wrong adapter".to_string()).to_string();
        let runtime = SolverError::Runtime("solver crashed".to_string()).to_string();
        let internal = SolverError::Internal("oops".to_string()).to_string();
        // All five should be distinct human-readable strings carrying
        // their inner message.
        for (name, text, needle) in [
            ("unsupported", &unsupported, "no can do"),
            ("invalid_request", &invalid_req, "missing mesh"),
            ("invalid_plan", &invalid_plan, "wrong adapter"),
            ("runtime", &runtime, "solver crashed"),
            ("internal", &internal, "oops"),
        ] {
            assert!(
                text.contains(needle),
                "{name} error did not include inner message: {text}"
            );
        }
        // Distinctness: at minimum, no two messages collide.
        let all = [
            unsupported.clone(),
            invalid_req.clone(),
            invalid_plan.clone(),
            runtime.clone(),
            internal.clone(),
        ];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "duplicate error text at {i},{j}");
            }
        }
    }
}
