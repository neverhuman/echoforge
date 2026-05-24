//! Unwired SagittaSBR solver adapter for EchoForge.
//!
//! SagittaSBR is an open-source GPU-accelerated shooting-and-bouncing-rays
//! (SBR) RCS solver. EchoForge's planner/runner consumes solvers through
//! the `SolverAdapter` trait defined in `echoforge-solver-contract`; this
//! crate provides the registration surface for SagittaSBR so the planner
//! can already reason about it.
//!
//! **This is an UNWIRED ADAPTER.** It does NOT compute RCS. The real compute
//! path requires a SagittaSBR container/binary which is not yet bundled or
//! pinned by digest in this repository. Until that lands:
//!
//! * `plan()` only succeeds when an adapter instance has been constructed
//!   with [`SagittaSbrAdapter::with_binary`]; otherwise it returns
//!   `SolverError::Unsupported` carrying a "binary not configured" note.
//! * `run()` ALWAYS returns `SolverError::Unsupported` — even when the
//!   binary path is set — with a message explicitly identifying this as
//!   the unconnected compute path. This is intentional: the unwired adapter
//!   will never fabricate RCS numbers.
//!
//! Capabilities are populated from SagittaSBR's publicly-documented
//! capability surface (GPU-accelerated SBR, 100 MHz – 100 GHz, PEC and
//! lossy isotropic dielectric, monostatic and bistatic far-field, no
//! native layered/anisotropic stack, electrically large up to ~1000 λ).
//! These remain honest declarations even though no compute happens —
//! they let the planner pick or reject SagittaSBR for a given request
//! before it discovers the compute path is unwired.
//!
//! See `.agents/receipts/sagittasbr-adapter/<UTC>.md` for the wiring
//! plan toward the real adapter (Docker image, digest pin, ABI/JSON
//! wire-up, container fan-out).

use std::path::PathBuf;

use echoforge_core::NumericRange;
use echoforge_solver_contract::{
    LicenseClass, ParallelismClass, RunStatus, SolverAdapter, SolverCapabilities, SolverError,
    SolverPlan, SolverPlanRequest, SolverRunReport,
};

/// Solver slug emitted by [`SagittaSbrAdapter::name`].
///
/// Matches the `solver_card.public_proxy_id` slug rules
/// (lowercase, `._-`-separated) so a future on-disk solver_card can
/// be reconciled against this adapter.
pub const ADAPTER_NAME: &str = "sagitta-sbr";

/// Version tag emitted by [`SagittaSbrAdapter::version`].
///
/// The `unwired-` prefix is load-bearing: planners and operators can grep
/// for it to confirm the deployed adapter is the unconnected build, not a
/// wired build that happens to share the slug.
pub const ADAPTER_VERSION: &str = "unwired-0.1.0";

/// Unwired SagittaSBR adapter.
///
/// Holds the bookkeeping a future real adapter will need (binary
/// location, container digest for reproducibility) but does not
/// shell out or invoke any external process.
#[derive(Debug, Clone)]
pub struct SagittaSbrAdapter {
    /// Path to a SagittaSBR container entrypoint or native binary.
    /// `None` when the adapter is unwired: it is constructible without
    /// a binary so the registry can advertise the slug, but
    /// `plan()` / `run()` will refuse to do useful work.
    pub binary_path: Option<PathBuf>,
    /// OCI container digest used to verify reproducibility when the
    /// compute path is wired up. Carried for forward-bridged with the
    /// future real adapter; the unwired adapter does not consult it.
    pub container_digest: Option<String>,
    /// Cached capability declaration. Held as an owned struct so the
    /// `capabilities()` trait method can return a `&` reference per
    /// the `SolverAdapter` contract.
    capabilities: SolverCapabilities,
}

impl SagittaSbrAdapter {
    /// Construct an unwired adapter. `is_available()` returns `false`.
    pub fn new() -> Self {
        Self {
            binary_path: None,
            container_digest: None,
            capabilities: default_capabilities(),
        }
    }

    /// Construct an adapter that points at a SagittaSBR binary or
    /// container entrypoint. `is_available()` returns `true`. The path
    /// is recorded as-is and is NOT validated for existence — the
    /// unwired adapter does not execute anything; existence checks belong
    /// to the future real adapter.
    pub fn with_binary(path: impl Into<PathBuf>) -> Self {
        Self {
            binary_path: Some(path.into()),
            container_digest: None,
            capabilities: default_capabilities(),
        }
    }

    /// Record an OCI container digest. The unwired adapter stores but does
    /// not use this; the real adapter will verify it before each `run()`.
    pub fn with_container_digest(mut self, digest: impl Into<String>) -> Self {
        self.container_digest = Some(digest.into());
        self
    }

    /// `true` when a binary path has been configured. The unwired adapter
    /// treats this as the sole gate on whether `plan()` is allowed to
    /// return a populated plan; `run()` ignores it (never executes).
    pub fn is_available(&self) -> bool {
        self.binary_path.is_some()
    }
}

impl Default for SagittaSbrAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the honest capability declaration for SagittaSBR.
///
/// Populated from SagittaSBR's published capability surface; conservative
/// where the upstream is ambiguous. All flags are `Some(...)` so the
/// planner knows the adapter has affirmatively declared them (an
/// an unwired-but-declared adapter is still useful for scheduler dry-runs).
fn default_capabilities() -> SolverCapabilities {
    SolverCapabilities {
        // SBR is a high-frequency method; the upstream targets the
        // microwave / mm-wave band. 100 MHz to 100 GHz captures the
        // practical envelope without overclaiming low-end accuracy.
        frequency_range_hz: Some(NumericRange {
            min: 1.0e8,
            max: 1.0e11,
        }),
        solves_pec: Some(true),
        solves_dielectric: Some(true),
        // SagittaSBR's open release does not natively solve stratified
        // dielectric stacks; layered handling is a downstream extension.
        solves_layered_dielectric: Some(false),
        // Anisotropic (tensor-permittivity) materials are not part of
        // the open release.
        solves_anisotropy: Some(false),
        solves_bistatic: Some(true),
        solves_monostatic: Some(true),
        supports_far_field: Some(true),
        // The open release focuses on far-field scattering; native
        // near-field volumetric output is not declared.
        supports_near_field: Some(false),
        // SBR scales to electrically large targets; ~1000 λ is a
        // conservative-but-honest ceiling for the open release.
        max_electrical_size_lambda: Some(1000.0),
        parallelism_class: Some(ParallelismClass::Gpu),
    }
}

impl SolverAdapter for SagittaSbrAdapter {
    fn name(&self) -> &str {
        ADAPTER_NAME
    }

    fn version(&self) -> &str {
        ADAPTER_VERSION
    }

    fn license_class(&self) -> LicenseClass {
        // SagittaSBR is open-source but is built behind a feature gate
        // (it pulls in a GPU container/binary at runtime), which is
        // exactly what `OptionalOpen` captures.
        LicenseClass::OptionalOpen
    }

    fn capabilities(&self) -> &SolverCapabilities {
        &self.capabilities
    }

    fn plan(&self, request: &SolverPlanRequest) -> Result<SolverPlan, SolverError> {
        if !self.is_available() {
            return Err(SolverError::Unsupported(format!(
                "{ADAPTER_NAME} adapter: SagittaSBR binary not configured \
                 (construct via SagittaSbrAdapter::with_binary)"
            )));
        }

        // Sanity-check the request against declared capabilities. These
        // checks are the same a wired adapter would perform; they let
        // the planner exercise rejection paths against the unwired adapter.
        if request.monostatic {
            if self.capabilities.solves_monostatic != Some(true) {
                return Err(SolverError::Unsupported(format!(
                    "{ADAPTER_NAME} adapter: monostatic not declared",
                )));
            }
        } else if self.capabilities.solves_bistatic != Some(true) {
            return Err(SolverError::Unsupported(format!(
                "{ADAPTER_NAME} adapter: bistatic not declared",
            )));
        }

        if let Some(declared) = self.capabilities.frequency_range_hz.as_ref() {
            if request.frequency_range_hz.min < declared.min
                || request.frequency_range_hz.max > declared.max
            {
                return Err(SolverError::InvalidRequest(format!(
                    "{ADAPTER_NAME} adapter: requested frequency band \
                     [{} Hz, {} Hz] is outside declared range [{} Hz, {} Hz]",
                    request.frequency_range_hz.min,
                    request.frequency_range_hz.max,
                    declared.min,
                    declared.max,
                )));
            }
        }

        // Return an honest plan envelope: deterministic plan_id, zero
        // cost estimates (the unwired adapter cannot estimate — that requires
        // the wired binary), and an explicit note so callers cannot
        // mistake unwired output for measured estimates.
        Ok(SolverPlan {
            plan_id: format!("{ADAPTER_NAME}-unwired-plan:{}", request.object_card_id),
            estimated_frequency_points: 0,
            estimated_angle_samples: 0,
            estimated_peak_memory_mb: 0,
            notes: vec![
                format!(
                    "{ADAPTER_NAME} adapter: cost estimates omitted \
                     (binary at {:?} not invoked; adapter is unwired)",
                    self.binary_path,
                ),
                "SagittaSBR container not yet connected; wire binary_path and container_digest to enable estimates".to_string(),
            ],
        })
    }

    fn run(&self, plan: &SolverPlan) -> Result<SolverRunReport, SolverError> {
        // Guard against foreign plans first so failure modes are
        // reported in the order a wired adapter would surface them.
        if !plan.plan_id.starts_with(&format!("{ADAPTER_NAME}-")) {
            return Err(SolverError::InvalidPlan(format!(
                "{ADAPTER_NAME} adapter: plan_id {:?} not produced by this adapter",
                plan.plan_id,
            )));
        }

        // Even with the binary set, the unwired adapter MUST NOT fabricate
        // output. The honest answer is Unsupported with a message that
        // explicitly names this as the unconnected compute path so operators
        // can grep for it in logs.
        Err(SolverError::Unsupported(format!(
            "{ADAPTER_NAME} adapter: compute path not yet wired \
             (no SagittaSBR container/binary integration in this build)"
        )))
    }
}

// Suppress reserved-import note when the reserved RunStatus re-export is
// reached only by downstream callers. Keeping the import in scope means
// docs and future wired-adapter code can refer to it without churn.
const _: Option<RunStatus> = None;

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
