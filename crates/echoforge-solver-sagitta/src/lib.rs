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
mod tests {
    use super::*;

    fn sample_request() -> SolverPlanRequest {
        SolverPlanRequest {
            object_card_id: "ef:object_card:proxy-1:0123456789abcdef:1".to_string(),
            mesh_manifest_id: Some("ef:mesh_manifest:proxy-1:fedcba9876543210:1".to_string()),
            material_card_id: None,
            frequency_range_hz: NumericRange {
                min: 8.0e9,
                max: 12.0e9,
            },
            tx_polarization: echoforge_solver_contract::Polarization::H,
            rx_polarization: echoforge_solver_contract::Polarization::H,
            monostatic: true,
        }
    }

    #[test]
    fn new_adapter_is_not_available() {
        let a = SagittaSbrAdapter::new();
        assert!(
            !a.is_available(),
            "freshly-constructed adapter must be unwired"
        );
        assert!(a.binary_path.is_none());
        assert!(a.container_digest.is_none());
    }

    #[test]
    fn with_binary_makes_adapter_available() {
        let a = SagittaSbrAdapter::with_binary("/fake/path/to/sagittasbr");
        assert!(a.is_available());
        assert_eq!(
            a.binary_path.as_deref(),
            Some(std::path::Path::new("/fake/path/to/sagittasbr"))
        );
    }

    #[test]
    fn with_container_digest_records_digest_without_changing_availability() {
        let a = SagittaSbrAdapter::new().with_container_digest(
            "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd",
        );
        assert!(!a.is_available(), "digest alone must not flip availability");
        assert!(a.container_digest.is_some());
    }

    #[test]
    fn identity_and_license_class_are_constant() {
        let a = SagittaSbrAdapter::new();
        assert_eq!(a.name(), "sagitta-sbr");
        assert_eq!(a.name(), ADAPTER_NAME);
        assert!(
            a.version().starts_with("unwired-"),
            "version must be unwired-tagged; got {:?}",
            a.version()
        );
        assert_eq!(a.license_class(), LicenseClass::OptionalOpen);

        // Wiring a binary must NOT change identity or license class.
        let b = SagittaSbrAdapter::with_binary("/fake");
        assert_eq!(b.name(), a.name());
        assert_eq!(b.version(), a.version());
        assert_eq!(b.license_class(), a.license_class());
    }

    #[test]
    fn capabilities_declare_honest_sbr_surface() {
        let caps = SagittaSbrAdapter::new().capabilities().clone();
        assert_eq!(caps.solves_pec, Some(true));
        assert_eq!(caps.solves_dielectric, Some(true));
        assert_eq!(caps.solves_layered_dielectric, Some(false));
        assert_eq!(caps.solves_anisotropy, Some(false));
        assert_eq!(caps.solves_monostatic, Some(true));
        assert_eq!(caps.solves_bistatic, Some(true));
        assert_eq!(caps.supports_far_field, Some(true));
        assert_eq!(caps.supports_near_field, Some(false));
        assert_eq!(caps.parallelism_class, Some(ParallelismClass::Gpu));
        let band = caps
            .frequency_range_hz
            .as_ref()
            .expect("frequency band declared");
        assert!(band.min > 0.0 && band.min < band.max);
        assert!(
            caps.max_electrical_size_lambda.unwrap_or(0.0) > 0.0,
            "max electrical size must be positive"
        );
    }

    #[test]
    fn plan_on_unavailable_adapter_returns_unsupported() {
        let a = SagittaSbrAdapter::new();
        match a.plan(&sample_request()) {
            Err(SolverError::Unsupported(msg)) => {
                let m = msg.to_lowercase();
                assert!(
                    m.contains("not configured"),
                    "unavailable plan() error should mention not configured; got {msg:?}",
                );
            }
            other => panic!("expected Unsupported from plan() on unwired adapter, got {other:?}"),
        }
    }

    #[test]
    fn plan_on_available_adapter_returns_populated_plan() {
        let a = SagittaSbrAdapter::with_binary("/fake/path/to/sagittasbr");
        let plan = a.plan(&sample_request()).expect("plan on unwired adapter");
        assert!(
            plan.plan_id.starts_with("sagitta-sbr-"),
            "plan_id must be namespaced by this adapter; got {:?}",
            plan.plan_id,
        );
        assert!(
            !plan.notes.is_empty(),
            "unwired plan must carry at least one note"
        );
        // Cost estimates are intentionally zero in the unwired adapter.
        // Document that expectation so a future wired adapter that starts
        // returning real estimates trips the test and forces a review.
        assert_eq!(plan.estimated_frequency_points, 0);
        assert_eq!(plan.estimated_angle_samples, 0);
        assert_eq!(plan.estimated_peak_memory_mb, 0);
        let joined = plan.notes.join(" ").to_lowercase();
        assert!(
            joined.contains("unwired") || joined.contains("not yet connected"),
            "plan notes must flag the unwired status; got {:?}",
            plan.notes,
        );
    }

    #[test]
    fn plan_rejects_out_of_band_frequency_request() {
        let mut req = sample_request();
        req.frequency_range_hz = NumericRange {
            min: 1.0e12, // 1 THz, well above declared 100 GHz ceiling
            max: 2.0e12,
        };
        let a = SagittaSbrAdapter::with_binary("/fake");
        match a.plan(&req) {
            Err(SolverError::InvalidRequest(msg)) => {
                assert!(
                    msg.to_lowercase().contains("outside declared range"),
                    "frequency rejection should name the declared range; got {msg:?}",
                );
            }
            other => panic!("expected InvalidRequest for out-of-band frequency, got {other:?}"),
        }
    }

    #[test]
    fn run_on_available_adapter_returns_unsupported_unwired_message() {
        let a = SagittaSbrAdapter::with_binary("/fake/path/to/sagittasbr");
        let plan = a
            .plan(&sample_request())
            .expect("plan succeeds on unwired adapter");
        match a.run(&plan) {
            Err(SolverError::Unsupported(msg)) => {
                let m = msg.to_lowercase();
                assert!(
                    m.contains("not yet wired") || m.contains("not wired"),
                    "run() error MUST say compute path is not wired; got {msg:?}",
                );
            }
            Ok(report) => panic!(
                "unwired run() must NEVER return Ok — got report {report:?}. \
                 Unwired adapters that return fake reports would silently corrupt \
                 downstream validation."
            ),
            other => panic!("expected Unsupported from run(), got {other:?}"),
        }
    }

    #[test]
    fn run_rejects_foreign_plan_id() {
        let a = SagittaSbrAdapter::with_binary("/fake");
        let foreign = SolverPlan {
            plan_id: "null-plan-1".to_string(),
            estimated_frequency_points: 0,
            estimated_angle_samples: 0,
            estimated_peak_memory_mb: 0,
            notes: Vec::new(),
        };
        match a.run(&foreign) {
            Err(SolverError::InvalidPlan(msg)) => {
                assert!(
                    msg.contains("not produced by this adapter"),
                    "foreign-plan rejection should name the cause; got {msg:?}",
                );
            }
            other => panic!("expected InvalidPlan for foreign plan_id, got {other:?}"),
        }
    }

    #[test]
    fn capabilities_round_trip_through_serde() {
        let caps = SagittaSbrAdapter::new().capabilities().clone();
        let encoded = serde_json::to_string(&caps).expect("encode caps");
        let decoded: SolverCapabilities = serde_json::from_str(&encoded).expect("decode caps");
        assert_eq!(decoded, caps);
        // Spot-check that the GPU enum survived as the expected wire form.
        assert!(
            encoded.contains("\"gpu\""),
            "encoded caps missing GPU class: {encoded}"
        );
    }

    #[test]
    fn adapter_object_safe_via_trait_object() {
        // Exercise SagittaSbrAdapter through a &dyn SolverAdapter so any
        // future change that breaks object-safety on the contract trait
        // trips here, not in a downstream registry crate.
        let a: Box<dyn SolverAdapter> = Box::new(SagittaSbrAdapter::with_binary("/fake"));
        assert_eq!(a.name(), ADAPTER_NAME);
        assert_eq!(a.license_class(), LicenseClass::OptionalOpen);
        let plan = a.plan(&sample_request()).expect("plan via trait object");
        assert!(plan.plan_id.starts_with("sagitta-sbr-"));
        assert!(matches!(a.run(&plan), Err(SolverError::Unsupported(_))));
    }
}
