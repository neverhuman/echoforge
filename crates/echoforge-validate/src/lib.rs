//! EchoForge validation crate.
//!
//! Strict-open analytic ground-truth library for canonical scatterers
//! (PEC sphere via Mie series + closed-form plate/dihedral/trihedral/cylinder/cone)
//! plus the V-tier promotion gate consumed by the `ef validate` CLI subcommand.
//!
//! All algorithmic implementations are original; references are cited inline
//! (Wiscombe 1980 NCAR/TN-140+STR; Bohren & Huffman 1983; Ruck RCS Handbook).

pub mod bessel;
pub mod cli;
pub mod compare;
pub mod convergence;
pub mod cross_solver;
pub mod determinism;
pub mod distribution_metrics;
pub mod error;
pub mod fidelity_rollup;
pub mod micro_doppler;
pub mod polarization;
pub mod primitives;
pub mod report;
pub mod tier_benchmarked;
pub mod tier_benchmarked_json;
pub mod tier_measured_anchored;
pub mod tolerance;
pub mod uncertainty;
pub mod units;

pub use cli::{run as run_validate, ValidateArgs, ValidateError};
pub use compare::check;
pub use convergence::{richardson, ConvergenceReport};
pub use cross_solver::{CrossSolverDelta, DeltaPair, DeltaSummary};
pub use determinism::{bit_identical_f64, fp_tolerant_f32, DeterminismReport};
pub use distribution_metrics::{ks_distance_1d_sorted, wasserstein_1d_sorted};
pub use fidelity_rollup::{
    assert_fidelity_floor, parse_fidelity_class, rollup_validation_envelopes, FidelityFloor,
    FidelityFloorError, FidelityRollup,
};
pub use polarization::completeness_check;
pub use primitives::{
    cone::PecCone,
    cylinder::PecCylinder,
    dihedral::PecDihedral,
    flat_plate::PecFlatPlate,
    sphere::PecSphere,
    trihedral::{PecTrihedral, TrihedralShape},
    CanonicalTruth, Conditions, Status, Truth,
};
pub use report::{ErrorBudget, TierAchieved, ValidateChecks, ValidateReport};
pub use tier_benchmarked::{
    evaluate_v3_gate, tier_alias_benchmarked, GateStatus, V3GateReport, V3MetricObservation,
    V3MetricThresholds,
};
pub use tier_benchmarked_json::evaluate_v3_from_benchmark_json;
pub use tier_measured_anchored::{
    evaluate_v4_from_calibration_report, evaluate_v4_gate, tier_alias_measured_anchored,
    AnchorMetric, CalibrationReportFile, V4AnchorOutcome, V4DistributionAnchor,
    V4DistributionObservation, V4GateReport, V4MetricThresholds,
};
pub use tolerance::{combine as combine_tolerance, ToleranceBand};
pub use uncertainty::{confidence_from_sigma_db, linearize};
pub use units::{
    dbsm_to_sigma, deg_to_rad, rad_to_deg, sigma_to_dbsm, DbSm, Degrees, Frame, Hz, Meters,
    Polarization, Radians, SigmaM2, SPEED_OF_LIGHT,
};
