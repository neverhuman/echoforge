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
pub mod error;
pub mod micro_doppler;
pub mod polarization;
pub mod primitives;
pub mod report;
pub mod tolerance;
pub mod uncertainty;
pub mod units;

pub use cli::{run as run_validate, ValidateArgs, ValidateError};
pub use compare::check;
pub use convergence::{richardson, ConvergenceReport};
pub use cross_solver::{CrossSolverDelta, DeltaPair, DeltaSummary};
pub use determinism::{bit_identical_f64, fp_tolerant_f32, DeterminismReport};
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
pub use tolerance::{combine as combine_tolerance, ToleranceBand};
pub use uncertainty::{confidence_from_sigma_db, linearize};
pub use units::{
    dbsm_to_sigma, deg_to_rad, rad_to_deg, sigma_to_dbsm, DbSm, Degrees, Frame, Hz, Meters,
    Polarization, Radians, SigmaM2, SPEED_OF_LIGHT,
};
