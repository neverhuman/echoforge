use serde::{Deserialize, Serialize};

use crate::artifact::ValidationTier;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnalyticPrimitive {
    PecSphere,
    FlatPlate,
    Dihedral,
    Trihedral,
    Cylinder,
    Cone,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStatus {
    Pending,
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnalyticValidationCase {
    pub primitive: AnalyticPrimitive,
    pub frequency_hz: f64,
    pub expected_rcs_dbsm: Option<f64>,
    pub measured_rcs_dbsm: Option<f64>,
    pub tolerance_db: f64,
    pub status: ValidationStatus,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnalyticValidationReport {
    pub validation_tier: ValidationTier,
    pub cases: Vec<AnalyticValidationCase>,
    pub passed: bool,
    pub summary: String,
}

pub fn pending_analytic_report() -> AnalyticValidationReport {
    let cases = vec![
        pending_case(AnalyticPrimitive::PecSphere, 10.0e9),
        pending_case(AnalyticPrimitive::FlatPlate, 9.6e9),
        pending_case(AnalyticPrimitive::Dihedral, 9.2e9),
        pending_case(AnalyticPrimitive::Trihedral, 8.8e9),
        pending_case(AnalyticPrimitive::Cylinder, 8.4e9),
        pending_case(AnalyticPrimitive::Cone, 8.0e9),
    ];

    AnalyticValidationReport {
        validation_tier: ValidationTier::Pending,
        cases,
        passed: false,
        summary: "analytic validation pending: cases are wired but not yet scored".to_string(),
    }
}

fn pending_case(primitive: AnalyticPrimitive, frequency_hz: f64) -> AnalyticValidationCase {
    AnalyticValidationCase {
        primitive,
        frequency_hz,
        expected_rcs_dbsm: None,
        measured_rcs_dbsm: None,
        tolerance_db: 0.0,
        status: ValidationStatus::Pending,
        notes: Some("pending validation record".to_string()),
    }
}
