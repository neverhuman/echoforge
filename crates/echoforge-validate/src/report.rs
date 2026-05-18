//! Validation report aggregation + V-tier promotion gate.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum TierAchieved {
    V0,
    V1,
    V2,
}

impl TierAchieved {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "v0" => Some(TierAchieved::V0),
            "v1" => Some(TierAchieved::V1),
            "v2" => Some(TierAchieved::V2),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            TierAchieved::V0 => "V0",
            TierAchieved::V1 => "V1",
            TierAchieved::V2 => "V2",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ErrorBudget {
    pub analytic_db: f64,
    pub numeric_db: f64,
    pub method_db: f64,
    pub total_db: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidateChecks {
    pub canonical_validation_present: bool,
    pub canonical_overall_pass: bool,
    pub polarization_complete: bool,
    pub determinism_pass: bool,
    pub units_frame_pass: bool,
    pub cross_solver_present: bool,
    pub cross_solver_pass: bool,
    pub convergence_present: bool,
    pub convergence_pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateReport {
    pub tier: TierAchieved,
    pub overall_status: String,
    pub primitives_checked: Vec<String>,
    pub n_pass: usize,
    pub n_warn: usize,
    pub n_fail: usize,
    pub error_budget: ErrorBudget,
    pub checks: ValidateChecks,
    pub notes: Vec<String>,
}

/// V-tier promotion gate per plan §6.
///
/// * V0 — always passes (declared only).
/// * V1 — canonical pass + report present + polarization complete + determinism + units/frame.
/// * V2 — V1 + cross-solver delta pass + convergence pass.
pub fn promotion_gate(target: TierAchieved, checks: &ValidateChecks) -> bool {
    match target {
        TierAchieved::V0 => true,
        TierAchieved::V1 => {
            checks.canonical_validation_present
                && checks.canonical_overall_pass
                && checks.polarization_complete
                && checks.determinism_pass
                && checks.units_frame_pass
        }
        TierAchieved::V2 => {
            promotion_gate(TierAchieved::V1, checks)
                && checks.cross_solver_present
                && checks.cross_solver_pass
                && checks.convergence_present
                && checks.convergence_pass
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v0_always_passes() {
        let c = ValidateChecks::default();
        assert!(promotion_gate(TierAchieved::V0, &c));
    }

    #[test]
    fn v1_requires_all_five() {
        let mut c = ValidateChecks::default();
        assert!(!promotion_gate(TierAchieved::V1, &c));
        c.canonical_validation_present = true;
        c.canonical_overall_pass = true;
        c.polarization_complete = true;
        c.determinism_pass = true;
        c.units_frame_pass = true;
        assert!(promotion_gate(TierAchieved::V1, &c));
    }

    #[test]
    fn v2_requires_v1_plus_cross_and_conv() {
        let mut c = ValidateChecks::default();
        c.canonical_validation_present = true;
        c.canonical_overall_pass = true;
        c.polarization_complete = true;
        c.determinism_pass = true;
        c.units_frame_pass = true;
        assert!(!promotion_gate(TierAchieved::V2, &c));
        c.cross_solver_present = true;
        c.cross_solver_pass = true;
        c.convergence_present = true;
        c.convergence_pass = true;
        assert!(promotion_gate(TierAchieved::V2, &c));
    }
}
