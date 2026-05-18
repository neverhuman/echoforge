//! Truth-vs-candidate comparator. Emits an `echoforge_core::ValidationCheck`.

use echoforge_core::ValidationCheck;

use crate::primitives::{Status, Truth};
use crate::tolerance::ToleranceBand;
use crate::units::sigma_to_dbsm;

pub fn check(name: &str, truth: &Truth, candidate: f64, tol: &ToleranceBand) -> ValidationCheck {
    if matches!(truth.status, Status::Skipped) {
        return ValidationCheck {
            name: name.to_string(),
            status: "warn".to_string(),
            message: format!("skipped: {}", truth.regime),
        };
    }
    if candidate <= 0.0 || truth.value <= 0.0 {
        return ValidationCheck {
            name: name.to_string(),
            status: "fail".to_string(),
            message: "non-positive sigma cannot be compared in dB".to_string(),
        };
    }
    let delta_db = (sigma_to_dbsm(candidate) - sigma_to_dbsm(truth.value)).abs();
    let status = if delta_db <= tol.total_db {
        "pass"
    } else if delta_db <= tol.total_db + 0.5 {
        "warn"
    } else {
        "fail"
    };
    ValidationCheck {
        name: name.to_string(),
        status: status.to_string(),
        message: format!(
            "delta={delta_db:.3} dB tol={tol_db:.3} dB regime={regime}",
            tol_db = tol.total_db,
            regime = truth.regime
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_when_within_tolerance() {
        let truth = Truth::ok(1.0, "broadside");
        let tol = ToleranceBand {
            total_db: 0.5,
            ..ToleranceBand::default()
        };
        let c = check("x", &truth, 1.0, &tol);
        assert_eq!(c.status, "pass");
    }

    #[test]
    fn fail_when_outside() {
        let truth = Truth::ok(1.0, "broadside");
        let tol = ToleranceBand {
            total_db: 0.2,
            ..ToleranceBand::default()
        };
        let c = check("x", &truth, 10.0, &tol);
        assert_eq!(c.status, "fail");
    }
}
