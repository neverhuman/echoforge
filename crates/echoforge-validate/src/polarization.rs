//! Polarization completeness check.

use echoforge_core::ValidationCheck;

use crate::units::Polarization;

/// Required: {(V,V),(H,H),(V,H),(H,V)}. Returns "pass" if all four pairs are
/// present, "warn" if a strict subset is documented but not complete.
pub fn completeness_check(present: &[(Polarization, Polarization)]) -> ValidationCheck {
    let required = [
        (Polarization::V, Polarization::V),
        (Polarization::H, Polarization::H),
        (Polarization::V, Polarization::H),
        (Polarization::H, Polarization::V),
    ];
    let mut missing = Vec::new();
    for r in required.iter() {
        if !present.iter().any(|p| p == r) {
            missing.push(format!("{:?}{:?}", r.0, r.1));
        }
    }
    let status = if missing.is_empty() { "pass" } else { "warn" };
    let message = if missing.is_empty() {
        "all four pol pairs present".to_string()
    } else {
        format!("missing pol pairs: {}", missing.join(","))
    };
    ValidationCheck {
        name: "polarization_completeness".to_string(),
        status: status.to_string(),
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_with_all_four() {
        let p = vec![
            (Polarization::V, Polarization::V),
            (Polarization::H, Polarization::H),
            (Polarization::V, Polarization::H),
            (Polarization::H, Polarization::V),
        ];
        let c = completeness_check(&p);
        assert_eq!(c.status, "pass");
    }

    #[test]
    fn warns_when_subset() {
        let p = vec![(Polarization::V, Polarization::V)];
        let c = completeness_check(&p);
        assert_eq!(c.status, "warn");
    }
}
