//! Cross-solver Δ analysis.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaPair {
    pub source_a: String,
    pub source_b: String,
    pub sigma_a_dbsm: f64,
    pub sigma_b_dbsm: f64,
    pub delta_db: f64,
    pub tolerance_db: f64,
    pub status: String,
}

impl DeltaPair {
    pub fn new(
        source_a: &str,
        source_b: &str,
        sigma_a_dbsm: f64,
        sigma_b_dbsm: f64,
        tolerance_db: f64,
    ) -> Self {
        let delta_db = (sigma_a_dbsm - sigma_b_dbsm).abs();
        let status = if delta_db <= tolerance_db {
            "pass"
        } else if delta_db <= tolerance_db + 0.5 {
            "warn"
        } else {
            "fail"
        };
        Self {
            source_a: source_a.to_string(),
            source_b: source_b.to_string(),
            sigma_a_dbsm,
            sigma_b_dbsm,
            delta_db,
            tolerance_db,
            status: status.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaSummary {
    pub count: usize,
    pub n_pass: usize,
    pub n_warn: usize,
    pub n_fail: usize,
    pub max_delta_db: f64,
    pub rms_delta_db: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSolverDelta {
    pub pairs: Vec<DeltaPair>,
    pub summary: DeltaSummary,
}

impl CrossSolverDelta {
    pub fn from_pairs(pairs: Vec<DeltaPair>) -> Self {
        let count = pairs.len();
        let n_pass = pairs.iter().filter(|p| p.status == "pass").count();
        let n_warn = pairs.iter().filter(|p| p.status == "warn").count();
        let n_fail = pairs.iter().filter(|p| p.status == "fail").count();
        let max_delta_db = pairs.iter().map(|p| p.delta_db).fold(0.0, f64::max);
        let rms_delta_db = if count == 0 {
            0.0
        } else {
            (pairs.iter().map(|p| p.delta_db.powi(2)).sum::<f64>() / count as f64).sqrt()
        };
        let summary = DeltaSummary {
            count,
            n_pass,
            n_warn,
            n_fail,
            max_delta_db,
            rms_delta_db,
        };
        Self { pairs, summary }
    }

    /// Hard-fail rule: rms_delta_db > 1.0 ⇒ fail regardless of pair pass rate.
    pub fn overall_status(&self) -> &'static str {
        if self.summary.rms_delta_db > 1.0 || self.summary.n_fail > 0 {
            "fail"
        } else if self.summary.n_warn > 0 {
            "warn"
        } else {
            "pass"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_status_thresholds() {
        let p = DeltaPair::new("a", "b", 10.0, 10.2, 0.5);
        assert_eq!(p.status, "pass");
        let p = DeltaPair::new("a", "b", 10.0, 10.6, 0.5);
        assert_eq!(p.status, "warn");
        let p = DeltaPair::new("a", "b", 10.0, 12.0, 0.5);
        assert_eq!(p.status, "fail");
    }

    #[test]
    fn rms_hard_fail() {
        let pairs = vec![DeltaPair::new("a", "b", 0.0, 1.5, 2.0)];
        let cs = CrossSolverDelta::from_pairs(pairs);
        // Single 1.5 dB delta → rms = 1.5 > 1.0 ⇒ fail.
        assert_eq!(cs.overall_status(), "fail");
    }
}
