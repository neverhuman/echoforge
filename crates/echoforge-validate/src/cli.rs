//! `ef validate` subcommand implementation.
//!
//! Bundle layout consumed:
//!   <bundle>/manifest.json                     (required)
//!   <bundle>/qa/canonical_validation.json      (V1 requirement)
//!   <bundle>/qa/polarization.json              (V1 requirement)
//!   <bundle>/qa/determinism_report.json        (V1 requirement)
//!   <bundle>/qa/units_frame_check.json         (V1 requirement)
//!   <bundle>/qa/cross_solver_delta.json        (V2 requirement)
//!   <bundle>/qa/convergence_report.json        (V2 requirement)
//!
//! Exit codes:
//!   0 = target tier OK
//!   1 = required check failed (or warn under --strict)
//!   2 = schema / IO error
//!   3 = bad args

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub use crate::error::ValidateError;
use crate::report::{promotion_gate, ErrorBudget, TierAchieved, ValidateChecks, ValidateReport};

#[derive(Debug, Clone)]
pub struct ValidateArgs {
    pub bundle: PathBuf,
    pub primitive: Option<String>,
    pub target_tier: String,
    pub write_report: Option<PathBuf>,
    pub strict: bool,
}

impl Default for ValidateArgs {
    fn default() -> Self {
        Self {
            bundle: PathBuf::new(),
            primitive: Some("auto".to_string()),
            target_tier: "v1".to_string(),
            write_report: None,
            strict: false,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ManifestSlim {
    #[serde(default)]
    object_card: Option<ObjectCardSlim>,
    #[serde(default)]
    object_card_kind: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ObjectCardSlim {
    #[serde(default)]
    kind: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CanonicalValidation {
    #[serde(default)]
    overall_status: String,
}

#[derive(Debug, Deserialize)]
struct PolarizationDoc {
    #[serde(default)]
    status: String,
}

#[derive(Debug, Deserialize)]
struct DeterminismDoc {
    #[serde(default)]
    status: String,
}

#[derive(Debug, Deserialize)]
struct UnitsFrameDoc {
    #[serde(default)]
    status: String,
}

#[derive(Debug, Deserialize)]
struct CrossSolverDoc {
    #[serde(default)]
    overall_status: String,
}

#[derive(Debug, Deserialize)]
struct ConvergenceDoc {
    #[serde(default)]
    pass: bool,
}

fn read_optional_json<T: serde::de::DeserializeOwned>(
    path: &Path,
) -> Result<Option<T>, ValidateError> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path)?;
    let v: T = serde_json::from_str(&raw)?;
    Ok(Some(v))
}

fn read_required_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, ValidateError> {
    let raw = fs::read_to_string(path)?;
    let v: T = serde_json::from_str(&raw)
        .map_err(|e| ValidateError::Schema(format!("{}: {}", path.display(), e)))?;
    Ok(v)
}

/// Run the validate gate. Returns an exit code on success path.
pub fn run(args: ValidateArgs) -> Result<i32, ValidateError> {
    if args.bundle.as_os_str().is_empty() {
        return Err(ValidateError::BadArgs("missing bundle path".to_string()));
    }
    if !args.bundle.exists() {
        return Err(ValidateError::BadArgs(format!(
            "bundle does not exist: {}",
            args.bundle.display()
        )));
    }
    let target = TierAchieved::parse(&args.target_tier)
        .ok_or_else(|| ValidateError::BadArgs(format!("unknown tier: {}", args.target_tier)))?;

    let manifest_path = args.bundle.join("manifest.json");
    if !manifest_path.exists() {
        return Err(ValidateError::Schema(format!(
            "manifest.json not found in {}",
            args.bundle.display()
        )));
    }
    let manifest: ManifestSlim = read_required_json(&manifest_path)?;

    let primitive_arg = args.primitive.clone().unwrap_or_else(|| "auto".to_string());
    let primitive = if primitive_arg == "auto" {
        manifest
            .object_card
            .as_ref()
            .and_then(|c| c.kind.clone())
            .or(manifest.object_card_kind.clone())
            .unwrap_or_else(|| "unknown".to_string())
    } else {
        primitive_arg
    };

    let qa = args.bundle.join("qa");
    let canonical: Option<CanonicalValidation> =
        read_optional_json(&qa.join("canonical_validation.json"))?;
    let polarization: Option<PolarizationDoc> = read_optional_json(&qa.join("polarization.json"))?;
    let determinism: Option<DeterminismDoc> =
        read_optional_json(&qa.join("determinism_report.json"))?;
    let units_frame: Option<UnitsFrameDoc> =
        read_optional_json(&qa.join("units_frame_check.json"))?;
    let cross: Option<CrossSolverDoc> = read_optional_json(&qa.join("cross_solver_delta.json"))?;
    let conv: Option<ConvergenceDoc> = read_optional_json(&qa.join("convergence_report.json"))?;

    let mut checks = ValidateChecks::default();
    let mut notes = Vec::new();
    checks.canonical_validation_present = canonical.is_some();
    checks.canonical_overall_pass = canonical
        .as_ref()
        .map(|c| c.overall_status == "pass")
        .unwrap_or(false);
    checks.polarization_complete = polarization
        .as_ref()
        .map(|p| p.status == "pass")
        .unwrap_or(false);
    checks.determinism_pass = determinism
        .as_ref()
        .map(|d| d.status == "pass")
        .unwrap_or(false);
    checks.units_frame_pass = units_frame
        .as_ref()
        .map(|u| u.status == "pass")
        .unwrap_or(false);
    checks.cross_solver_present = cross.is_some();
    checks.cross_solver_pass = cross
        .as_ref()
        .map(|c| c.overall_status == "pass")
        .unwrap_or(false);
    checks.convergence_present = conv.is_some();
    checks.convergence_pass = conv.as_ref().map(|c| c.pass).unwrap_or(false);

    if !checks.canonical_validation_present {
        notes.push("qa/canonical_validation.json missing".to_string());
    }
    if !checks.polarization_complete {
        notes.push("polarization completeness not pass".to_string());
    }
    if !checks.determinism_pass {
        notes.push("determinism replay not pass".to_string());
    }
    if !checks.units_frame_pass {
        notes.push("units/frame check not pass".to_string());
    }
    if target == TierAchieved::V2 {
        if !checks.cross_solver_present {
            notes.push("qa/cross_solver_delta.json missing".to_string());
        }
        if !checks.convergence_present {
            notes.push("qa/convergence_report.json missing".to_string());
        }
    }

    let pass = promotion_gate(target, &checks);
    let overall_status = if pass {
        "pass".to_string()
    } else if args.strict || notes.iter().any(|n| n.contains("missing")) {
        "fail".to_string()
    } else {
        "warn".to_string()
    };

    let n_pass = [
        checks.canonical_overall_pass,
        checks.polarization_complete,
        checks.determinism_pass,
        checks.units_frame_pass,
        checks.cross_solver_pass,
        checks.convergence_pass,
    ]
    .iter()
    .filter(|b| **b)
    .count();
    let n_fail = if pass { 0 } else { notes.len() };

    let report = ValidateReport {
        tier: target,
        overall_status: overall_status.clone(),
        primitives_checked: vec![primitive.clone()],
        n_pass,
        n_warn: 0,
        n_fail,
        error_budget: ErrorBudget::default(),
        checks,
        notes,
    };

    let report_path = args
        .write_report
        .clone()
        .unwrap_or_else(|| qa.join("validation_report.json"));
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(&report)?;
    fs::write(&report_path, body)?;

    let exit_code = if pass {
        0
    } else if overall_status == "fail" || args.strict {
        1
    } else {
        1
    };
    Ok(exit_code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_manifest(dir: &Path, kind: &str) {
        let body = format!(r#"{{"object_card_kind":"{kind}"}}"#);
        fs::write(dir.join("manifest.json"), body).unwrap();
    }

    fn write_qa(dir: &Path, name: &str, body: &str) {
        let qa = dir.join("qa");
        fs::create_dir_all(&qa).unwrap();
        fs::write(qa.join(name), body).unwrap();
    }

    #[test]
    fn v0_always_passes() {
        let td = tempdir().unwrap();
        write_manifest(td.path(), "sphere");
        let args = ValidateArgs {
            bundle: td.path().to_path_buf(),
            primitive: Some("auto".to_string()),
            target_tier: "v0".to_string(),
            write_report: None,
            strict: false,
        };
        let rc = run(args).unwrap();
        assert_eq!(rc, 0);
    }

    #[test]
    fn v1_fails_when_qa_missing() {
        let td = tempdir().unwrap();
        write_manifest(td.path(), "sphere");
        let args = ValidateArgs {
            bundle: td.path().to_path_buf(),
            primitive: Some("auto".to_string()),
            target_tier: "v1".to_string(),
            write_report: None,
            strict: false,
        };
        let rc = run(args).unwrap();
        assert_eq!(rc, 1);
    }

    #[test]
    fn v1_passes_with_full_qa() {
        let td = tempdir().unwrap();
        write_manifest(td.path(), "sphere");
        write_qa(
            td.path(),
            "canonical_validation.json",
            r#"{"overall_status":"pass"}"#,
        );
        write_qa(td.path(), "polarization.json", r#"{"status":"pass"}"#);
        write_qa(td.path(), "determinism_report.json", r#"{"status":"pass"}"#);
        write_qa(td.path(), "units_frame_check.json", r#"{"status":"pass"}"#);
        let args = ValidateArgs {
            bundle: td.path().to_path_buf(),
            primitive: Some("auto".to_string()),
            target_tier: "v1".to_string(),
            write_report: None,
            strict: false,
        };
        let rc = run(args).unwrap();
        assert_eq!(rc, 0);
    }

    #[test]
    fn bad_tier_returns_bad_args() {
        let td = tempdir().unwrap();
        write_manifest(td.path(), "sphere");
        let args = ValidateArgs {
            bundle: td.path().to_path_buf(),
            primitive: Some("auto".to_string()),
            target_tier: "v9".to_string(),
            write_report: None,
            strict: false,
        };
        let err = run(args).err().unwrap();
        assert!(matches!(err, ValidateError::BadArgs(_)));
    }
}
