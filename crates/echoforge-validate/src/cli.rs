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

struct QaBundle {
    canonical: Option<CanonicalValidation>,
    polarization: Option<PolarizationDoc>,
    determinism: Option<DeterminismDoc>,
    units_frame: Option<UnitsFrameDoc>,
    cross: Option<CrossSolverDoc>,
    conv: Option<ConvergenceDoc>,
}

fn read_qa_bundle(qa: &Path) -> Result<QaBundle, ValidateError> {
    Ok(QaBundle {
        canonical: read_optional_json(&qa.join("canonical_validation.json"))?,
        polarization: read_optional_json(&qa.join("polarization.json"))?,
        determinism: read_optional_json(&qa.join("determinism_report.json"))?,
        units_frame: read_optional_json(&qa.join("units_frame_check.json"))?,
        cross: read_optional_json(&qa.join("cross_solver_delta.json"))?,
        conv: read_optional_json(&qa.join("convergence_report.json"))?,
    })
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
    let target = match TierAchieved::parse(&args.target_tier) {
        Some(v) => v,
        None => {
            return Err(ValidateError::BadArgs(format!(
                "unknown tier: {}",
                args.target_tier
            )))
        }
    };

    let manifest_path = args.bundle.join("manifest.json");
    if !manifest_path.exists() {
        return Err(ValidateError::Schema(format!(
            "manifest.json not found in {}",
            args.bundle.display()
        )));
    }
    let manifest: ManifestSlim = read_required_json(&manifest_path)?;

    let primitive_arg = match args.primitive.clone() {
        Some(v) => v,
        None => "auto".to_string(),
    };
    let primitive = if primitive_arg == "auto" {
        let inferred = manifest
            .object_card
            .as_ref()
            .and_then(|c| c.kind.clone())
            .or(manifest.object_card_kind.clone());
        if let Some(v) = inferred {
            v
        } else {
            "unknown".to_string()
        }
    } else {
        primitive_arg
    };

    let qa = args.bundle.join("qa");
    let qa_data = read_qa_bundle(&qa)?;

    let mut checks = ValidateChecks::default();
    let mut notes = Vec::new();
    checks.canonical_validation_present = qa_data.canonical.is_some();
    checks.canonical_overall_pass = qa_data
        .canonical
        .as_ref()
        .map(|c| c.overall_status == "pass")
        .unwrap_or(false);
    checks.polarization_complete = qa_data
        .polarization
        .as_ref()
        .map(|p| p.status == "pass")
        .unwrap_or(false);
    checks.determinism_pass = qa_data
        .determinism
        .as_ref()
        .map(|d| d.status == "pass")
        .unwrap_or(false);
    checks.units_frame_pass = qa_data
        .units_frame
        .as_ref()
        .map(|u| u.status == "pass")
        .unwrap_or(false);
    checks.cross_solver_present = qa_data.cross.is_some();
    checks.cross_solver_pass = qa_data
        .cross
        .as_ref()
        .map(|c| c.overall_status == "pass")
        .unwrap_or(false);
    checks.convergence_present = qa_data.conv.is_some();
    checks.convergence_pass = qa_data.conv.as_ref().map(|c| c.pass).unwrap_or(false);

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

    let report_path = match args.write_report.clone() {
        Some(p) => p,
        None => qa.join("validation_report.json"),
    };
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(&report)?;
    fs::write(&report_path, body)?;

    let exit_code = if pass { 0 } else { 1 };
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

    fn make_args(bundle: std::path::PathBuf, tier: &str) -> ValidateArgs {
        ValidateArgs {
            bundle,
            primitive: Some("auto".to_string()),
            target_tier: tier.to_string(),
            write_report: None,
            strict: false,
        }
    }

    #[test]
    fn v0_always_passes() {
        let td = tempdir().unwrap();
        write_manifest(td.path(), "sphere");
        assert_eq!(run(make_args(td.path().to_path_buf(), "v0")).unwrap(), 0);
    }

    #[test]
    fn v1_fails_when_qa_missing() {
        let td = tempdir().unwrap();
        write_manifest(td.path(), "sphere");
        assert_eq!(run(make_args(td.path().to_path_buf(), "v1")).unwrap(), 1);
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
        assert_eq!(run(make_args(td.path().to_path_buf(), "v1")).unwrap(), 0);
    }

    #[test]
    fn bad_tier_returns_bad_args() {
        let td = tempdir().unwrap();
        write_manifest(td.path(), "sphere");
        let err = run(make_args(td.path().to_path_buf(), "v9")).err().unwrap();
        assert!(matches!(err, ValidateError::BadArgs(_)));
    }
}
