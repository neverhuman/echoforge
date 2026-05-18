//! End-to-end exercise of the `ef validate` engine via the library entry point.

use std::fs;
use std::path::Path;

use echoforge_validate::cli::{run as validate_run, ValidateArgs};
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

fn full_qa(dir: &Path) {
    write_qa(
        dir,
        "canonical_validation.json",
        r#"{"overall_status":"pass"}"#,
    );
    write_qa(dir, "polarization.json", r#"{"status":"pass"}"#);
    write_qa(dir, "determinism_report.json", r#"{"status":"pass"}"#);
    write_qa(dir, "units_frame_check.json", r#"{"status":"pass"}"#);
}

#[test]
fn v0_declared_always_passes() {
    let td = tempdir().unwrap();
    write_manifest(td.path(), "sphere");
    let rc = validate_run(ValidateArgs {
        bundle: td.path().to_path_buf(),
        primitive: Some("auto".to_string()),
        target_tier: "v0".to_string(),
        write_report: None,
        strict: false,
    })
    .unwrap();
    assert_eq!(rc, 0);
}

#[test]
fn v1_with_no_qa_fails_one() {
    let td = tempdir().unwrap();
    write_manifest(td.path(), "sphere");
    let rc = validate_run(ValidateArgs {
        bundle: td.path().to_path_buf(),
        primitive: Some("auto".to_string()),
        target_tier: "v1".to_string(),
        write_report: None,
        strict: false,
    })
    .unwrap();
    assert_eq!(rc, 1);
}

#[test]
fn v1_with_full_qa_passes() {
    let td = tempdir().unwrap();
    write_manifest(td.path(), "sphere");
    full_qa(td.path());
    let rc = validate_run(ValidateArgs {
        bundle: td.path().to_path_buf(),
        primitive: Some("auto".to_string()),
        target_tier: "v1".to_string(),
        write_report: None,
        strict: false,
    })
    .unwrap();
    assert_eq!(rc, 0);
}

#[test]
fn v2_requires_cross_solver_and_convergence() {
    let td = tempdir().unwrap();
    write_manifest(td.path(), "sphere");
    full_qa(td.path());
    let rc_before = validate_run(ValidateArgs {
        bundle: td.path().to_path_buf(),
        primitive: Some("auto".to_string()),
        target_tier: "v2".to_string(),
        write_report: None,
        strict: false,
    })
    .unwrap();
    assert_eq!(rc_before, 1);
    write_qa(
        td.path(),
        "cross_solver_delta.json",
        r#"{"overall_status":"pass"}"#,
    );
    write_qa(td.path(), "convergence_report.json", r#"{"pass":true}"#);
    let rc_after = validate_run(ValidateArgs {
        bundle: td.path().to_path_buf(),
        primitive: Some("auto".to_string()),
        target_tier: "v2".to_string(),
        write_report: None,
        strict: false,
    })
    .unwrap();
    assert_eq!(rc_after, 0);
}

#[test]
fn bad_args_returns_error() {
    let td = tempdir().unwrap();
    write_manifest(td.path(), "sphere");
    let err = validate_run(ValidateArgs {
        bundle: td.path().to_path_buf(),
        primitive: Some("auto".to_string()),
        target_tier: "v9".to_string(),
        write_report: None,
        strict: false,
    });
    assert!(err.is_err());
}

#[test]
fn fixture_v1_pass_bundle_returns_zero() {
    // Walk up from the workspace root to the committed v1_pass fixture.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/bundles/v1_pass");
    if !manifest.exists() {
        return;
    }
    let rc = validate_run(ValidateArgs {
        bundle: manifest,
        primitive: Some("auto".to_string()),
        target_tier: "v1".to_string(),
        write_report: None,
        strict: false,
    })
    .unwrap();
    assert_eq!(rc, 0);
}
