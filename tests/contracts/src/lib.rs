#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use jsonschema::JSONSchema;
use serde_json::Value;

pub const KINDS: &[&str] = &[
    "object_card",
    "material_card",
    "mesh_manifest",
    "solver_card",
    "rcs_campaign",
    "echosig_manifest",
    "sensor_archetype",
    "scenario",
    "radar_episode",
    "detector_graph",
    "dataset_card",
    "validation_report",
];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("canonicalize workspace root")
}

fn load_json(path: &Path) -> Value {
    let raw =
        fs::read_to_string(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|err| panic!("parse {}: {err}", path.display()))
}

fn schema_docs() -> BTreeMap<String, Value> {
    let mut docs = BTreeMap::new();
    let schemas = workspace_root().join("schemas");
    for entry in
        fs::read_dir(&schemas).unwrap_or_else(|err| panic!("read_dir {}: {err}", schemas.display()))
    {
        let entry = entry.unwrap_or_else(|err| panic!("schema entry: {err}"));
        let path = entry.path();
        if !path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".schema.json"))
        {
            continue;
        }
        let value = load_json(&path);
        let id = value
            .get("$id")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("{} missing $id", path.display()))
            .to_string();
        docs.insert(id, value);
    }
    docs
}

fn validator_for(kind: &str) -> JSONSchema {
    let docs = schema_docs();
    let schema_path = workspace_root()
        .join("schemas")
        .join(format!("{kind}.schema.json"));
    let schema = load_json(&schema_path);
    let mut options = JSONSchema::options();
    for (id, doc) in &docs {
        options.with_document(id.clone(), doc.clone());
    }
    options
        .compile(&schema)
        .unwrap_or_else(|err| panic!("compile {}: {err}", schema_path.display()))
}

fn sample_paths() -> Vec<PathBuf> {
    let samples = workspace_root().join("tests/schemas");
    let mut out = Vec::new();
    for entry in
        fs::read_dir(&samples).unwrap_or_else(|err| panic!("read_dir {}: {err}", samples.display()))
    {
        let entry = entry.unwrap_or_else(|err| panic!("sample entry: {err}"));
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".sample.json"))
        {
            out.push(path);
        }
    }
    out.sort();
    out
}

fn canonical_json(kind: &str) -> String {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(cargo)
        .args([
            "run",
            "--quiet",
            "--locked",
            "-p",
            "echoforge-core",
            "--bin",
            "emit_canonical",
            "--",
            kind,
        ])
        .current_dir(workspace_root())
        .output()
        .unwrap_or_else(|err| panic!("emit_canonical {kind}: {err}"));
    if !output.status.success() {
        panic!(
            "emit_canonical {kind} failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout).expect("canonical utf8")
}

#[test]
fn schema_catalog_has_twelve_entries() {
    let root = load_json(&workspace_root().join("contracts/schema_catalog.json"));
    let catalog: Vec<Value> = root
        .get("schemas")
        .and_then(Value::as_array)
        .cloned()
        .expect("catalog.schemas array");
    assert_eq!(catalog.len(), 12);
}

#[test]
fn schemas_and_samples_validate() {
    for kind in KINDS {
        let validator = validator_for(kind);
        let payload = load_json(
            &workspace_root()
                .join("tests/schemas")
                .join(format!("{kind}.sample.json")),
        );
        assert!(
            validator.validate(&payload).is_ok(),
            "sample {kind} did not validate against schemas/{kind}.schema.json"
        );
    }

    for sample_path in sample_paths() {
        let sample = load_json(&sample_path);
        let kind = sample
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("{} missing kind", sample_path.display()));
        if !KINDS.contains(&kind) {
            continue;
        }
        let validator = validator_for(kind);
        assert!(
            validator.validate(&sample).is_ok(),
            "{} failed validation",
            sample_path.display()
        );
    }
}

#[test]
fn canonical_payloads_match_golden_hashes() {
    let golden_dir = workspace_root().join("target/contract-fixtures");
    fs::create_dir_all(&golden_dir).expect("golden dir");

    for kind in KINDS {
        let canonical = canonical_json(kind);
        let parsed: Value = serde_json::from_str(&canonical).expect("canonical json");
        let validator = validator_for(kind);
        assert!(
            validator.validate(&parsed).is_ok(),
            "canonical {kind} failed schema validation"
        );

        let hash = <sha2::Sha256 as sha2::Digest>::digest(canonical.as_bytes());
        let actual = format!("{hash:x}");
        let golden_path = golden_dir.join(format!("{kind}.rust.sha256"));
        if !golden_path.exists() {
            fs::write(&golden_path, format!("{actual}\n")).expect("seed golden");
            continue;
        }

        let expected = fs::read_to_string(&golden_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", golden_path.display()))
            .trim()
            .to_string();
        assert_eq!(expected, actual, "SHA mismatch for {kind}");
    }
}
