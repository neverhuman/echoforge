// Schema round-trip test: builds a canonical, finalized minimal instance of
// every EchoForge document kind, serializes it via serde_json, and validates
// the result against the hand-written JSON Schema at
// `schemas/<kind>.schema.json`. Catches drift between the Rust serde wire
// format and the published schema contract.

use std::fs;
use std::path::PathBuf;

use jsonschema::JSONSchema;
use serde_json::Value;

#[path = "common/samples.rs"]
mod samples;

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR -> crates/echoforge-core; ../../ is the workspace root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .canonicalize()
        .expect("canonicalize workspace root")
}

fn load_schema(kind: &str) -> Value {
    let path = workspace_root()
        .join("schemas")
        .join(format!("{kind}.schema.json"));
    let raw =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("read schema {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse schema {}: {e}", path.display()))
}

fn load_common() -> Value {
    let path = workspace_root().join("schemas/common.schema.json");
    let raw = fs::read_to_string(&path).expect("read common.schema.json");
    serde_json::from_str(&raw).expect("parse common.schema.json")
}

fn compile(kind: &str) -> JSONSchema {
    let schema = load_schema(kind);
    let common = load_common();
    // Register both `$id` URLs so `$ref` resolution works without network.
    JSONSchema::options()
        .with_document(
            "https://echoforge.local/schemas/common.schema.json".to_string(),
            common,
        )
        .with_document(
            format!("https://echoforge.local/schemas/{kind}.schema.json"),
            schema.clone(),
        )
        .compile(&schema)
        .unwrap_or_else(|e| panic!("compile {kind} schema: {e}"))
}

fn assert_round_trip(kind: &str) {
    let value = samples::canonical_value(kind).expect("sample value");
    let compiled = compile(kind);
    let result = compiled.validate(&value);
    if let Err(errors) = result {
        let messages: Vec<String> = errors
            .map(|e| format!("  - {} @ {}", e, e.instance_path))
            .collect();
        panic!(
            "schema round-trip FAILED for kind={kind}:\n{}\npayload was:\n{}",
            messages.join("\n"),
            serde_json::to_string_pretty(&value).unwrap_or_default(),
        );
    }
}

#[test]
fn object_card_round_trip() {
    assert_round_trip("object_card");
}

#[test]
fn material_card_round_trip() {
    assert_round_trip("material_card");
}

#[test]
fn mesh_manifest_round_trip() {
    assert_round_trip("mesh_manifest");
}

#[test]
fn solver_card_round_trip() {
    assert_round_trip("solver_card");
}

#[test]
fn rcs_campaign_round_trip() {
    assert_round_trip("rcs_campaign");
}

#[test]
fn echosig_manifest_round_trip() {
    assert_round_trip("echosig_manifest");
}

#[test]
fn sensor_archetype_round_trip() {
    assert_round_trip("sensor_archetype");
}

#[test]
fn scenario_round_trip() {
    assert_round_trip("scenario");
}

#[test]
fn radar_episode_round_trip() {
    assert_round_trip("radar_episode");
}

#[test]
fn detector_graph_round_trip() {
    assert_round_trip("detector_graph");
}

#[test]
fn dataset_card_round_trip() {
    assert_round_trip("dataset_card");
}

#[test]
fn validation_report_round_trip() {
    assert_round_trip("validation_report");
}

#[test]
fn all_twelve_kinds_covered() {
    assert_eq!(
        samples::KINDS.len(),
        12,
        "expected exactly 12 EchoForge document kinds"
    );
}
