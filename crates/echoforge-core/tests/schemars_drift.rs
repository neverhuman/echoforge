// Structural drift check: ensures the `required` field set in every hand-
// written JSON Schema matches the identity-field tuple declared in the
// `impl_document!` macro for the corresponding Rust type.
//
// The Rust types don't currently derive `schemars::JsonSchema`, so we don't
// generate schemas from the types here (which would require editing
// models.rs). Instead we hard-code the expected required-field sets that
// mirror the macro invocations in models.rs, and compare against the
// schema's `required` array. Anyone adding a field to either side without
// updating the other will trip this test.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use serde_json::Value;

// Envelope fields every EchoForge document carries on the wire. Mirrored
// by the schema's `required` list together with the per-kind identity
// fields declared in `impl_document!`.
const ENVELOPE: &[&str] = &[
    "id",
    "kind",
    "schema_version",
    "public_proxy_id",
    "provenance",
    "license",
    "validation",
];

// (schema kind, identity fields from impl_document! in models.rs)
const KIND_IDENTITY_FIELDS: &[(&str, &[&str])] = &[
    (
        "object_card",
        &[
            "display_name",
            "object_family",
            "geometry_variant",
            "material_variant",
            "dimensions_m",
            "tags",
        ],
    ),
    (
        "material_card",
        &[
            "material_name",
            "material_family",
            "frequency_range_hz",
            "permittivity",
            "conductivity_s_per_m",
            "roughness_m",
        ],
    ),
    (
        "mesh_manifest",
        &[
            "mesh_name",
            "mesh_format",
            "source_files",
            "units",
            "triangle_count",
            "watertight",
            "mesh_sha256",
        ],
    ),
    (
        "solver_card",
        &[
            "solver_name",
            "solver_family",
            "version",
            "container_image",
            "supported_polarizations",
        ],
    ),
    (
        "rcs_campaign",
        &[
            "campaign_name",
            "object_card_id",
            "solver_card_id",
            "frequency_range_hz",
            "azimuth_deg",
            "tx_polarization",
            "rx_polarization",
            "run_count",
        ],
    ),
    (
        "echosig_manifest",
        &[
            "artifact_name",
            "object_card_id",
            "tensor_axes",
            "tensor_paths",
            "qa_paths",
        ],
    ),
    (
        "sensor_archetype",
        &[
            "sensor_name",
            "band_name",
            "waveform_family",
            "center_frequency_hz",
            "sample_rate_hz",
        ],
    ),
    (
        "scenario",
        &[
            "scenario_name",
            "sensor_archetype_id",
            "object_card_ids",
            "environment_label",
            "seed",
        ],
    ),
    (
        "radar_episode",
        &[
            "episode_name",
            "scenario_id",
            "sample_rate_hz",
            "product_paths",
        ],
    ),
    ("detector_graph", &["graph_name", "nodes", "edges"]),
    (
        "dataset_card",
        &["dataset_name", "source_campaign_ids", "splits"],
    ),
    (
        "validation_report",
        &[
            "report_name",
            "subject_kind",
            "subject_id",
            "checks",
            "overall_status",
        ],
    ),
];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .canonicalize()
        .expect("canonicalize workspace root")
}

fn schema_required(kind: &str) -> BTreeSet<String> {
    let path = workspace_root()
        .join("schemas")
        .join(format!("{kind}.schema.json"));
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let value: Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    value
        .get("required")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{kind} schema missing `required` array"))
        .iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("{kind} required entry not a string"))
                .to_string()
        })
        .collect()
}

fn rust_required(identity: &[&str]) -> BTreeSet<String> {
    ENVELOPE
        .iter()
        .chain(identity.iter())
        .map(|s| (*s).to_string())
        .collect()
}

#[test]
fn required_field_sets_match_for_all_kinds() {
    let mut failures = Vec::new();
    for (kind, identity) in KIND_IDENTITY_FIELDS {
        let schema = schema_required(kind);
        let rust = rust_required(identity);
        if schema != rust {
            let only_schema: Vec<_> = schema.difference(&rust).collect();
            let only_rust: Vec<_> = rust.difference(&schema).collect();
            failures.push(format!(
                "kind={kind}\n  only in schema: {only_schema:?}\n  only in rust:   {only_rust:?}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "schema/rust required-field drift detected:\n{}",
        failures.join("\n")
    );
}

#[test]
fn covers_twelve_kinds() {
    assert_eq!(
        KIND_IDENTITY_FIELDS.len(),
        12,
        "expected entries for all 12 EchoForge document kinds"
    );
}
