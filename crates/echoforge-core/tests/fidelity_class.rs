// Fidelity-class field round-trip tests (fidelity-class-field packet).
//
// The `fidelity_class` field on `ValidationInfo` is an additive optional
// method-ceiling tier introduced per the local coordination archive
// resolution that separates method-ceiling from
// evidence-tier (evidence tier aliases / `ValidationInfo.tier`). This test file
// pins three invariants the packet ships:
//
//   1. A `ValidationInfo` payload without `fidelity_class` continues to
//      round-trip cleanly (backward compatibility — pre-existing
//      serialized payloads stay valid and the field stays absent on the
//      wire).
//   2. A `ValidationInfo` payload with `fidelity_class: "production_sbr"` round-trips
//      cleanly, both via the Rust `serde` round-trip and against the JSON
//      Schema at `schemas/common.schema.json#/$defs/validation`.
//   3. An invalid enum value (e.g. `"unknown_fidelity"`) fails JSON Schema validation.
//      The Rust struct itself accepts any `Option<String>` for forward
//      compatibility with future fidelity vocabularies; the schema is the
//      enforcement surface for the enum.

use std::fs;
use std::path::PathBuf;

use jsonschema::JSONSchema;
use serde_json::{json, Value};

use echoforge_core::{ValidationCheck, ValidationInfo};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .canonicalize()
        .expect("canonicalize workspace root")
}

fn load_common() -> Value {
    let path = workspace_root().join("schemas/common.schema.json");
    let raw = fs::read_to_string(&path).expect("read common.schema.json");
    serde_json::from_str(&raw).expect("parse common.schema.json")
}

/// Compile a JSON Schema validator anchored at the `validation` $def in
/// `schemas/common.schema.json`. We can't compile the sub-schema directly
/// because its `$ref: "#/$defs/validation_check"` resolves to the parent
/// document — so we build a tiny wrapper schema that `$ref`s into the
/// common doc's `$defs/validation`, register the common doc by its `$id`,
/// and let jsonschema walk the references.
///
/// This is the same envelope every EchoForge document embeds, so a payload
/// that passes here is byte-for-byte acceptable inside any of the 12
/// document kinds.
fn validation_envelope_validator() -> JSONSchema {
    let common = load_common();
    let wrapper = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": "https://echoforge.local/schemas/common.schema.json#/$defs/validation"
    });
    JSONSchema::options()
        .with_document(
            "https://echoforge.local/schemas/common.schema.json".to_string(),
            common,
        )
        .compile(&wrapper)
        .expect("compile validation envelope wrapper schema")
}

fn describe_errors<'a>(value: &'a Value, validator: &'a JSONSchema) -> String {
    match validator.validate(value) {
        Ok(()) => "<valid>".to_string(),
        Err(errors) => errors
            .map(|e| format!("    - {} @ {}", e, e.instance_path))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn check_ok() -> ValidationCheck {
    ValidationCheck {
        name: "fidelity-class-round-trip".to_string(),
        status: "pass".to_string(),
        message: "ok".to_string(),
    }
}

#[test]
fn validation_info_without_fidelity_class_round_trips() {
    let info = ValidationInfo {
        tier: "basic".to_string(),
        status: "pass".to_string(),
        uncertainty_score: 0.25,
        checks: vec![check_ok()],
        fidelity_class: None,
    };

    // serde round-trip preserves the absence of fidelity_class.
    let serialized = serde_json::to_value(&info).expect("serialize");
    assert!(
        serialized.get("fidelity_class").is_none(),
        "fidelity_class must be absent when None (skip_serializing_if), got: {serialized}"
    );

    let deserialized: ValidationInfo =
        serde_json::from_value(serialized.clone()).expect("deserialize");
    assert_eq!(deserialized, info);

    // Schema accepts the payload without fidelity_class (additive optional
    // field; existing fixtures must keep validating).
    let validator = validation_envelope_validator();
    let errors = describe_errors(&serialized, &validator);
    assert!(
        validator.validate(&serialized).is_ok(),
        "validation envelope without fidelity_class must remain schema-valid: {serialized}\nerrors:\n{errors}"
    );
}

#[test]
fn validation_info_with_fidelity_class_round_trips() {
    let info = ValidationInfo {
        tier: "basic".to_string(),
        status: "pass".to_string(),
        uncertainty_score: 0.1,
        checks: vec![check_ok()],
        fidelity_class: Some("production_sbr".to_string()),
    };

    let serialized = serde_json::to_value(&info).expect("serialize");
    assert_eq!(
        serialized.get("fidelity_class").and_then(Value::as_str),
        Some("production_sbr"),
        "fidelity_class must serialize as 'production_sbr'"
    );

    let deserialized: ValidationInfo =
        serde_json::from_value(serialized.clone()).expect("deserialize");
    assert_eq!(deserialized, info);
    assert_eq!(
        deserialized.fidelity_class.as_deref(),
        Some("production_sbr")
    );

    let validator = validation_envelope_validator();
    assert!(
        validator.validate(&serialized).is_ok(),
        "validation envelope with fidelity_class='production_sbr' must be schema-valid: {serialized}"
    );
}

#[test]
fn validation_info_accepts_every_documented_fidelity_class() {
    // Pin the full named enum: every documented method-ceiling tier must
    // be schema-accepted. Anyone shrinking the enum without bumping
    // documentation will trip this test.
    let validator = validation_envelope_validator();
    for class in [
        "analytic_prior",
        "primitive_decomposed",
        "production_sbr",
        "cross_solver_bridged",
        "measured_anchored",
        "lawful_measured_validated",
    ] {
        let payload = json!({
            "tier": "basic",
            "status": "pass",
            "uncertainty_score": 0.5,
            "checks": [
                {"name": "fidelity-class-enum", "status": "pass", "message": "ok"}
            ],
            "fidelity_class": class,
        });
        assert!(
            validator.validate(&payload).is_ok(),
            "fidelity_class={class} must validate against the schema; payload={payload}"
        );
    }
}

#[test]
fn validation_info_rejects_invalid_fidelity_class() {
    // The Rust struct accepts any String (forward compatibility); the
    // schema is the enforcement surface for the named enum.
    let validator = validation_envelope_validator();
    for bad in [
        "unknown_fidelity",
        "bad_numeric_label",
        "fidelity_lowercase",
        "production_sbr ",
        "",
        "cross_checked",
        "fidelity_unknown",
    ] {
        let payload = json!({
            "tier": "basic",
            "status": "pass",
            "uncertainty_score": 0.5,
            "checks": [
                {"name": "fidelity-class-enum", "status": "pass", "message": "ok"}
            ],
            "fidelity_class": bad,
        });
        let result = validator.validate(&payload);
        assert!(
            result.is_err(),
            "fidelity_class={bad:?} must be rejected by the schema; payload={payload}"
        );
    }
}

#[test]
fn fidelity_class_is_independent_from_tier() {
    // The cross-tip resolution treats the two axes as orthogonal: a card
    // may carry low evidence (unvalidated) but a high method-ceiling
    // claim (measured_anchored) — or vice versa. This test pins that
    // the schema accepts every cross-axis combination so producers can
    // surface the two axes independently.
    let validator = validation_envelope_validator();
    let tiers = [
        "unvalidated",
        "basic",
        "cross_checked",
        "benchmarked",
        "measured",
    ];
    let classes = [
        "analytic_prior",
        "primitive_decomposed",
        "production_sbr",
        "cross_solver_bridged",
        "measured_anchored",
        "lawful_measured_validated",
    ];
    for tier in tiers {
        for class in classes {
            let payload = json!({
                "tier": tier,
                "status": "pass",
                "uncertainty_score": 0.5,
                "checks": [
                    {"name": "cross-axis", "status": "pass", "message": "ok"}
                ],
                "fidelity_class": class,
            });
            assert!(
                validator.validate(&payload).is_ok(),
                "tier={tier} + fidelity_class={class} must validate; payload={payload}"
            );
        }
    }
}
