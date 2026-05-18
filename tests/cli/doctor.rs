use clap::Parser;
use echoforge_cli::manifest::EchoSigManifest;
use echoforge_cli::schema::validate_inputs;
use serde_json::json;
use std::fs;

#[test]
fn renders_doctor_style_summary() {
    let temp = tempfile::tempdir().expect("tempdir");
    let schema_path = temp.path().join("example.schema.json");
    fs::write(
        &schema_path,
        r#"{"$schema":"https://json-schema.org/draft/2020-12/schema","title":"Example","type":"object"}"#,
    )
    .expect("write schema");

    let manifest_path = temp.path().join("manifest.json");
    fs::write(
        &manifest_path,
        json!({
            "artifact": "ef:echo:sphere:abc123:1",
            "validation_tier": "V0",
            "provenance": {"source": "synthetic"},
            "license": {"spdx": "CC-BY-4.0"}
        })
        .to_string(),
    )
    .expect("write manifest");

    let manifest = EchoSigManifest::load(&manifest_path);
    let manifest_report = manifest.inspect();
    let schema_report = validate_inputs(&[schema_path]);

    let doctor = format!("{}\n{}\n", manifest_report.render(), schema_report.render());

    assert!(doctor.contains("EchoSig manifest"));
    assert!(doctor.contains("Schema validation"));
}

#[test]
fn doctor_args_are_parseable() {
    let cli = echoforge_cli::cli::Cli::try_parse_from([
        "ef",
        "doctor",
        "--schema",
        "example.schema.json",
        "--manifest",
        "manifest.json",
    ]);

    assert!(cli.is_ok());
}
