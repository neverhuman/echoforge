use echoforge_cli::manifest::EchoSigManifest;
use serde_json::json;
use std::fs;

#[test]
fn inspects_echo_sig_manifest() {
    let temp = tempfile::tempdir().expect("tempdir");
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
    let report = manifest.inspect();

    assert_eq!(report.status.overall(), echoforge_cli::core::Health::Ok);
    assert_eq!(report.artifact.as_deref(), Some("ef:echo:sphere:abc123:1"));
    assert_eq!(report.validation_tier.as_deref(), Some("V0"));
}

#[test]
fn warns_on_sparse_manifest() {
    let manifest = EchoSigManifest::from_value("sparse.json", json!({}));
    let report = manifest.inspect();
    assert_eq!(report.status.overall(), echoforge_cli::core::Health::Warn);
}
