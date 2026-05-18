use echoforge_cli::schema::validate_inputs;
use std::fs;

#[test]
fn validates_schema_files() {
    let temp = tempfile::tempdir().expect("tempdir");
    let schema_path = temp.path().join("example.schema.json");
    fs::write(
        &schema_path,
        r#"{"$schema":"https://json-schema.org/draft/2020-12/schema","title":"Example","type":"object"}"#,
    )
    .expect("write schema");

    let report = validate_inputs(&[schema_path]);
    assert_eq!(report.summary.overall(), echoforge_cli::core::Health::Ok);
    assert_eq!(report.files.len(), 1);
}

#[test]
fn warns_on_missing_schema_inputs() {
    let empty: Vec<std::path::PathBuf> = Vec::new();
    let report = validate_inputs(&empty);
    assert_eq!(report.summary.overall(), echoforge_cli::core::Health::Warn);
}
