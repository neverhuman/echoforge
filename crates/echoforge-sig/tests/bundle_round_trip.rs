use ndarray::ArrayD;
use tempfile::tempdir;

use echoforge_core::{EchosigManifest, LicenseInfo, Provenance, ValidationCheck, ValidationInfo};
use echoforge_sig::{CardKind, Dtype, EchosigBundleReader, EchosigBundleWriter, TensorDecl};

fn sample_manifest() -> EchosigManifest {
    let manifest = EchosigManifest {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy-bundle-1".to_string(),
        provenance: Provenance {
            source_kind: "synthetic".to_string(),
            source_refs: vec!["tests/bundle_round_trip".to_string()],
            generated_by: "echoforge-sig-test".to_string(),
            generated_at: "2026-05-18T00:00:00Z".to_string(),
            fingerprint_sha256: String::new(),
        },
        license: LicenseInfo {
            spdx_id: "CC0-1.0".to_string(),
            notice: String::new(),
        },
        validation: ValidationInfo {
            tier: "basic".to_string(),
            status: "pass".to_string(),
            uncertainty_score: 0.25,
            checks: vec![ValidationCheck {
                name: "round-trip".to_string(),
                status: "pass".to_string(),
                message: "ok".to_string(),
            }],
            fidelity_class: None,
        },
        artifact_name: "round-trip-artifact".to_string(),
        object_card_id: "ef:object_card:proxy-bundle-1:abcdef0123456789:1".to_string(),
        tensor_axes: vec!["frequency_hz".to_string(), "azimuth_deg".to_string()],
        tensor_paths: vec!["tensors/sigma_m2.zarr".to_string()],
        qa_paths: vec!["qa/canonical_validation.json".to_string()],
    };
    manifest.finalize().expect("finalize manifest")
}

#[test]
fn round_trip_preserves_manifest_fingerprint_and_tensor_bytes() {
    let dir = tempdir().expect("tempdir");
    let manifest = sample_manifest();
    let original_fingerprint = manifest.provenance.fingerprint_sha256.clone();

    let mut writer =
        EchosigBundleWriter::create(dir.path(), manifest.clone()).expect("create writer");

    writer
        .declare_tensor(
            "tensors/sigma_m2.zarr",
            TensorDecl {
                dtype: Dtype::F32,
                shape: vec![4, 8],
                chunks: vec![4, 8],
            },
        )
        .expect("declare tensor");

    let shape: Vec<usize> = vec![4, 8];
    let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();
    let array: ArrayD<f32> = ArrayD::from_shape_vec(shape, data.clone()).expect("array");

    writer
        .write_tensor_f32("tensors/sigma_m2.zarr", &array)
        .expect("write tensor");

    writer
        .write_qa_json(
            "canonical_validation",
            &serde_json::json!({
                "overall_status": "pass",
                "checks": [],
            }),
        )
        .expect("write qa");

    writer
        .attach_card_yaml(CardKind::ObjectCard, "kind: object\nname: demo\n")
        .expect("attach object card");

    writer
        .write_state_sequence_csv(&[("t_s", vec![0.0, 0.1, 0.2]), ("x_m", vec![1.0, 2.0, 3.0])])
        .expect("write state seq");

    let bundle = writer.finalize().expect("finalize");
    assert_eq!(bundle.root, dir.path());

    let reader = EchosigBundleReader::open(dir.path()).expect("open reader");
    let round_trip = reader.manifest();
    assert_eq!(round_trip.id, manifest.id);
    assert_eq!(
        round_trip.provenance.fingerprint_sha256,
        original_fingerprint
    );
    assert_eq!(round_trip.tensor_axes, manifest.tensor_axes);
    assert_eq!(round_trip.tensor_paths, manifest.tensor_paths);

    let tensor = reader
        .read_tensor_f32("tensors/sigma_m2.zarr")
        .expect("read tensor");
    assert_eq!(tensor.shape(), &[4, 8]);
    let actual: Vec<f32> = tensor.iter().copied().collect();
    assert_eq!(actual, data);

    let qa: serde_json::Value = reader
        .read_qa_json("canonical_validation")
        .expect("read qa");
    assert_eq!(qa["overall_status"], "pass");

    let state = reader.read_state_sequence_csv().expect("read state seq");
    assert_eq!(state.len(), 2);
    assert_eq!(state[0].0, "t_s");
    assert_eq!(state[0].1, vec![0.0, 0.1, 0.2]);
    assert_eq!(state[1].0, "x_m");
    assert_eq!(state[1].1, vec![1.0, 2.0, 3.0]);

    let card_path = dir.path().join("object_card.yaml");
    assert!(card_path.exists());

    let listed = reader.list_tensor_files().expect("list tensors");
    assert_eq!(listed, vec!["tensors/sigma_m2.zarr".to_string()]);
}

#[test]
fn missing_tensor_fails_finalize() {
    let dir = tempdir().expect("tempdir");
    let manifest = sample_manifest();
    let writer = EchosigBundleWriter::create(dir.path(), manifest).expect("create writer");
    let err = writer.finalize().expect_err("expected missing tensor");
    matches!(err, echoforge_sig::SigError::MissingTensor(_));
}

#[test]
fn undeclared_tensor_path_rejected() {
    let dir = tempdir().expect("tempdir");
    let manifest = sample_manifest();
    let mut writer = EchosigBundleWriter::create(dir.path(), manifest).expect("create writer");
    let array: ArrayD<f32> = ArrayD::from_shape_vec(vec![4, 8], vec![0.0; 32]).expect("array");
    let err = writer
        .write_tensor_f32("tensors/not_in_manifest.zarr", &array)
        .expect_err("expected undeclared path");
    matches!(err, echoforge_sig::SigError::UndeclaredPath(_));
}
