use echoforge_core::{
    deterministic_id, LicenseInfo, ObjectCard, Provenance, ValidationCheck, ValidationInfo, Vector3,
};
use serde_json::json;

#[test]
fn deterministic_id_uses_expected_shape() {
    let payload = json!({
        "display_name": "Proxy 1",
        "object_family": "airframe",
        "geometry_variant": "baseline",
        "material_variant": "default",
        "dimensions_m": {"x": 1.0, "y": 2.0, "z": 3.0},
        "tags": ["proxy"]
    });

    let id = deterministic_id("object_card", "proxy-1", &payload).expect("deterministic id");
    assert!(id.starts_with("ef:object_card:proxy-1:"));
}

#[test]
fn object_card_finalizes_with_schema_version_and_fingerprint() {
    let card = ObjectCard {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy-1".to_string(),
        provenance: Provenance {
            source_kind: "synthetic".to_string(),
            source_refs: vec!["tests/schemas".to_string()],
            generated_by: "echoforge-core-test".to_string(),
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
                name: "proxy-1-check".to_string(),
                status: "pass".to_string(),
                message: "ok".to_string(),
            }],
        },
        display_name: "Proxy 1".to_string(),
        object_family: "airframe".to_string(),
        geometry_variant: "baseline".to_string(),
        material_variant: "default".to_string(),
        dimensions_m: Vector3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        },
        tags: vec!["proxy".to_string()],
    };

    let finalized = card.clone().finalize().expect("finalize");
    assert!(finalized.id.starts_with("ef:object_card:proxy-1:"));
    assert_eq!(finalized.kind, "object_card");
    assert_eq!(finalized.schema_version, "1.0.0");
    assert_eq!(finalized.provenance.source_kind, "synthetic");
    assert_eq!(finalized.validation.status, "pass");
    assert!(card.validate().is_ok());
}
