pub mod error;
pub mod artifact;
pub mod id;
pub mod models;
pub mod validation;

pub use artifact::{
    deterministic_artifact_id, validate_metadata, validate_required_fields, ArtifactId,
    LicenseRecord, ProvenanceRecord, ValidationIssue, ValidationReport, ValidationTier,
};
pub use error::CoreError;
pub use id::{canonical_json, deterministic_id, fingerprint_sha256};
pub use models::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_common(public_proxy_id: &str) -> (Provenance, LicenseInfo, ValidationInfo) {
        (
            Provenance {
                source_kind: "synthetic".to_string(),
                source_refs: vec!["tests/schemas".to_string()],
                generated_by: "echoforge-core-test".to_string(),
                generated_at: "2026-05-18T00:00:00Z".to_string(),
                fingerprint_sha256: String::new(),
            },
            LicenseInfo {
                spdx_id: "CC0-1.0".to_string(),
                notice: String::new(),
            },
            ValidationInfo {
                tier: "basic".to_string(),
                status: "pass".to_string(),
                uncertainty_score: 0.25,
                checks: vec![ValidationCheck {
                    name: format!("{}-check", public_proxy_id),
                    status: "pass".to_string(),
                    message: "ok".to_string(),
                }],
            },
        )
    }

    #[test]
    fn deterministic_id_is_stable() {
        let (provenance, license, validation) = sample_common("proxy-1");
        let card = ObjectCard {
            id: String::new(),
            kind: String::new(),
            schema_version: String::new(),
            public_proxy_id: "proxy-1".to_string(),
            provenance,
            license,
            validation,
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
        assert!(finalized.provenance.fingerprint_sha256.len() == 64);
        assert!(card.validate().is_ok());
    }
}
