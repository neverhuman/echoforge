use proptest::prelude::*;

proptest! {
    #[test]
    fn core_error_validation_display_mentions_reason(reason in "[a-zA-Z0-9 ]{1,64}") {
        use echoforge_core::CoreError;
        let err = CoreError::Validation(reason.clone());
        prop_assert!(
            err.to_string().contains(&reason),
            "display output should mention the validation reason"
        );
    }

    #[test]
    fn ensure_slug_accepts_valid_slugs(s in "[a-z][a-z0-9]{0,15}") {
        use echoforge_core::validation::ensure_slug;
        prop_assert!(
            ensure_slug(&s, "field").is_ok(),
            "slug '{s}' should be accepted"
        );
    }

    #[test]
    fn ensure_probability_accepts_range(value in 0.0f64..=1.0f64) {
        use echoforge_core::validation::ensure_probability;
        prop_assert!(
            ensure_probability(value, "p").is_ok(),
            "probability {value} should be accepted"
        );
    }

    #[test]
    fn ensure_probability_rejects_out_of_range(value in 1.000001f64..1e10f64) {
        use echoforge_core::validation::ensure_probability;
        prop_assert!(
            ensure_probability(value, "p").is_err(),
            "probability {value} should be rejected"
        );
    }

    #[test]
    fn artifact_id_format_invariant(
        kind in "[a-z][a-z0-9]{1,8}",
        proxy in "[a-z][a-z0-9]{1,8}",
        hash in "[a-f0-9]{8,16}",
        version in "[a-z][a-z0-9]{0,4}",
    ) {
        use echoforge_core::ArtifactId;
        let id = ArtifactId::new(&kind, &proxy, &hash, &version);
        prop_assert!(id.is_ok(), "ArtifactId::new should succeed for valid inputs");
        let id = id.unwrap();
        let s = id.as_str();
        prop_assert!(s.starts_with("ef:"), "id should start with ef: got {s}");
        prop_assert!(s.contains(&kind), "id should contain kind");
        prop_assert!(s.contains(&proxy), "id should contain proxy");
    }
}

#[test]
fn core_error_json_display_is_nonempty() {
    let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
    let err = echoforge_core::CoreError::Json(json_err);
    assert!(!err.to_string().is_empty());
}
