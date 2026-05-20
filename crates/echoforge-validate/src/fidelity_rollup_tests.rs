use super::*;
use echoforge_core::ValidationInfo;

fn envelope(tier: &str, fidelity: Option<&str>) -> ValidationInfo {
    ValidationInfo {
        tier: tier.to_string(),
        status: "pass".to_string(),
        uncertainty_score: 0.0,
        checks: Vec::new(),
        fidelity_class: fidelity.map(|s| s.to_string()),
    }
}

#[test]
fn parse_fidelity_class_accepts_f0_through_f5() {
    assert_eq!(parse_fidelity_class("F0"), Some(0));
    assert_eq!(parse_fidelity_class("F1"), Some(1));
    assert_eq!(parse_fidelity_class("F2"), Some(2));
    assert_eq!(parse_fidelity_class("F3"), Some(3));
    assert_eq!(parse_fidelity_class("F4"), Some(4));
    assert_eq!(parse_fidelity_class("F5"), Some(5));
}

#[test]
fn parse_fidelity_class_rejects_unknown_labels() {
    assert_eq!(parse_fidelity_class("F6"), None);
    assert_eq!(parse_fidelity_class("F"), None);
    assert_eq!(parse_fidelity_class(""), None);
    assert_eq!(parse_fidelity_class("X"), None);
    assert_eq!(parse_fidelity_class("f2"), None);
    assert_eq!(parse_fidelity_class("F2 "), None);
    assert_eq!(parse_fidelity_class("V2"), None);
    assert_eq!(parse_fidelity_class("F10"), None);
}

#[test]
fn rollup_empty_input_returns_zeros() {
    let rollup = rollup_validation_envelopes(&[]);
    assert_eq!(rollup.artifact_count, 0);
    assert_eq!(rollup.with_fidelity_class, 0);
    assert_eq!(rollup.without_fidelity_class, 0);
    assert!(rollup.by_fidelity.is_empty());
    assert!(rollup.by_tier_x_fidelity.is_empty());
}

#[test]
fn rollup_counts_with_and_without_fidelity() {
    let envs = vec![
        envelope("V0", None),
        envelope("V1", Some("F1")),
        envelope("V1", Some("F2")),
        envelope("V2", Some("F2")),
        envelope("V0", None),
    ];
    let refs: Vec<&ValidationInfo> = envs.iter().collect();
    let rollup = rollup_validation_envelopes(&refs);

    assert_eq!(rollup.artifact_count, 5);
    assert_eq!(rollup.with_fidelity_class, 3);
    assert_eq!(rollup.without_fidelity_class, 2);
    assert_eq!(rollup.by_fidelity.get("F1"), Some(&1));
    assert_eq!(rollup.by_fidelity.get("F2"), Some(&2));
    assert!(rollup.by_fidelity.get("F0").is_none());
}

#[test]
fn rollup_by_tier_x_fidelity_counts_pairs() {
    let envs = vec![
        envelope("V1", Some("F2")),
        envelope("V1", Some("F2")),
        envelope("V2", Some("F2")),
        envelope("V2", Some("F4")),
        envelope("V0", None),
    ];
    let refs: Vec<&ValidationInfo> = envs.iter().collect();
    let rollup = rollup_validation_envelopes(&refs);

    assert_eq!(
        rollup
            .by_tier_x_fidelity
            .get(&("V1".to_string(), "F2".to_string())),
        Some(&2)
    );
    assert_eq!(
        rollup
            .by_tier_x_fidelity
            .get(&("V2".to_string(), "F2".to_string())),
        Some(&1)
    );
    assert_eq!(
        rollup
            .by_tier_x_fidelity
            .get(&("V2".to_string(), "F4".to_string())),
        Some(&1)
    );
    assert_eq!(rollup.by_tier_x_fidelity.len(), 3);
}

#[test]
fn rollup_preserves_unknown_fidelity_labels() {
    let envs = vec![envelope("V1", Some("F9")), envelope("V1", Some("F9"))];
    let refs: Vec<&ValidationInfo> = envs.iter().collect();
    let rollup = rollup_validation_envelopes(&refs);
    assert_eq!(rollup.by_fidelity.get("F9"), Some(&2));
}

#[test]
fn assert_floor_passes_when_envelope_at_or_above_floor() {
    let floor = FidelityFloor::new("F2");
    assert!(assert_fidelity_floor(&envelope("V1", Some("F2")), &floor).is_ok());
    assert!(assert_fidelity_floor(&envelope("V1", Some("F3")), &floor).is_ok());
    assert!(assert_fidelity_floor(&envelope("V2", Some("F5")), &floor).is_ok());
}

#[test]
fn assert_floor_fails_when_envelope_below_floor() {
    let floor = FidelityFloor::new("F2");
    let err = assert_fidelity_floor(&envelope("V1", Some("F1")), &floor).unwrap_err();
    match err {
        FidelityFloorError::Below {
            envelope,
            floor,
            got,
            required,
        } => {
            assert_eq!(envelope, "F1");
            assert_eq!(floor, "F2");
            assert_eq!(got, 1);
            assert_eq!(required, 2);
        }
        other => panic!("expected Below, got {:?}", other),
    }
}

#[test]
fn assert_floor_fails_when_envelope_missing_field() {
    let floor = FidelityFloor::new("F1");
    let err = assert_fidelity_floor(&envelope("V0", None), &floor).unwrap_err();
    match err {
        FidelityFloorError::Missing { floor } => assert_eq!(floor, "F1"),
        other => panic!("expected Missing, got {:?}", other),
    }
}

#[test]
fn assert_floor_fails_when_envelope_label_unparseable() {
    let floor = FidelityFloor::new("F1");
    let err = assert_fidelity_floor(&envelope("V1", Some("F9")), &floor).unwrap_err();
    match err {
        FidelityFloorError::UnparseableEnvelope { value } => assert_eq!(value, "F9"),
        other => panic!("expected UnparseableEnvelope, got {:?}", other),
    }
}

#[test]
fn assert_floor_fails_when_floor_label_unparseable() {
    let floor = FidelityFloor::new("F9");
    let err = assert_fidelity_floor(&envelope("V1", Some("F3")), &floor).unwrap_err();
    match err {
        FidelityFloorError::UnparseableFloor { value } => assert_eq!(value, "F9"),
        other => panic!("expected UnparseableFloor, got {:?}", other),
    }
}
