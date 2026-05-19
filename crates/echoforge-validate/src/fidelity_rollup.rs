//! Fidelity-class (F0-F5) rollup + promotion-floor helper.
//!
//! Wires the optional `ValidationInfo.fidelity_class` field — added in the
//! wave-2 `fidelity-class-field` packet (see
//! `.agents/receipts/fidelity-class-field/`) and documented in
//! `docs/validation-tiers.md` — into reporting helpers usable by
//! downstream consumers.
//!
//! Two distinct surfaces:
//!
//! - [`rollup_validation_envelopes`] aggregates a slice of
//!   [`ValidationInfo`] envelopes into a [`FidelityRollup`] with
//!   per-fidelity-class counts and a (tier x fidelity) cross-axis
//!   summary. Envelopes without a declared `fidelity_class` are bucketed
//!   into `without_fidelity_class` so the count stays auditable.
//! - [`assert_fidelity_floor`] enforces a minimum F-class on a single
//!   envelope. Returns [`FidelityFloorError`] when the envelope is
//!   missing the field, carries an unparseable label, or sits below the
//!   numeric floor.
//!
//! Per FUCKIT.md `validate-fidelity-rollup` boundaries, this module is
//! purely additive: it does **not** alter the V0/V1/V2 promotion gate in
//! [`crate::report`] and does **not** modify `ValidationInfo` itself.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use echoforge_core::ValidationInfo;

/// Parse a fidelity-class string ("F0".."F5") into its numeric rank.
///
/// Returns `None` for unknown labels (e.g. `"F6"`, `"f2"`, `""`, `"V2"`).
/// Parsing is intentionally strict — uppercase prefix `F` followed by a
/// single decimal digit `0..=5`. Lowercase or whitespace-padded inputs
/// are rejected so the rollup never silently coerces malformed values.
pub fn parse_fidelity_class(s: &str) -> Option<u8> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    if bytes[0] != b'F' {
        return None;
    }
    match bytes[1] {
        b'0' => Some(0),
        b'1' => Some(1),
        b'2' => Some(2),
        b'3' => Some(3),
        b'4' => Some(4),
        b'5' => Some(5),
        _ => None,
    }
}

/// Aggregate fidelity-class statistics over a set of validation envelopes.
///
/// `by_fidelity` keys are the literal label strings observed
/// (e.g. `"F0"`, `"F2"`); unknown labels are passed through verbatim so
/// the rollup never silently drops data. `by_tier_x_fidelity` is keyed on
/// `(tier, fidelity)` and serves the dual-axis dashboards described in
/// `docs/validation-tiers.md`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FidelityRollup {
    pub artifact_count: usize,
    pub with_fidelity_class: usize,
    pub without_fidelity_class: usize,
    pub by_fidelity: BTreeMap<String, usize>,
    pub by_tier_x_fidelity: BTreeMap<(String, String), usize>,
}

/// Minimum acceptable fidelity class for a promotion check.
///
/// `min_fidelity` is the literal label (e.g. `"F2"`); it is parsed via
/// [`parse_fidelity_class`] each time [`assert_fidelity_floor`] is
/// invoked so that floor values stored as strings (e.g. in a producer
/// config) round-trip without an upstream conversion step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FidelityFloor {
    pub min_fidelity: String,
}

impl FidelityFloor {
    pub fn new(min_fidelity: impl Into<String>) -> Self {
        Self {
            min_fidelity: min_fidelity.into(),
        }
    }
}

/// Reasons an envelope can fail the fidelity-floor check.
#[derive(Debug, Error)]
pub enum FidelityFloorError {
    #[error("envelope has no fidelity_class declared (floor requires at least {floor})")]
    Missing { floor: String },
    #[error("envelope fidelity_class \"{value}\" is not a recognised F0..F5 label")]
    UnparseableEnvelope { value: String },
    #[error("floor min_fidelity \"{value}\" is not a recognised F0..F5 label")]
    UnparseableFloor { value: String },
    #[error(
        "envelope fidelity_class \"{envelope}\" (F{got}) is below floor \"{floor}\" (F{required})"
    )]
    Below {
        envelope: String,
        floor: String,
        got: u8,
        required: u8,
    },
}

/// Walk a slice of envelopes and emit a [`FidelityRollup`].
///
/// `envelopes` is a `&[&ValidationInfo]` so callers can pass references
/// they already hold (e.g. each card's `.validation`) without cloning.
/// The rollup is deterministic: `BTreeMap` ordering guarantees stable
/// serialization.
pub fn rollup_validation_envelopes(envelopes: &[&ValidationInfo]) -> FidelityRollup {
    let mut rollup = FidelityRollup {
        artifact_count: envelopes.len(),
        ..FidelityRollup::default()
    };

    for env in envelopes {
        match env.fidelity_class.as_deref() {
            Some(class) => {
                rollup.with_fidelity_class += 1;
                *rollup.by_fidelity.entry(class.to_string()).or_insert(0) += 1;
                *rollup
                    .by_tier_x_fidelity
                    .entry((env.tier.clone(), class.to_string()))
                    .or_insert(0) += 1;
            }
            None => {
                rollup.without_fidelity_class += 1;
            }
        }
    }

    rollup
}

/// Assert that an envelope's `fidelity_class` meets the configured floor.
///
/// Errors when (a) the envelope omits `fidelity_class`, (b) the
/// envelope's label cannot be parsed as F0..F5, (c) the floor's label
/// cannot be parsed as F0..F5, or (d) the envelope's class is strictly
/// below the floor.
pub fn assert_fidelity_floor(
    envelope: &ValidationInfo,
    floor: &FidelityFloor,
) -> Result<(), FidelityFloorError> {
    let required = match parse_fidelity_class(&floor.min_fidelity) {
        Some(v) => v,
        None => {
            return Err(FidelityFloorError::UnparseableFloor {
                value: floor.min_fidelity.clone(),
            })
        }
    };

    let envelope_label = match envelope.fidelity_class.as_deref() {
        Some(value) => value,
        None => {
            return Err(FidelityFloorError::Missing {
                floor: floor.min_fidelity.clone(),
            })
        }
    };

    let got = match parse_fidelity_class(envelope_label) {
        Some(v) => v,
        None => {
            return Err(FidelityFloorError::UnparseableEnvelope {
                value: envelope_label.to_string(),
            })
        }
    };

    if got < required {
        return Err(FidelityFloorError::Below {
            envelope: envelope_label.to_string(),
            floor: floor.min_fidelity.clone(),
            got,
            required,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
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
            envelope("V0", None), // ignored on tier x fidelity axis
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
        // Envelopes without fidelity_class are NOT counted in the cross
        // axis (the pair key would be ambiguous).
        assert_eq!(rollup.by_tier_x_fidelity.len(), 3);
    }

    #[test]
    fn rollup_preserves_unknown_fidelity_labels() {
        // Unknown labels are still aggregated verbatim so the rollup
        // surfaces producer drift instead of silently dropping data.
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
}
