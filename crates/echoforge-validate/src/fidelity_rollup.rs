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
#[path = "fidelity_rollup_tests.rs"]
mod tests;
