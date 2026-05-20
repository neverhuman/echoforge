//! Integration tests against the seeded `object-packs/` directory.
//!
//! These tests assert that the existing seeded packs (`public-proxy*`
//! and `hard-negatives`) can be discovered, parsed, and JSON-schema
//! validated end-to-end via the public `Registry` + `SchemaCatalog`
//! surface. They are the Wave-8 acceptance gate for downstream
//! consumers (Wave 9 case study + Wave 13 runtime migration).

use std::path::PathBuf;

use echoforge_packs::validation::SchemaCatalog;
use echoforge_packs::Registry;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn seeded_packs_discovered_with_expected_card_counts() {
    let registry = Registry::discover(&repo_root().join("object-packs"))
        .expect("seeded object-packs/ must be discoverable");
    assert!(
        registry.pack_count() >= 2,
        "expected ≥2 seeded packs, found {}",
        registry.pack_count()
    );
    // Hard-negatives ships 23 object_card files; tolerate small drift.
    let hn = registry.pack("hard-negatives").expect("hard-negatives pack");
    let object_card_count = hn
        .cards
        .get("object_card")
        .map(|m| m.len())
        .unwrap_or(0);
    assert!(
        object_card_count >= 20,
        "hard-negatives expected ≥20 object_card entries, got {}",
        object_card_count
    );
}

#[test]
fn seeded_pack_cards_validate_against_schema_catalog() {
    let registry = Registry::discover(&repo_root().join("object-packs")).unwrap();
    let mut catalog =
        SchemaCatalog::from_repo(&repo_root()).expect("schemas/ exists in repo");
    // Schema validation should pass for every seeded card today: the
    // hard-negatives and public-proxy fixtures all conform to
    // object_card.schema.json / material_card.schema.json.
    match registry.validate_with_catalog(&mut catalog) {
        Ok(()) => {}
        Err(errs) => {
            // Surface the full list so the failure message is debuggable.
            panic!(
                "seeded pack cards failed schema validation:\n{}",
                errs.iter()
                    .map(|e| format!(" - {e}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
    }
    // Confirm the catalog actually compiled at least one validator
    // (otherwise the pass would be vacuous because no cards routed).
    assert!(
        catalog.cached_validator_count() >= 1,
        "expected ≥1 cached validator after validation"
    );
}

#[test]
fn find_card_locates_hard_negatives_by_slug() {
    let registry = Registry::discover(&repo_root().join("object-packs")).unwrap();
    let bird = registry
        .find_card("object_card", "bird-flock-dense")
        .expect("bird-flock-dense must be loadable from hard-negatives pack");
    assert_eq!(bird.kind, "object_card");
    assert_eq!(bird.slug, "bird-flock-dense");
    let rain = registry
        .find_card("object_card", "rain-cell")
        .expect("rain-cell card must be present");
    assert_eq!(rain.slug, "rain-cell");
}

#[test]
fn total_card_count_non_zero_across_seeded_packs() {
    let registry = Registry::discover(&repo_root().join("object-packs")).unwrap();
    assert!(
        registry.total_card_count() > 20,
        "expected >20 cards total across seeded packs, got {}",
        registry.total_card_count()
    );
}
