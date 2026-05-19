//! In-memory registry of loaded packs and their cards.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;

use crate::discovery::{discover_packs, load_pack_cards};
use crate::error::PackError;
use crate::manifest::PackManifest;
use crate::validation::SchemaCatalog;

/// One loaded card payload: the parsed JSON/YAML value + the absolute
/// path it was loaded from. Kept opaque (a raw `serde_json::Value`) so
/// the registry stays decoupled from downstream typed structs; consumers
/// deserialize into their own types via `serde_json::from_value`.
#[derive(Debug, Clone)]
pub struct CardPayload {
    pub path: PathBuf,
    pub kind: String,
    pub slug: String,
    pub value: Arc<Value>,
}

/// One loaded pack: its manifest and the cards grouped by kind.
#[derive(Debug, Clone)]
pub struct PackEntry {
    pub root: PathBuf,
    pub manifest: PackManifest,
    /// kind -> slug -> CardPayload
    pub cards: BTreeMap<String, BTreeMap<String, CardPayload>>,
}

impl PackEntry {
    /// Total cards loaded across all kinds.
    pub fn card_count(&self) -> usize {
        self.cards.values().map(BTreeMap::len).sum()
    }

    /// Iterate all loaded cards in (kind, slug) order.
    pub fn iter_cards(&self) -> impl Iterator<Item = (&str, &str, &CardPayload)> {
        self.cards.iter().flat_map(|(kind, by_slug)| {
            by_slug
                .iter()
                .map(move |(slug, card)| (kind.as_str(), slug.as_str(), card))
        })
    }
}

/// Top-level pack registry — discovered packs keyed by pack slug.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    packs: BTreeMap<String, PackEntry>,
}

impl Registry {
    /// Walk `packs_root` (typically `<repo>/object-packs/`) and return a
    /// fully populated registry. Each manifest is validated and each
    /// card is JSON-schema-validated against the appropriate
    /// `schemas/<kind>.schema.json`. Schema validation failures are
    /// **soft warnings** by default — the card is still registered but
    /// its `validation` envelope is augmented. Caller can inspect via
    /// [`Registry::validate_with_catalog`].
    pub fn discover(packs_root: &Path) -> Result<Self, PackError> {
        let mut registry = Self::default();
        for pack_path in discover_packs(packs_root)? {
            let manifest_path = pack_path.join("pack.manifest.json");
            let raw = std::fs::read_to_string(&manifest_path).map_err(|e| PackError::Io {
                path: manifest_path.clone(),
                source: e,
            })?;
            let value: Value = serde_json::from_str(&raw).map_err(|e| PackError::Json {
                path: manifest_path.clone(),
                source: e,
            })?;
            let manifest = PackManifest::from_value(&manifest_path, &value)?;
            let cards = load_pack_cards(&pack_path, &manifest)?;
            let pack_slug = manifest.id.trim_start_matches("pack.").to_string();
            if let Some(existing) = registry.packs.get(&pack_slug) {
                return Err(PackError::DuplicatePack {
                    slug: pack_slug,
                    existing: existing.root.clone(),
                });
            }
            registry.packs.insert(
                pack_slug,
                PackEntry {
                    root: pack_path,
                    manifest,
                    cards,
                },
            );
        }
        Ok(registry)
    }

    /// Validate every loaded card against its declared schema using the
    /// supplied catalog. Returns `Ok(())` when all cards pass; returns
    /// the first error otherwise. Use [`SchemaCatalog::from_repo`] to
    /// build the catalog.
    pub fn validate_with_catalog(
        &self,
        catalog: &mut SchemaCatalog,
    ) -> Result<(), Vec<PackError>> {
        let mut errs = Vec::new();
        for entry in self.packs.values() {
            for (_kind, _slug, card) in entry.iter_cards() {
                if let Err(e) = catalog.validate(&card.path, &card.value) {
                    errs.push(e);
                }
            }
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }

    /// Iterate packs in slug order.
    pub fn packs(&self) -> impl Iterator<Item = (&str, &PackEntry)> {
        self.packs.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Total pack count.
    pub fn pack_count(&self) -> usize {
        self.packs.len()
    }

    /// Total card count across all packs.
    pub fn total_card_count(&self) -> usize {
        self.packs.values().map(PackEntry::card_count).sum()
    }

    /// Look up a pack by slug (e.g. `"public-proxy"`, `"hard-negatives"`,
    /// `"radar-platforms-v1"`).
    pub fn pack(&self, slug: &str) -> Option<&PackEntry> {
        self.packs.get(slug)
    }

    /// Look up a single card by `(kind, slug)` across all packs. Returns
    /// the first match in pack-slug order; useful when callers do not
    /// know which pack owns a given card.
    pub fn find_card(&self, kind: &str, card_slug: &str) -> Option<&CardPayload> {
        self.packs.values().find_map(|p| {
            p.cards
                .get(kind)
                .and_then(|by_slug| by_slug.get(card_slug))
        })
    }

    /// Return the minimum `validation.tier` string across the cards in
    /// the named pack, or `None` if the pack does not exist or has no
    /// cards. Tier ordering: `unvalidated` < `basic` < `cross_checked`
    /// < `benchmarked` < `measured`.
    pub fn min_tier_across_pack(&self, pack_slug: &str) -> Option<String> {
        let entry = self.packs.get(pack_slug)?;
        let mut min_rank = usize::MAX;
        let mut min_label: Option<String> = None;
        for (_k, _s, card) in entry.iter_cards() {
            let tier = card
                .value
                .get("validation")
                .and_then(|v| v.get("tier"))
                .and_then(Value::as_str)
                .unwrap_or("unvalidated");
            let rank = tier_rank(tier);
            if rank < min_rank {
                min_rank = rank;
                min_label = Some(tier.to_string());
            }
        }
        min_label
    }
}

fn tier_rank(tier: &str) -> usize {
    match tier {
        "unvalidated" => 0,
        "basic" => 1,
        "cross_checked" => 2,
        "benchmarked" => 3,
        "measured" => 4,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    #[test]
    fn discover_seeded_packs_includes_public_proxy_and_hard_negatives() {
        let registry = Registry::discover(&repo_root().join("object-packs"))
            .expect("seeded object-packs/ should exist");
        assert!(registry.pack_count() >= 2);
        let slugs: Vec<&str> = registry.packs().map(|(s, _)| s).collect();
        assert!(
            slugs.iter().any(|s| s.starts_with("public-proxy")),
            "expected a `public-proxy*` pack, got {:?}",
            slugs
        );
        assert!(
            slugs.iter().any(|s| *s == "hard-negatives"),
            "expected `hard-negatives` pack, got {:?}",
            slugs
        );
    }

    #[test]
    fn registered_cards_include_at_least_one_object_card() {
        let registry = Registry::discover(&repo_root().join("object-packs")).unwrap();
        assert!(
            registry.total_card_count() > 0,
            "expected at least one card across all packs"
        );
        // Hard-negatives ships 23 object cards.
        let hn = registry.pack("hard-negatives").expect("hard-negatives pack");
        assert!(
            hn.cards
                .get("object_card")
                .map(BTreeMap::len)
                .unwrap_or(0)
                >= 20,
            "hard-negatives expected ≥20 object cards, got {}",
            hn.cards
                .get("object_card")
                .map(BTreeMap::len)
                .unwrap_or(0)
        );
    }

    #[test]
    fn min_tier_across_pack_returns_unvalidated_for_pending_fixtures() {
        let registry = Registry::discover(&repo_root().join("object-packs")).unwrap();
        // The seeded hard-negatives cards declare tier="unvalidated".
        let tier = registry.min_tier_across_pack("hard-negatives");
        assert_eq!(tier.as_deref(), Some("unvalidated"));
    }

    #[test]
    fn find_card_locates_seeded_object_card() {
        let registry = Registry::discover(&repo_root().join("object-packs")).unwrap();
        // The hard-negatives pack ships a bird-flock-dense card.
        assert!(
            registry.find_card("object_card", "bird-flock-dense").is_some(),
            "expected to find bird-flock-dense"
        );
    }
}
