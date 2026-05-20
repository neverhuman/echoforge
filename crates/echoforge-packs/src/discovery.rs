//! Filesystem discovery and per-card loading.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;

use crate::error::PackError;
use crate::manifest::PackManifest;
use crate::registry::CardPayload;

/// Walk `packs_root` (typically `<repo>/object-packs/`) and return every
/// directory that contains a `pack.manifest.json`. Sub-pack directories
/// (rare; intended for future use) are not recursed: discovery stops at
/// the first manifest in each subtree.
pub fn discover_packs(packs_root: &Path) -> Result<Vec<PathBuf>, PackError> {
    if !packs_root.exists() {
        return Err(PackError::Io {
            path: packs_root.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("packs root not found: {}", packs_root.display()),
            ),
        });
    }
    let mut packs = Vec::new();
    let entries = std::fs::read_dir(packs_root).map_err(|e| PackError::Io {
        path: packs_root.to_path_buf(),
        source: e,
    })?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && path.join("pack.manifest.json").exists() {
            packs.push(path);
        }
    }
    packs.sort();
    Ok(packs)
}

/// Load every card declared by the manifest, parsing YAML or JSON based
/// on extension. Returns a (kind -> slug -> CardPayload) two-level map.
pub fn load_pack_cards(
    pack_root: &Path,
    manifest: &PackManifest,
) -> Result<BTreeMap<String, BTreeMap<String, CardPayload>>, PackError> {
    let mut out: BTreeMap<String, BTreeMap<String, CardPayload>> = BTreeMap::new();
    for (default_kind, paths) in &manifest.card_refs {
        for rel in paths {
            let path = pack_root.join(rel);
            let value = load_card_value(&path)?;
            // The declared kind in the manifest is a hint; the actual
            // `kind` field on the card payload is authoritative when
            // present (lets a pack declare a card under one bucket but
            // the card itself self-identify as a more specific kind).
            let kind = match value.get("kind").and_then(Value::as_str) {
                Some(k) => k.to_string(),
                None => default_kind.clone(),
            };
            let slug = match value.get("public_proxy_id").and_then(Value::as_str) {
                Some(id) => id.to_string(),
                None => derive_slug_from_path(&path),
            };
            let by_slug = out.entry(kind.clone()).or_default();
            if let Some(existing) = by_slug.get(&slug) {
                return Err(PackError::DuplicateCard {
                    pack: manifest.id.clone(),
                    slug,
                    existing: existing.path.clone(),
                });
            }
            by_slug.insert(
                slug.clone(),
                CardPayload {
                    path: path.clone(),
                    kind,
                    slug,
                    value: Arc::new(value),
                },
            );
        }
    }
    Ok(out)
}

fn load_card_value(path: &Path) -> Result<Value, PackError> {
    let text = std::fs::read_to_string(path).map_err(|e| PackError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let ext = match path.extension().and_then(|s| s.to_str()) {
        Some(s) => s.to_ascii_lowercase(),
        None => String::new(),
    };
    match ext.as_str() {
        "yaml" | "yml" => serde_yaml::from_str(&text).map_err(|e| PackError::Yaml {
            path: path.to_path_buf(),
            source: e,
        }),
        // Default to JSON for .json and unknown extensions; producers
        // who ship odd-extension cards should rename them.
        _ => serde_json::from_str(&text).map_err(|e| PackError::Json {
            path: path.to_path_buf(),
            source: e,
        }),
    }
}

fn derive_slug_from_path(path: &Path) -> String {
    match path.file_stem().and_then(|s| s.to_str()) {
        Some(s) => s.to_string(),
        None => String::from("unknown"),
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
    fn discover_packs_finds_seeded_two_or_more() {
        let packs =
            discover_packs(&repo_root().join("object-packs")).expect("object-packs exists");
        assert!(
            packs.len() >= 2,
            "expected ≥2 seeded packs, found {}",
            packs.len()
        );
    }

    #[test]
    fn discover_packs_missing_root_errors() {
        let err =
            discover_packs(&repo_root().join("does-not-exist-zz")).unwrap_err();
        assert!(matches!(err, PackError::Io { .. }));
    }
}
