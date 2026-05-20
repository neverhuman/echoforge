//! Mining-loop driver.
//!
//! Orchestrates one iteration of the closed-loop curriculum:
//!
//! 1. Load every `model_eval_*.json` rollup under `<campaign_root>/qa/`.
//! 2. Build a `FailureClusterStore` from them at the configured threshold.
//! 3. Look up the base `ObjectClass` for each cluster from the supplied
//!    `airspace-objects.json` library and call `synthesize_batch` to
//!    produce variant proposals.
//! 4. Write:
//!    `<output_dir>/iteration-<N>/failure_clusters.json`,
//!    `<output_dir>/iteration-<N>/proposed_variants.json`, and
//!    `<output_dir>/iteration-<N>/iteration_manifest.json`.
//!
//! The loop deliberately stops here. Applying variants to the live
//! `airspace-objects.json` is the responsibility of a follow-up
//! `apply_variants` packet so a curator stays in the loop. The driver returns
//! a `MiningIterationReport` summarising what it wrote.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::cluster::FailureClusterStore;
use crate::error::MiningError;
use crate::synthesize::{synthesize_batch, ObjectClassDelta};
use crate::types::AirspaceObjectsConfig;

/// Inputs to one mining iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningLoopConfig {
    /// Campaign root (must contain a `qa/` directory with at least one
    /// `model_eval_*.json` file).
    pub campaign_root: PathBuf,
    /// Path to the object library to read base classes from.
    pub airspace_objects_path: PathBuf,
    /// FP rate above which a cluster is considered trouble. Recommended
    /// default 0.15.
    pub fp_threshold: f64,
    /// Cap on the number of variants written per iteration.
    pub max_variants_per_iteration: usize,
    /// Root directory under which `iteration-<N>/` will be created.
    pub output_dir: PathBuf,
    /// Optional human-readable iteration label written into the manifest;
    /// the directory name is derived from `iteration_number`.
    #[serde(default)]
    pub iteration_label: Option<String>,
    /// Iteration number. Drives the output directory name
    /// (`iteration-<N>`) AND the starting variant index so variant ids are
    /// globally unique across iterations: `<base>-mined-variant-<iteration*100 + i>`.
    pub iteration_number: u32,
}

impl MiningLoopConfig {
    /// Convenience constructor with the recommended defaults.
    pub fn new(
        campaign_root: PathBuf,
        airspace_objects_path: PathBuf,
        output_dir: PathBuf,
        iteration_number: u32,
    ) -> Self {
        Self {
            campaign_root,
            airspace_objects_path,
            fp_threshold: 0.15,
            max_variants_per_iteration: 8,
            output_dir,
            iteration_label: None,
            iteration_number,
        }
    }
}

/// Report returned by `run_one_iteration`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningIterationReport {
    pub iteration_number: u32,
    pub iteration_label: Option<String>,
    pub iteration_dir: PathBuf,
    pub cluster_count: usize,
    pub variant_count: usize,
    pub fp_threshold: f64,
    pub clusters_path: PathBuf,
    pub variants_path: PathBuf,
    pub manifest_path: PathBuf,
    pub source_rollups: Vec<String>,
    /// Set when the cluster store is empty so callers can short-circuit a
    /// dataset-rebuild without inspecting cluster_count.
    pub no_failures: bool,
}

/// On-disk manifest written alongside the cluster / variant outputs.
///
/// The schema is intentionally narrow so curator review tools can pick up
/// just the bits that matter (which iteration, what config, which detectors
/// contributed). Kept separate from `MiningIterationReport` because that
/// type holds absolute paths the caller cares about; the manifest holds
/// relative paths so the iteration dir can be moved/inspected in isolation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationManifest {
    pub iteration_number: u32,
    pub iteration_label: Option<String>,
    pub campaign_root: String,
    pub airspace_objects_path: String,
    pub fp_threshold: f64,
    pub max_variants_per_iteration: usize,
    pub cluster_count: usize,
    pub variant_count: usize,
    pub source_rollups: Vec<String>,
    pub clusters_filename: String,
    pub variants_filename: String,
}

/// Run one iteration end-to-end.
pub fn run_one_iteration(cfg: &MiningLoopConfig) -> Result<MiningIterationReport, MiningError> {
    let store = FailureClusterStore::from_campaign_root(&cfg.campaign_root, cfg.fp_threshold)?;

    let airspace_text =
        fs::read_to_string(&cfg.airspace_objects_path).map_err(|e| MiningError::Io {
            path: cfg.airspace_objects_path.display().to_string(),
            source: e,
        })?;
    let airspace: AirspaceObjectsConfig =
        serde_json::from_str(&airspace_text).map_err(|e| MiningError::Json {
            path: cfg.airspace_objects_path.display().to_string(),
            source: e,
        })?;

    let starting_index = cfg.iteration_number.saturating_mul(100);
    let variants = synthesize_batch(
        &store.clusters,
        &airspace.object_classes,
        cfg.fp_threshold,
        cfg.max_variants_per_iteration,
        starting_index,
    );

    let iteration_dir = cfg
        .output_dir
        .join(format!("iteration-{}", cfg.iteration_number));
    fs::create_dir_all(&iteration_dir).map_err(|e| MiningError::Output {
        path: iteration_dir.display().to_string(),
        source: e,
    })?;

    let clusters_path = iteration_dir.join("failure_clusters.json");
    let variants_path = iteration_dir.join("proposed_variants.json");
    let manifest_path = iteration_dir.join("iteration_manifest.json");

    write_json(&clusters_path, &store)?;
    write_json(
        &variants_path,
        &VariantsOutput {
            variants: variants.clone(),
        },
    )?;

    let manifest = IterationManifest {
        iteration_number: cfg.iteration_number,
        iteration_label: cfg.iteration_label.clone(),
        campaign_root: cfg.campaign_root.display().to_string(),
        airspace_objects_path: cfg.airspace_objects_path.display().to_string(),
        fp_threshold: cfg.fp_threshold,
        max_variants_per_iteration: cfg.max_variants_per_iteration,
        cluster_count: store.clusters.len(),
        variant_count: variants.len(),
        source_rollups: store.source_rollups.clone(),
        clusters_filename: "failure_clusters.json".to_string(),
        variants_filename: "proposed_variants.json".to_string(),
    };
    write_json(&manifest_path, &manifest)?;

    Ok(MiningIterationReport {
        iteration_number: cfg.iteration_number,
        iteration_label: cfg.iteration_label.clone(),
        iteration_dir,
        cluster_count: store.clusters.len(),
        variant_count: variants.len(),
        fp_threshold: cfg.fp_threshold,
        clusters_path,
        variants_path,
        manifest_path,
        source_rollups: store.source_rollups,
        no_failures: store.clusters.is_empty(),
    })
}

/// Wrapper struct so the variants file has a stable top-level shape (curator
/// tools can rely on `{ "variants": [ ... ] }` instead of a bare array).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VariantsOutput {
    variants: Vec<ObjectClassDelta>,
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), MiningError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| MiningError::Output {
            path: parent.display().to_string(),
            source: e,
        })?;
    }
    let text = serde_json::to_string_pretty(value).map_err(|e| MiningError::Json {
        path: path.display().to_string(),
        source: e,
    })?;
    fs::write(path, text).map_err(|e| MiningError::Output {
        path: path.display().to_string(),
        source: e,
    })?;
    Ok(())
}
