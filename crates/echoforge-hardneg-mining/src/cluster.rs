//! Failure-cluster store.
//!
//! Reads one or more `model_eval_*.json` rollups from a campaign root and
//! aggregates per-class false-alarm statistics into `FailureCluster`s. A
//! cluster is "trouble" when its false-alarm rate (false-positive count
//! divided by the negative-record count for the model that flagged it)
//! exceeds the configured threshold.
//!
//! The store deliberately uses the `false_alarm_by_hard_negative_family`
//! map keyed by family-id (e.g. `"commercial_aircraft_corridor"`) because
//! that's the level at which the upstream `evaluate_model` aggregator emits
//! counts; we treat the family-id as the cluster's `class_id`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::MiningError;
use crate::types::ModelEvalRollup;

/// One trouble cluster — a (class_id, family) pair whose false-alarm rate
/// across all considered detectors exceeded the configured threshold.
///
/// `representative_features` is a string-keyed numeric map used by the
/// synthesizer to decide which envelope to widen. Today the store populates
/// `fp_rate` and per-model `fp_rate__<model_id>` entries; the synthesizer
/// treats unknown keys as opaque so the schema is forward-compatible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailureCluster {
    /// The hard-negative family key as reported by the model_eval rollup.
    /// Treated as the cluster's class identifier.
    pub class_id: String,
    /// Total number of negative-record evaluations summed across all
    /// considered detectors.
    pub frame_count: usize,
    /// Total false-positive count summed across all considered detectors.
    pub fp_count: usize,
    /// `fp_count / frame_count`. Always in `[0.0, 1.0]`.
    pub fp_rate: f64,
    /// Negative-record IDs sampled from this cluster (limited to
    /// `MAX_SAMPLE_RECORDS_PER_CLUSTER` for readability). Empty when the
    /// rollup does not provide per-record breakdowns; future versions of the
    /// upstream rollup may populate this.
    pub sample_record_ids: Vec<String>,
    /// Forward-compatible feature bag. Currently:
    ///   * `fp_rate` (overall)
    ///   * `fp_rate__<model_id>` for each detector that contributed FPs
    ///   * `fp_count`
    ///   * `frame_count`
    pub representative_features: BTreeMap<String, f64>,
    /// Which detectors contributed to this cluster. Useful for triage even
    /// though the synthesizer doesn't read it.
    pub contributing_models: Vec<String>,
}

/// Cap on the number of sample record IDs surfaced per cluster. Keeps the
/// emitted JSON readable when a detector mis-fires hundreds of times.
pub const MAX_SAMPLE_RECORDS_PER_CLUSTER: usize = 12;

/// In-memory failure cluster store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FailureClusterStore {
    pub clusters: Vec<FailureCluster>,
    /// Threshold the store was built with (echoed in the output for audit).
    pub fp_threshold: f64,
    /// Which model_eval files contributed to the rollup (basename + path).
    pub source_rollups: Vec<String>,
}

impl FailureClusterStore {
    /// Build a store from a campaign root's `qa/model_eval_*.json` files.
    ///
    /// `fp_threshold` is the false-alarm rate above which a cluster is
    /// considered "trouble"; clusters at or below the threshold are dropped
    /// so the output is curator-actionable.
    pub fn from_campaign_root(
        campaign_root: &Path,
        fp_threshold: f64,
    ) -> Result<Self, MiningError> {
        if !(fp_threshold.is_finite() && fp_threshold > 0.0 && fp_threshold <= 1.0) {
            return Err(MiningError::InvalidThreshold {
                threshold: fp_threshold,
            });
        }
        if !campaign_root.exists() {
            return Err(MiningError::MissingCampaignRoot {
                path: campaign_root.display().to_string(),
            });
        }

        let qa_dir = campaign_root.join("qa");
        let rollups = load_rollups(&qa_dir)?;
        if rollups.is_empty() {
            return Err(MiningError::NoModelEvalReports {
                path: qa_dir.display().to_string(),
            });
        }

        let source_rollups: Vec<String> = rollups.iter().map(|(path, _)| path.clone()).collect();
        let rollup_refs: Vec<&ModelEvalRollup> = rollups.iter().map(|(_, r)| r).collect();
        let clusters = aggregate_clusters(&rollup_refs, fp_threshold);

        Ok(Self {
            clusters,
            fp_threshold,
            source_rollups,
        })
    }

    /// Build a store directly from in-memory rollups. Used by tests and by
    /// callers that already have the rollups loaded (e.g. the streaming
    /// runner could feed reports in without round-tripping disk).
    pub fn from_rollups(
        rollups: &[ModelEvalRollup],
        fp_threshold: f64,
    ) -> Result<Self, MiningError> {
        if !(fp_threshold.is_finite() && fp_threshold > 0.0 && fp_threshold <= 1.0) {
            return Err(MiningError::InvalidThreshold {
                threshold: fp_threshold,
            });
        }
        let rollup_refs: Vec<&ModelEvalRollup> = rollups.iter().collect();
        let clusters = aggregate_clusters(&rollup_refs, fp_threshold);
        Ok(Self {
            clusters,
            fp_threshold,
            source_rollups: rollups.iter().map(|r| r.model_id.clone()).collect(),
        })
    }
}

fn load_rollups(qa_dir: &Path) -> Result<Vec<(String, ModelEvalRollup)>, MiningError> {
    let entries = match fs::read_dir(qa_dir) {
        Ok(e) => e,
        Err(_) => return Ok(Vec::new()),
    };
    let mut rollups = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| MiningError::Io {
            path: qa_dir.display().to_string(),
            source: e,
        })?;
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if !(name.starts_with("model_eval_") && name.ends_with(".json")) {
            continue;
        }
        let text = fs::read_to_string(&path).map_err(|e| MiningError::Io {
            path: path.display().to_string(),
            source: e,
        })?;
        let rollup: ModelEvalRollup =
            serde_json::from_str(&text).map_err(|e| MiningError::Json {
                path: path.display().to_string(),
                source: e,
            })?;
        rollups.push((path.display().to_string(), rollup));
    }
    rollups.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(rollups)
}

fn aggregate_clusters(rollups: &[&ModelEvalRollup], fp_threshold: f64) -> Vec<FailureCluster> {
    // First, find the union of all families any rollup mentioned. A family
    // is a cluster candidate iff at least one detector flagged a member of
    // it as a false alarm. We then evaluate the cluster's fp_rate against
    // ALL contributing detectors' negative-record counts (not just the ones
    // that reported a non-zero FP for the family), so detectors that
    // correctly stayed silent on the family lower the aggregate rate
    // rather than disappearing from the denominator.
    let mut families: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for rollup in rollups {
        for family in rollup.false_alarm_by_hard_negative_family.keys() {
            if !family.is_empty() {
                families.insert(family.clone());
            }
        }
    }

    let mut accum: BTreeMap<String, ClusterAccum> = BTreeMap::new();
    for family in &families {
        let mut acc = ClusterAccum::default();
        for rollup in rollups {
            let fp = rollup
                .false_alarm_by_hard_negative_family
                .get(family)
                .copied()
                .unwrap_or(0);
            // Every rollup contributes its negative_records to the
            // denominator — that's the "of the negatives this detector saw,
            // how many were misclassified" interpretation, summed.
            acc.neg_count = acc.neg_count.saturating_add(rollup.negative_records);
            acc.fp_count = acc.fp_count.saturating_add(fp);
            acc.per_model_fp.insert(rollup.model_id.clone(), fp as f64);
            acc.per_model_neg
                .insert(rollup.model_id.clone(), rollup.negative_records);
            if fp > 0 {
                acc.contributing_models.push(rollup.model_id.clone());
            }
        }
        accum.insert(family.clone(), acc);
    }

    let mut clusters = Vec::new();
    for (class_id, acc) in accum {
        let fp_rate = if acc.neg_count == 0 {
            0.0
        } else {
            acc.fp_count as f64 / acc.neg_count as f64
        };
        if fp_rate <= fp_threshold {
            continue;
        }
        let mut features: BTreeMap<String, f64> = BTreeMap::new();
        features.insert("fp_rate".to_string(), fp_rate);
        features.insert("fp_count".to_string(), acc.fp_count as f64);
        features.insert("frame_count".to_string(), acc.neg_count as f64);
        for (model_id, fp) in &acc.per_model_fp {
            let neg = acc.per_model_neg.get(model_id).copied().unwrap_or(0) as f64;
            if neg > 0.0 {
                features.insert(format!("fp_rate__{model_id}"), fp / neg);
            }
        }
        clusters.push(FailureCluster {
            class_id,
            frame_count: acc.neg_count,
            fp_count: acc.fp_count,
            fp_rate,
            sample_record_ids: Vec::new(),
            representative_features: features,
            contributing_models: acc.contributing_models,
        });
    }
    // Deterministic ordering: highest FP rate first, then class_id.
    clusters.sort_by(|a, b| {
        b.fp_rate
            .partial_cmp(&a.fp_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.class_id.cmp(&b.class_id))
    });
    clusters
}

#[derive(Debug, Default)]
struct ClusterAccum {
    fp_count: usize,
    neg_count: usize,
    contributing_models: Vec<String>,
    per_model_fp: BTreeMap<String, f64>,
    per_model_neg: BTreeMap<String, usize>,
}
