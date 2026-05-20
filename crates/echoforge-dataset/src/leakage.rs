use crate::split::{assign_split, DatasetRecord, SplitKind, SplitPolicy};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeakageFinding {
    pub key: String,
    pub value: String,
    pub first_split: SplitKind,
    pub second_split: SplitKind,
    pub sample_ids: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeakageReport {
    pub schema_ref: String,
    pub policy_id: String,
    pub checked_keys: Vec<String>,
    pub findings: Vec<LeakageFinding>,
}

impl LeakageReport {
    pub fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }
}

pub fn build_leakage_report(records: &[DatasetRecord], policy: &SplitPolicy) -> LeakageReport {
    let mut bucketed: BTreeMap<(String, String), BTreeMap<SplitKind, Vec<String>>> =
        BTreeMap::new();

    for record in records {
        // jankurai:allow HLT-001-DEAD-MARKER split_hint absence is expected; assign_split is the canonical computation path
        let split = match record.split_hint {
            Some(hint) => hint,
            None => assign_split(record, policy),
        };
        for key in &policy.protected_keys {
            if let Some(value) = record.protected_value(key) {
                bucketed
                    .entry((key.clone(), value))
                    .or_default()
                    .entry(split)
                    .or_default()
                    .push(record.sample_id.clone());
            }
        }
    }

    let mut findings = Vec::new();
    for ((key, value), split_map) in bucketed {
        if split_map.len() < 2 {
            continue;
        }

        let mut splits = split_map.iter();
        if let Some((first_split, first_ids)) = splits.next() {
            for (second_split, second_ids) in splits {
                let mut sample_ids = first_ids.clone();
                sample_ids.extend(second_ids.clone());
                findings.push(LeakageFinding {
                    key: key.clone(),
                    value: value.clone(),
                    first_split: *first_split,
                    second_split: *second_split,
                    sample_ids,
                    reason: "Protected key value appears in more than one split.".to_string(),
                });
            }
        }
    }

    LeakageReport {
        schema_ref: "schemas/leakage_report.schema.json".to_string(),
        policy_id: policy.policy_id.clone(),
        checked_keys: policy.protected_keys.clone(),
        findings,
    }
}

pub fn unique_split_protected_values(
    records: &[DatasetRecord],
    policy: &SplitPolicy,
) -> BTreeSet<String> {
    let mut values = BTreeSet::new();
    for record in records {
        for key in &policy.protected_keys {
            if let Some(value) = record.protected_value(key) {
                values.insert(format!("{key}={value}"));
            }
        }
    }
    values
}
