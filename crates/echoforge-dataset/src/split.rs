use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitKind {
    Train,
    Validation,
    Test,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitRatios {
    pub train_bps: u16,
    pub validation_bps: u16,
    pub test_bps: u16,
}

impl SplitRatios {
    pub fn validate(&self) -> bool {
        self.train_bps + self.validation_bps + self.test_bps == 10_000
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitPolicy {
    pub schema_ref: String,
    pub policy_id: String,
    pub display_name: String,
    pub ratios: SplitRatios,
    pub protected_keys: Vec<String>,
    pub grouping_mode: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetRecord {
    pub sample_id: String,
    pub split_hint: Option<SplitKind>,
    pub object_family: String,
    pub geometry_hash: String,
    pub material_sample_hash: String,
    pub scenario_seed: u64,
    pub sensor_archetype: String,
    pub hard_negative_family: String,
}

impl DatasetRecord {
    pub fn protected_value(&self, key: &str) -> Option<String> {
        match key {
            "object_family" => Some(self.object_family.clone()),
            "geometry_hash" => Some(self.geometry_hash.clone()),
            "material_sample_hash" => Some(self.material_sample_hash.clone()),
            "scenario_seed" => Some(self.scenario_seed.to_string()),
            "sensor_archetype" => Some(self.sensor_archetype.clone()),
            "hard_negative_family" => Some(self.hard_negative_family.clone()),
            _ => None,
        }
    }

    pub fn split_key(&self, policy: &SplitPolicy) -> String {
        policy
            .protected_keys
            .iter()
            .filter_map(|key| self.protected_value(key).map(|value| format!("{key}={value}")))
            .collect::<Vec<_>>()
            .join("|")
    }
}

pub fn default_public_proxy_split_policy() -> SplitPolicy {
    SplitPolicy {
        schema_ref: "schemas/dataset_card.schema.json".to_string(),
        policy_id: "dataset.split.public_proxy_v1".to_string(),
        display_name: "Public Proxy Split Policy v1".to_string(),
        ratios: SplitRatios {
            train_bps: 7_000,
            validation_bps: 1_500,
            test_bps: 1_500,
        },
        protected_keys: vec![
            "object_family".to_string(),
            "geometry_hash".to_string(),
            "material_sample_hash".to_string(),
            "scenario_seed".to_string(),
            "sensor_archetype".to_string(),
            "hard_negative_family".to_string(),
        ],
        grouping_mode: "hash_protected_keys".to_string(),
        notes: vec![
            "No object family, geometry hash, material sample, scenario seed, sensor archetype, or hard-negative family may straddle splits.".to_string(),
            "Split assignment uses a deterministic hash over the protected key tuple.".to_string(),
        ],
    }
}

pub fn assign_split(record: &DatasetRecord, policy: &SplitPolicy) -> SplitKind {
    assert!(policy.ratios.validate(), "split ratios must sum to 10000 basis points");
    let bucket = deterministic_hash(&record.split_key(policy)) % 10_000;
    if bucket < policy.ratios.train_bps as u64 {
        SplitKind::Train
    } else if bucket < (policy.ratios.train_bps + policy.ratios.validation_bps) as u64 {
        SplitKind::Validation
    } else {
        SplitKind::Test
    }
}

pub fn deterministic_hash(input: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
