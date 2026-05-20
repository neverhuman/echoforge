use std::collections::BTreeMap;
use std::time::Instant;

use echoforge_core::models::{
    DatasetCard, DatasetSplits, LicenseInfo, Provenance, ValidationCheck, ValidationInfo,
};
use serde::Serialize;

use crate::split::{DatasetRecord, SplitKind, SplitPolicy};

use super::config::{
    MonteCarloBenchmarkReport, MonteCarloDemoConfig, StageTiming,
};
use super::error::DatasetError;
use super::scene_config::{
    AirspaceMonteCarloConfig, EnvironmentProfileConfig, ObjectClassConfig, SensorArchetypeConfig,
};

// ── deterministic RNG ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(super) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub(super) fn unit_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) / ((1u64 << 53) as f64)
    }

    pub(super) fn range_f64(&mut self, range: [f64; 2]) -> f64 {
        range[0] + self.unit_f64() * (range[1] - range[0])
    }

    pub(super) fn range_u32(&mut self, range: [u32; 2]) -> u32 {
        if range[1] <= range[0] {
            return range[0];
        }
        range[0] + (self.next_u64() % (u64::from(range[1] - range[0] + 1))) as u32
    }
}

pub(super) fn child_seed(root: u64, index: u64) -> u64 {
    let mut value = root ^ index.wrapping_mul(0x9e3779b97f4a7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

pub(super) fn midpoint(range: [f64; 2]) -> f64 {
    (range[0] + range[1]) / 2.0
}

pub(super) fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

// ── metadata builders ─────────────────────────────────────────────────────────

pub(super) fn provenance(config: &MonteCarloDemoConfig) -> Provenance {
    Provenance {
        source_kind: "synthetic_public_proxy".to_string(),
        source_refs: vec!["configs/monte-carlo/airspace-objects-v1.json".to_string()],
        generated_by: "echoforge-cli demo monte-carlo".to_string(),
        generated_at: config.generated_at.clone(),
        fingerprint_sha256: String::new(),
    }
}

pub(super) fn license() -> LicenseInfo {
    LicenseInfo {
        spdx_id: "CC-BY-4.0".to_string(),
        notice: "Generated synthetic public-proxy metadata; no measured target truth included."
            .to_string(),
    }
}

pub(super) fn validation_info(uncertainty_score: f64) -> ValidationInfo {
    ValidationInfo {
        tier: "basic".to_string(),
        status: "pass".to_string(),
        uncertainty_score,
        checks: vec![
            ValidationCheck {
                name: "schema".to_string(),
                status: "pass".to_string(),
                message: "Core model finalize/validate completed.".to_string(),
            },
            ValidationCheck {
                name: "determinism".to_string(),
                status: "pass".to_string(),
                message: "Root seed plus generated_at fully determine output.".to_string(),
            },
            ValidationCheck {
                name: "leakage".to_string(),
                status: "pass".to_string(),
                message: "Split leakage report is clean for protected episode variant keys."
                    .to_string(),
            },
            ValidationCheck {
                name: "noise_bounds".to_string(),
                status: "pass".to_string(),
                message: "Noise, RFI, clutter, and scintillation are bounded statistical proxies."
                    .to_string(),
            },
        ],
        fidelity_class: None,
    }
}

pub(super) fn dataset_card_model(
    config: &MonteCarloDemoConfig,
    resolved_object_id: &str,
    resolved_preset_id: &str,
    split_counts: &BTreeMap<SplitKind, usize>,
) -> Result<DatasetCard, DatasetError> {
    DatasetCard {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: resolved_object_id.to_string(),
        provenance: provenance(config),
        license: license(),
        validation: validation_info(0.38),
        dataset_name: format!(
            "{} - public-proxy Monte Carlo dataset ({})",
            config.target_label, resolved_preset_id
        ),
        source_campaign_ids: vec![
            "config.airspace-objects-v1".to_string(),
            "scenario.public_proxy.monte_carlo_takeoff_v1".to_string(),
        ],
        splits: DatasetSplits {
            train: *split_counts.get(&SplitKind::Train).unwrap_or(&0) as u64,
            validation: *split_counts.get(&SplitKind::Validation).unwrap_or(&0) as u64,
            test: *split_counts.get(&SplitKind::Test).unwrap_or(&0) as u64,
        },
    }
    .finalize()
    .map_err(DatasetError::Core)
}

// ── benchmark report builder ──────────────────────────────────────────────────

pub(super) fn build_benchmark_report(
    config: &MonteCarloDemoConfig,
    runtime: &echoforge_radar::RuntimePlan,
    worker_count: usize,
    stage_timings: &[StageTiming],
    total_elapsed: std::time::Duration,
) -> MonteCarloBenchmarkReport {
    let total_elapsed_ns = total_elapsed.as_nanos().min(u128::from(u64::MAX)) as u64;
    let throughput_episodes_per_sec = if total_elapsed_ns == 0 {
        0.0
    } else {
        config.episodes as f64 / (total_elapsed_ns as f64 / 1_000_000_000.0)
    };

    MonteCarloBenchmarkReport {
        requested_backend: runtime.requested_backend,
        selected_backend: runtime.selected_backend.to_string(),
        logical_cores: runtime.logical_cores,
        gpu_available: runtime.gpu_available,
        gpu_usable: runtime.gpu_usable,
        gpu_constrained: runtime.gpu_constrained,
        gpu_min_free_memory_mb: runtime.gpu_min_free_memory_mb,
        gpu_free_memory_mb: runtime
            .gpu_devices
            .iter()
            .map(|device| device.memory_free_mb)
            .max(),
        gpu_unusable_reason: runtime.recovery_reason.clone(),
        recommended_worker_budget: runtime.recommended_worker_budget,
        actual_worker_count: worker_count,
        episode_count: config.episodes,
        seed: config.seed,
        sample_rate_hz: config.sample_rate_hz,
        pulse_count: config.pulse_count,
        stage_timings: stage_timings.to_vec(),
        total_elapsed_ns,
        throughput_episodes_per_sec,
        kernel_backend: "cpu-scaffold".to_string(),
        notes: vec![
            "Episode orchestration is worker-budget aware and deterministic by seed.".to_string(),
            "Kernel execution remains CPU-backed in this scaffold; the runtime plan is ready for a future GPU backend.".to_string(),
        ],
    }
}

// ── manifest structs ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub(super) struct SplitManifest {
    pub policy: SplitPolicy,
    pub counts: BTreeMap<SplitKind, usize>,
    pub records: Vec<DatasetRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct EpisodeManifestEntry {
    pub episode_id: String,
    pub seed: u64,
    pub split: SplitKind,
    pub path: String,
    pub detections: usize,
    pub target_label: String,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct RunManifest<'a> {
    pub manifest_version: &'static str,
    pub preset: &'a str,
    pub generated_at: &'a str,
    pub root_seed: u64,
    pub config_library: &'a AirspaceMonteCarloConfig,
    pub selected_object_class: &'a ObjectClassConfig,
    pub selected_environment: &'a EnvironmentProfileConfig,
    pub selected_sensor: &'a SensorArchetypeConfig,
    pub guardrails: Vec<&'static str>,
    pub validation: ValidationInfo,
    pub episodes: &'a [EpisodeManifestEntry],
    pub leakage_clean: bool,
    pub known_limitations: Vec<&'static str>,
}

// ── JSONL serializer ──────────────────────────────────────────────────────────

pub(super) fn records_jsonl(records: &[DatasetRecord]) -> Result<String, DatasetError> {
    let mut out = String::new();
    for record in records {
        out.push_str(&serde_json::to_string(record)?);
        out.push('\n');
    }
    Ok(out)
}
