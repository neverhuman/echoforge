use std::collections::BTreeMap;
use std::path::PathBuf;

use echoforge_radar::{BackendMode, RuntimePlan};
use serde::{Deserialize, Serialize};

use crate::split::SplitKind;

/// Neutral default target label used by the monte-carlo demo.
///
/// Country-specific or platform-specific aliases live in
/// `object-packs/public-proxy-v1/source_dossier.yaml`
/// (`policy: source_dossier_only`). The default label never names a single
/// nation or system so the strict-open dataset stays publishable as-is.
pub const DEFAULT_TARGET_LABEL: &str = "Low-Altitude Fixed-Wing UAV Takeoff (public proxy)";
pub const DEFAULT_PRESET: &str = "low-altitude-fixed-wing-takeoff-v1";
pub(super) const DEFAULT_NOISE_PROFILE: &str = "real-world-proxy-v1";

#[derive(Debug, Clone)]
pub struct MonteCarloDemoConfig {
    pub preset: String,
    pub episodes: usize,
    pub seed: u64,
    pub generated_at: String,
    pub output_dir: PathBuf,
    pub target_label: String,
    pub noise_profile: String,
    pub sample_rate_hz: f64,
    pub pulse_count: usize,
    pub runtime: MonteCarloRuntimePolicy,
}

impl MonteCarloDemoConfig {
    /// Neutral default constructor — public-proxy low-altitude fixed-wing
    /// takeoff. Use this for new code paths; the demo CLI defaults to the
    /// same preset.
    pub fn low_altitude_fixed_wing_default(output_dir: PathBuf) -> Self {
        Self {
            preset: DEFAULT_PRESET.to_string(),
            episodes: 32,
            seed: 20_260_518,
            generated_at: "2026-05-18T00:00:00Z".to_string(),
            output_dir,
            target_label: DEFAULT_TARGET_LABEL.to_string(),
            noise_profile: DEFAULT_NOISE_PROFILE.to_string(),
            sample_rate_hz: 2_000_000.0,
            pulse_count: 32,
            runtime: MonteCarloRuntimePolicy::default(),
        }
    }

    /// Superseded alias retained for one release. Delegates to
    /// [`Self::low_altitude_fixed_wing_default`] so external callers do not
    /// silently break.
    pub fn iranian_takeoff_default(output_dir: PathBuf) -> Self {
        Self::low_altitude_fixed_wing_default(output_dir)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MonteCarloRuntimePolicy {
    pub backend: BackendMode,
    pub benchmark: bool,
}

impl Default for MonteCarloRuntimePolicy {
    fn default() -> Self {
        Self {
            backend: BackendMode::Auto,
            benchmark: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MonteCarloDemoReport {
    pub output_dir: PathBuf,
    pub preset: String,
    pub episode_count: usize,
    pub split_counts: BTreeMap<SplitKind, usize>,
    pub leakage_clean: bool,
    pub validation_status: String,
    pub manifest_path: PathBuf,
    pub dataset_card_path: PathBuf,
    pub runtime: RuntimePlan,
    pub benchmark_report_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonteCarloBenchmarkReport {
    pub requested_backend: BackendMode,
    pub selected_backend: String,
    pub logical_cores: usize,
    pub gpu_available: bool,
    pub gpu_usable: bool,
    pub gpu_constrained: bool,
    pub gpu_min_free_memory_mb: u64,
    pub gpu_free_memory_mb: Option<u64>,
    pub gpu_unusable_reason: Option<String>,
    pub recommended_worker_budget: usize,
    pub actual_worker_count: usize,
    pub episode_count: usize,
    pub seed: u64,
    pub sample_rate_hz: f64,
    pub pulse_count: usize,
    pub stage_timings: Vec<StageTiming>,
    pub total_elapsed_ns: u64,
    pub throughput_episodes_per_sec: f64,
    pub kernel_backend: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageTiming {
    pub stage: String,
    pub elapsed_ns: u64,
}
