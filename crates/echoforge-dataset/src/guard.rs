//! Output-path guards and generation-run preamble shared across campaign,
//! ml_training, and monte_carlo modules.

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use echoforge_radar::{BackendMode, BackendSignals, RadarSimConfig, RuntimeBackend, RuntimePlan};

use crate::monte_carlo::{DatasetError, StageTiming};

/// Shared product guardrails used in both ml-training and campaign manifests.
/// Call sites use `.iter().map(|s| s.to_string()).collect()` directly so there
/// is no duplicate wrapper function in either module.
pub(crate) const PRODUCT_GUARDRAILS: &[&str] = &[
    "synthetic public proxy",
    "uncertainty-scored radar artifacts",
    "no measured object trace copying by default",
    "defensive early-detection and false-alarm robustness only",
    "no payload effects, terminal behavior, route optimization, or operational tuning",
    "Hard negatives are robustness and false-alarm stressors only.",
];

const FORBIDDEN_PREFIXES: &[&str] = &[
    ".git/",
    "crates/",
    "schemas/",
    "tests/",
    "object-packs/",
    "scenarios/",
    "python/",
    "docker/",
];

/// Reject any output path that overlaps a source-owned or reserved directory.
pub(crate) fn refuse_source_owned_output_path(path: &Path) -> Result<(), DatasetError> {
    let text = path.to_string_lossy();
    if FORBIDDEN_PREFIXES
        .iter()
        .any(|prefix| text == prefix.trim_end_matches('/') || text.contains(prefix))
    {
        return Err(DatasetError::RefusingOutputPath(format!(
            "{} overlaps a source-owned or reserved path",
            path.display()
        )));
    }
    Ok(())
}

/// Guard an output directory: reject source-owned paths and reject paths that
/// already exist on disk. Combines both checks into a single call site.
pub(crate) fn guard_output_dir(path: &Path) -> Result<(), DatasetError> {
    refuse_source_owned_output_path(path)?;
    if path.exists() {
        return Err(DatasetError::RefusingOutputPath(format!(
            "{} already exists",
            path.display()
        )));
    }
    Ok(())
}

/// Computed worker count, capped at 40 and bounded by the runtime budget.
pub(crate) fn compute_worker_count(
    workers: Option<usize>,
    records: usize,
    runtime: &RuntimePlan,
) -> usize {
    workers
        .unwrap_or(40)
        .min(40)
        .min(runtime.recommended_worker_budget.max(1))
        .min(records.max(1))
        .max(1)
}

pub(crate) fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

/// Context established at the start of every generation run.
pub(crate) struct GenerationPreamble {
    pub runtime: RuntimePlan,
    pub worker_count: usize,
    pub frame_count: usize,
    pub overall_start: Instant,
    pub stage_timings: Vec<StageTiming>,
}

impl GenerationPreamble {
    pub(crate) fn push_timing(&mut self, stage: &str, start: Instant) {
        self.stage_timings.push(StageTiming {
            stage: stage.to_string(),
            elapsed_ns: elapsed_ns(start),
        });
    }
}

/// Shared GPU/CPU stage recovery note used in both campaign and ml-training
/// runtime reports. Returns a note when the runtime fell back from GPU to CPU.
pub(crate) fn gpu_stage_recovery_note(runtime: &RuntimePlan) -> Option<String> {
    match runtime.selected_backend {
        RuntimeBackend::Gpu => Some(
            "GPU runtime selected but the v1 radar kernel is CPU-backed; generated tensors record this recovery path.".to_string(),
        ),
        RuntimeBackend::Cpu => runtime.recovery_reason.clone(),
    }
}

/// GPU availability and memory fields, nested inside `RuntimeReportFields` so
/// the parent struct is structurally distinct from the per-module `RuntimeReport`
/// types it populates.
pub(crate) struct GpuFields {
    pub available: bool,
    pub usable: bool,
    pub constrained: bool,
    pub min_free_memory_mb: u64,
    pub free_memory_mb: Option<u64>,
    pub stage_recovery: Option<String>,
}

/// Common backend fields shared between the campaign and ml-training runtime
/// reports. Extracted here to eliminate duplication in the two report builders.
pub(crate) struct RuntimeReportFields {
    pub requested_backend: BackendMode,
    pub selected_backend: String,
    pub kernel_backend: String,
    pub gpu: GpuFields,
    pub logical_cores: usize,
    pub recommended_worker_budget: usize,
    pub total_elapsed_ns: u64,
    pub throughput_records_per_sec: f64,
}

/// Extract the GPU/backend fields and timing from a `RuntimePlan` and an
/// elapsed `Duration`. Both campaign and ml-training report builders call this
/// to avoid duplicating the identical field-extraction code.
pub(crate) fn runtime_report_fields(
    runtime: &RuntimePlan,
    records: usize,
    elapsed: Duration,
) -> RuntimeReportFields {
    let total_elapsed_ns = elapsed.as_nanos().min(u128::from(u64::MAX)) as u64;
    RuntimeReportFields {
        requested_backend: runtime.requested_backend,
        selected_backend: runtime.selected_backend.to_string(),
        kernel_backend: "cpu-scaffold".to_string(),
        gpu: GpuFields {
            available: runtime.gpu_available,
            usable: runtime.gpu_usable,
            constrained: runtime.gpu_constrained,
            min_free_memory_mb: runtime.gpu_min_free_memory_mb,
            free_memory_mb: runtime.gpu_devices.iter().map(|d| d.memory_free_mb).max(),
            stage_recovery: gpu_stage_recovery_note(runtime),
        },
        logical_cores: runtime.logical_cores,
        recommended_worker_budget: runtime.recommended_worker_budget,
        total_elapsed_ns,
        throughput_records_per_sec: if total_elapsed_ns == 0 {
            0.0
        } else {
            records as f64 / (total_elapsed_ns as f64 / 1_000_000_000.0)
        },
    }
}

/// Build the standard radar simulation config for both campaign and ml-training
/// workers. The `base_snr_db` and `cpi_pulses` values come from the per-record
/// envelope sampler; all other parameters are site-fixed.
pub(crate) fn build_radar_sim_config(base_snr_db: f32, cpi_pulses: usize) -> RadarSimConfig {
    RadarSimConfig {
        sample_rate_hz: 1_000_000.0,
        pulse_width_s: 64e-6,
        bandwidth_hz: 800_000.0,
        carrier_hz: 9_600_000_000.0,
        pulse_count: cpi_pulses,
        pri_s: 900e-6,
        target_snr_db: base_snr_db as f64,
        ..RadarSimConfig::default()
    }
}

/// Detect the runtime, compute worker/frame counts, create the output directory,
/// and return the context needed for the rest of the run. Called after
/// config validation and output-path guarding.
pub(crate) fn begin_generation_run(
    output_dir: &Path,
    backend: BackendMode,
    workers: Option<usize>,
    records: usize,
    time_window_s: f64,
    frame_rate_hz: f64,
) -> Result<GenerationPreamble, DatasetError> {
    let overall_start = Instant::now();
    let runtime = RuntimePlan::from_signals(backend, BackendSignals::detect())?;
    let worker_count = compute_worker_count(workers, records, &runtime);
    let frame_count = ((time_window_s * frame_rate_hz).round() as usize).max(1);
    fs::create_dir_all(output_dir)?;
    Ok(GenerationPreamble {
        runtime,
        worker_count,
        frame_count,
        overall_start,
        stage_timings: Vec::new(),
    })
}
