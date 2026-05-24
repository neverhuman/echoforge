use std::fmt;

use serde::{Deserialize, Serialize};

#[path = "backend_helpers.rs"]
mod backend_helpers;
pub use backend_helpers::GpuBackendUnavailable;
pub use backend_helpers::{cpu_worker_budget, gpu_worker_budget};
use backend_helpers::{default_gpu_min_free_memory_mb, detect_backend_signals};

use crate::cfar::{ca_cfar_1d, CfarDecision, CfarParams};
use crate::pulse_compression::matched_filter;
use crate::waveform::{lfm_chirp, LfmChirp};
use crate::ComplexSample;

pub trait ArrayBackend {
    fn name(&self) -> &'static str;
    fn chirp(&self, config: &LfmChirp) -> Vec<ComplexSample>;
    fn pulse_compress(
        &self,
        received: &[ComplexSample],
        reference: &[ComplexSample],
    ) -> Vec<ComplexSample>;
    fn ca_cfar_1d(&self, power: &[f32], params: CfarParams) -> Vec<CfarDecision>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl ArrayBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn chirp(&self, config: &LfmChirp) -> Vec<ComplexSample> {
        lfm_chirp(config)
    }

    fn pulse_compress(
        &self,
        received: &[ComplexSample],
        reference: &[ComplexSample],
    ) -> Vec<ComplexSample> {
        matched_filter(received, reference)
    }

    fn ca_cfar_1d(&self, power: &[f32], params: CfarParams) -> Vec<CfarDecision> {
        ca_cfar_1d(power, params)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendMode {
    Auto,
    Cpu,
    Gpu,
}

impl BackendMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

impl fmt::Display for BackendMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeBackend {
    Cpu,
    Gpu,
}

impl RuntimeBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

impl fmt::Display for RuntimeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuDeviceStatus {
    pub index: usize,
    pub name: String,
    pub memory_total_mb: u64,
    pub memory_used_mb: u64,
    pub memory_free_mb: u64,
    pub utilization_gpu_percent: u32,
    pub compute_mode: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendSignals {
    pub logical_cores: usize,
    pub device_nodes: Vec<String>,
    pub gpu_devices: Vec<GpuDeviceStatus>,
    pub nvidia_visible_devices: Option<String>,
    pub nvidia_driver_capabilities: Option<String>,
    pub cuda_visible_devices: Option<String>,
    pub gpu_available: bool,
    pub gpu_usable: bool,
    pub gpu_constrained: bool,
    pub gpu_min_free_memory_mb: u64,
    pub gpu_unusable_reason: Option<String>,
}

impl BackendSignals {
    pub fn new(logical_cores: usize, gpu_available: bool, gpu_constrained: bool) -> Self {
        Self {
            logical_cores: logical_cores.max(1),
            device_nodes: Vec::new(),
            gpu_devices: Vec::new(),
            nvidia_visible_devices: None,
            nvidia_driver_capabilities: None,
            cuda_visible_devices: None,
            gpu_available,
            gpu_usable: gpu_available,
            gpu_constrained,
            gpu_min_free_memory_mb: default_gpu_min_free_memory_mb(),
            gpu_unusable_reason: (!gpu_available)
                .then(|| "no GPU reported by test signals".to_string()),
        }
    }

    pub fn detect() -> Self {
        detect_backend_signals()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlan {
    pub requested_backend: BackendMode,
    pub selected_backend: RuntimeBackend,
    pub logical_cores: usize,
    pub device_nodes: Vec<String>,
    pub nvidia_visible_devices: Option<String>,
    pub nvidia_driver_capabilities: Option<String>,
    pub gpu_available: bool,
    pub cuda_visible_devices: Option<String>,
    pub gpu_usable: bool,
    pub gpu_constrained: bool,
    pub gpu_min_free_memory_mb: u64,
    pub gpu_devices: Vec<GpuDeviceStatus>,
    pub recommended_worker_budget: usize,
    pub recovery_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendSelectionError {
    GpuUnavailable {
        requested_backend: BackendMode,
        logical_cores: usize,
        reason: String,
    },
}

impl fmt::Display for BackendSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GpuUnavailable {
                requested_backend,
                logical_cores,
                reason,
            } => write!(
                f,
                "requested backend {requested_backend} but no usable GPU runtime was detected on {logical_cores} logical core(s): {reason}"
            ),
        }
    }
}

impl std::error::Error for BackendSelectionError {}

impl RuntimePlan {
    pub fn detect(requested_backend: BackendMode) -> Result<Self, BackendSelectionError> {
        Self::from_signals(requested_backend, BackendSignals::detect())
    }

    pub fn from_signals(
        requested_backend: BackendMode,
        signals: BackendSignals,
    ) -> Result<Self, BackendSelectionError> {
        let cpu_budget = cpu_worker_budget(signals.logical_cores);
        let gpu_budget = gpu_worker_budget(signals.logical_cores, signals.gpu_constrained);

        let BackendSignals {
            logical_cores,
            device_nodes,
            gpu_devices,
            nvidia_visible_devices,
            nvidia_driver_capabilities,
            cuda_visible_devices,
            gpu_available,
            gpu_usable,
            gpu_constrained,
            gpu_min_free_memory_mb,
            gpu_unusable_reason,
        } = signals;

        let (selected_backend, recommended_worker_budget, recovery_reason) = match requested_backend
        {
            BackendMode::Auto => {
                if gpu_usable {
                    let note = gpu_constrained.then(|| {
                        "GPU probe reported constraints; reducing host worker budget".to_string()
                    });
                    (RuntimeBackend::Gpu, gpu_budget, note)
                } else {
                    let reason = Some(if gpu_available {
                        format!(
                            "GPU probe detected hardware but it is not currently usable: {}; using CPU worker budget",
                            gpu_unusable_reason.as_deref().unwrap_or("unknown GPU readiness failure")
                        )
                    } else {
                        "GPU probe unavailable; using CPU worker budget".to_string()
                    });
                    (RuntimeBackend::Cpu, cpu_budget, reason)
                }
            }
            BackendMode::Cpu => (RuntimeBackend::Cpu, cpu_budget, None),
            BackendMode::Gpu => {
                if !gpu_usable {
                    return Err(BackendSelectionError::GpuUnavailable {
                        requested_backend,
                        logical_cores,
                        reason: match gpu_unusable_reason {
                            Some(r) => r,
                            None => "GPU is unavailable or failed readiness checks".to_string(),
                        },
                    });
                }
                let note = gpu_constrained.then(|| {
                    "GPU probe reported constraints; reducing host worker budget".to_string()
                });
                (RuntimeBackend::Gpu, gpu_budget, note)
            }
        };

        Ok(Self {
            requested_backend,
            selected_backend,
            logical_cores,
            device_nodes,
            nvidia_visible_devices,
            nvidia_driver_capabilities,
            cuda_visible_devices,
            gpu_available,
            gpu_usable,
            gpu_constrained,
            gpu_min_free_memory_mb,
            gpu_devices,
            recommended_worker_budget,
            recovery_reason,
        })
    }
}

#[cfg(test)]
#[path = "backend_tests.rs"]
mod tests;
