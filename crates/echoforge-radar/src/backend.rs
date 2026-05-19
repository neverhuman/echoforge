use std::env;
use std::fmt;
use std::path::Path;
use std::process::Command;
use std::thread;

use serde::{Deserialize, Serialize};

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
        let logical_cores = if let Some(n) = env_override_usize("ECHOFORGE_RUNTIME_LOGICAL_CORES") {
            n
        } else {
            thread::available_parallelism().map(usize::from).unwrap_or(1)
        }
        .max(1);
        let device_nodes = device_nodes();
        let nvidia_visible_devices = env::var("NVIDIA_VISIBLE_DEVICES").ok();
        let nvidia_driver_capabilities = env::var("NVIDIA_DRIVER_CAPABILITIES").ok();
        let cuda_visible_devices = env::var("CUDA_VISIBLE_DEVICES").ok();

        let gpu_devices = query_nvidia_smi_devices();
        let gpu_min_free_memory_mb = match env_override_u64("ECHOFORGE_RUNTIME_GPU_MIN_FREE_MB") {
            Some(n) => n,
            None => default_gpu_min_free_memory_mb(),
        };
        let gpu_available = match env_override_bool("ECHOFORGE_RUNTIME_GPU_AVAILABLE") {
            Some(b) => b,
            None => !device_nodes.is_empty() || !gpu_devices.is_empty(),
        };
        let masked = is_device_masked(nvidia_visible_devices.as_deref())
            || is_device_masked(cuda_visible_devices.as_deref());
        let best_free_mb = gpu_devices
            .iter()
            .map(|device| device.memory_free_mb)
            .max()
            .unwrap_or(0);
        let memory_ready = gpu_devices.is_empty() || best_free_mb >= gpu_min_free_memory_mb;
        let gpu_unusable_reason = if !gpu_available {
            Some("no NVIDIA GPU device node or nvidia-smi device was detected".to_string())
        } else if masked {
            Some(
                "GPU visibility is masked by NVIDIA_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES"
                    .to_string(),
            )
        } else if !memory_ready {
            Some(format!(
                "GPU detected but free memory is {best_free_mb} MiB; need at least {gpu_min_free_memory_mb} MiB for this runtime"
            ))
        } else {
            None
        };
        let gpu_usable = match env_override_bool("ECHOFORGE_RUNTIME_GPU_USABLE") {
            Some(b) => b,
            None => gpu_available && gpu_unusable_reason.is_none(),
        };
        let gpu_constrained = match env_override_bool("ECHOFORGE_RUNTIME_GPU_CONSTRAINED") {
            Some(b) => b,
            None => {
                gpu_available
                    && (device_nodes.len() <= 1
                        || is_device_restricted(nvidia_visible_devices.as_deref())
                        || is_device_restricted(cuda_visible_devices.as_deref())
                        || best_free_mb < gpu_min_free_memory_mb.saturating_mul(2)
                        || gpu_devices
                            .iter()
                            .any(|device| device.utilization_gpu_percent >= 85))
            }
        };

        Self {
            logical_cores,
            device_nodes,
            nvidia_visible_devices,
            gpu_devices,
            nvidia_driver_capabilities,
            cuda_visible_devices,
            gpu_available,
            gpu_usable,
            gpu_constrained,
            gpu_min_free_memory_mb,
            gpu_unusable_reason: gpu_unusable_reason.filter(|_| !gpu_usable),
        }
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

pub fn cpu_worker_budget(logical_cores: usize) -> usize {
    ((logical_cores as f64) * 0.30).floor().max(1.0) as usize
}

pub fn gpu_worker_budget(logical_cores: usize, gpu_constrained: bool) -> usize {
    let fraction = if gpu_constrained { 0.10 } else { 0.15 };
    let budget = ((logical_cores as f64) * fraction).floor().max(1.0) as usize;
    budget.min(cpu_worker_budget(logical_cores))
}

fn env_override_bool(name: &str) -> Option<bool> {
    let value = env::var(name).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn env_override_usize(name: &str) -> Option<usize> {
    let value = env::var(name).ok()?;
    value.trim().parse().ok()
}

fn env_override_u64(name: &str) -> Option<u64> {
    let value = env::var(name).ok()?;
    value.trim().parse().ok()
}

fn default_gpu_min_free_memory_mb() -> u64 {
    2_048
}

fn device_nodes() -> Vec<String> {
    ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"]
        .into_iter()
        .filter(|path| Path::new(path).exists())
        .map(|path| path.to_string())
        .collect()
}

fn query_nvidia_smi_devices() -> Vec<GpuDeviceStatus> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };

    if !output.status.success() {
        return Vec::new();
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(parse_nvidia_smi_device_line)
        .collect()
}

fn parse_nvidia_smi_device_line(line: &str) -> Option<GpuDeviceStatus> {
    let parts = line.split(',').map(|part| part.trim()).collect::<Vec<_>>();
    if parts.len() != 7 {
        return None;
    }

    Some(GpuDeviceStatus {
        index: parts[0].parse().ok()?,
        name: parts[1].to_string(),
        memory_total_mb: parts[2].parse().ok()?,
        memory_used_mb: parts[3].parse().ok()?,
        memory_free_mb: parts[4].parse().ok()?,
        utilization_gpu_percent: parts[5].parse().ok()?,
        compute_mode: parts[6].to_string(),
    })
}

fn is_device_masked(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        None | Some("") => false,
        Some("all") | Some("ALL") | Some("unset") => false,
        Some("void") | Some("none") | Some("NoDevFiles") => true,
        Some(_) => false,
    }
}

fn is_device_restricted(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        None | Some("") => false,
        Some("all") | Some("ALL") | Some("unset") => false,
        Some("void") | Some("none") | Some("NoDevFiles") => true,
        Some(_) => true,
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GpuBackendUnavailable;

impl GpuBackendUnavailable {
    pub fn is_available(&self) -> bool {
        false
    }

    pub fn name(&self) -> &'static str {
        "gpu-unavailable"
    }

    pub fn reason(&self) -> &'static str {
        "GPU backend is not wired into this scaffold; use the CPU recovery path explicitly."
    }

    pub fn cpu_recovery(&self) -> CpuBackend {
        CpuBackend
    }
}

#[cfg(test)]
#[path = "backend_tests.rs"]
mod tests;
