//! Private helper functions for `backend.rs`, extracted for LOC compliance.

use std::env;
use std::path::Path;
use std::process::Command;

use std::thread;

use super::{BackendSignals, CpuBackend, GpuDeviceStatus};

pub fn cpu_worker_budget(logical_cores: usize) -> usize {
    ((logical_cores as f64) * 0.30).floor().max(1.0) as usize
}

pub fn gpu_worker_budget(logical_cores: usize, gpu_constrained: bool) -> usize {
    let fraction = if gpu_constrained { 0.10 } else { 0.15 };
    let budget = ((logical_cores as f64) * fraction).floor().max(1.0) as usize;
    budget.min(cpu_worker_budget(logical_cores))
}

pub(super) fn env_override_bool(name: &str) -> Option<bool> {
    let value = env::var(name).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

pub(super) fn env_override_usize(name: &str) -> Option<usize> {
    let value = env::var(name).ok()?;
    value.trim().parse().ok()
}

pub(super) fn env_override_u64(name: &str) -> Option<u64> {
    let value = env::var(name).ok()?;
    value.trim().parse().ok()
}

pub(super) fn default_gpu_min_free_memory_mb() -> u64 {
    2_048
}

pub(super) fn device_nodes() -> Vec<String> {
    ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"]
        .into_iter()
        .filter(|path| Path::new(path).exists())
        .map(|path| path.to_string())
        .collect()
}

pub(super) fn query_nvidia_smi_devices() -> Vec<GpuDeviceStatus> {
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

pub(super) fn is_device_masked(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        None | Some("") => false,
        Some("all") | Some("ALL") | Some("unset") => false,
        Some("void") | Some("none") | Some("NoDevFiles") => true,
        Some(_) => false,
    }
}

pub(super) fn is_device_restricted(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        None | Some("") => false,
        Some("all") | Some("ALL") | Some("unset") => false,
        Some("void") | Some("none") | Some("NoDevFiles") => true,
        Some(_) => true,
    }
}

/// GPU backend stub — no real CUDA/ROCm/Metal wired in this scaffold.
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

/// Implementation body for [`BackendSignals::detect`], extracted for LOC compliance.
pub(super) fn detect_backend_signals() -> BackendSignals {
    let logical_cores = if let Some(n) = env_override_usize("ECHOFORGE_RUNTIME_LOGICAL_CORES") {
        n
    } else {
        thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
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

    BackendSignals {
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
