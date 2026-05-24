use std::fs;
use std::path::Path;

use serde::Serialize;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

#[derive(Debug, Clone, Serialize)]
pub struct GpuDoctorCheck {
    pub name: String,
    pub status: String,
    pub details: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuDoctorEnvironment {
    pub nvidia_visible_devices: String,
    pub nvidia_driver_capabilities: String,
    pub cuda_visible_devices: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuDoctorReport {
    pub kind: String,
    pub status: String,
    pub mode: String,
    pub timestamp: String,
    pub environment: GpuDoctorEnvironment,
    pub checks: Vec<GpuDoctorCheck>,
    pub notes: Vec<String>,
}

fn device_nodes() -> Vec<String> {
    ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"]
        .into_iter()
        .filter(|path| Path::new(path).exists())
        .map(|path| path.to_string())
        .collect()
}

pub fn build_gpu_doctor_report() -> GpuDoctorReport {
    let device_nodes = device_nodes();
    GpuDoctorReport {
        kind: "echoforge.gpu_doctor".to_string(),
        status: "placeholder".to_string(),
        mode: "scaffold".to_string(),
        timestamp: OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string()),
        environment: GpuDoctorEnvironment {
            nvidia_visible_devices: std::env::var("NVIDIA_VISIBLE_DEVICES")
                .unwrap_or_else(|_| "unset".to_string()),
            nvidia_driver_capabilities: std::env::var("NVIDIA_DRIVER_CAPABILITIES")
                .unwrap_or_else(|_| "unset".to_string()),
            cuda_visible_devices: std::env::var("CUDA_VISIBLE_DEVICES")
                .unwrap_or_else(|_| "unset".to_string()),
        },
        checks: vec![
            GpuDoctorCheck {
                name: "nvidia-device-nodes".to_string(),
                status: if device_nodes.is_empty() {
                    "missing".to_string()
                } else {
                    "present".to_string()
                },
                details: serde_json::json!(device_nodes),
            },
            GpuDoctorCheck {
                name: "nvidia-smi".to_string(),
                status: "not-run".to_string(),
                details: serde_json::json!(
                    "This is a scaffolded doctor stub, not a real validation claim."
                ),
            },
            GpuDoctorCheck {
                name: "cupy-allocation".to_string(),
                status: "not-run".to_string(),
                details: serde_json::json!("Deferred until the real GPU runtime is wired in."),
            },
        ],
        notes: vec![
            "This report is explicitly a placeholder.".to_string(),
            "It records environment signals without claiming hardware validation.".to_string(),
        ],
    }
}

pub fn write_report(path: &Path) -> Result<(), std::io::Error> {
    let payload = serde_json::to_string_pretty(&build_gpu_doctor_report())
        .expect("serialize gpu doctor report");
    fs::write(path, payload.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_is_placeholder_only() {
        let report = build_gpu_doctor_report();
        assert_eq!(report.kind, "echoforge.gpu_doctor");
        assert_eq!(report.status, "placeholder");
        assert_eq!(report.mode, "scaffold");
        assert!(report.notes.iter().any(|note| note.contains("placeholder")));
    }
}
