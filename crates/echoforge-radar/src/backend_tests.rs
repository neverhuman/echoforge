use super::*;

#[test]
fn cpu_budget_uses_thirty_percent_floor() {
    assert_eq!(cpu_worker_budget(128), 38);
    assert_eq!(cpu_worker_budget(1), 1);
}

#[test]
fn gpu_budget_reduces_host_workers_when_constrained() {
    assert_eq!(gpu_worker_budget(128, false), 19);
    assert_eq!(gpu_worker_budget(128, true), 12);
}

#[test]
fn cpu_only_signals_select_cpu_and_keep_budget() {
    let plan =
        RuntimePlan::from_signals(BackendMode::Cpu, BackendSignals::new(128, false, false))
            .expect("cpu plan");

    assert_eq!(plan.selected_backend, RuntimeBackend::Cpu);
    assert_eq!(plan.recommended_worker_budget, 38);
    assert!(plan.recovery_reason.is_none());
}

#[test]
fn auto_backend_prefers_gpu_when_available() {
    let plan =
        RuntimePlan::from_signals(BackendMode::Auto, BackendSignals::new(64, true, true))
            .expect("auto plan");

    assert_eq!(plan.selected_backend, RuntimeBackend::Gpu);
    assert!(plan.recommended_worker_budget <= cpu_worker_budget(64));
    assert!(plan
        .recovery_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("constraints")));
}

#[test]
fn forced_gpu_without_device_fails_cleanly() {
    let err =
        RuntimePlan::from_signals(BackendMode::Gpu, BackendSignals::new(32, false, false))
            .expect_err("gpu selection should fail without a detected device");

    assert!(matches!(
        err,
        BackendSelectionError::GpuUnavailable {
            requested_backend: BackendMode::Gpu,
            logical_cores: 32,
            ..
        }
    ));
}

#[test]
fn auto_backend_falls_back_when_gpu_memory_is_too_low() {
    let mut signals = BackendSignals::new(128, true, true);
    signals.gpu_devices = vec![GpuDeviceStatus {
        index: 0,
        name: "NVIDIA GeForce RTX 3090".to_string(),
        memory_total_mb: 24_576,
        memory_used_mb: 23_528,
        memory_free_mb: 606,
        utilization_gpu_percent: 100,
        compute_mode: "Default".to_string(),
    }];
    signals.gpu_min_free_memory_mb = 2_048;
    signals.gpu_usable = false;
    signals.gpu_unusable_reason = Some(
        "GPU detected but free memory is 606 MiB; need at least 2048 MiB for this runtime"
            .to_string(),
    );

    let plan = RuntimePlan::from_signals(BackendMode::Auto, signals).expect("auto plan");
    assert_eq!(plan.selected_backend, RuntimeBackend::Cpu);
    assert!(plan
        .recovery_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("free memory is 606 MiB")));
}

#[test]
fn forced_gpu_fails_when_gpu_memory_is_too_low() {
    let mut signals = BackendSignals::new(128, true, true);
    signals.gpu_usable = false;
    signals.gpu_unusable_reason = Some(
        "GPU detected but free memory is 606 MiB; need at least 2048 MiB for this runtime"
            .to_string(),
    );
    let err = RuntimePlan::from_signals(BackendMode::Gpu, signals)
        .expect_err("forced GPU should fail when readiness gate fails");
    assert!(err.to_string().contains("free memory is 606 MiB"));
}
