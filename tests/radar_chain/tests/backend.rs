use echoforge_radar::{
    ca_cfar_1d, ca_cfar_scale, magnitude, pulse_compress, ArrayBackend, CfarParams, CpuBackend,
    GpuBackendUnavailable, LfmChirp,
};

#[test]
fn cpu_backend_matches_free_functions() {
    let backend = CpuBackend;
    assert_eq!(backend.name(), "cpu");

    let chirp = LfmChirp {
        sample_rate_hz: 10_000.0,
        pulse_width_s: 0.001,
        bandwidth_hz: 1_000.0,
        carrier_hz: 0.0,
        initial_phase_rad: 0.0,
    };
    let reference = backend.chirp(&chirp);
    let compressed = backend.pulse_compress(&reference, &reference);
    let cfar_params = CfarParams::new(4, 2, 1e-3);
    let power = magnitude(&compressed);
    let backend_cfar = backend.ca_cfar_1d(&power, cfar_params);
    let direct_cfar = ca_cfar_1d(&power, cfar_params);

    assert_eq!(backend_cfar, direct_cfar);
    assert!(ca_cfar_scale(4, 1e-3) > 0.0);
}

#[test]
fn gpu_backend_is_explicitly_unavailable_and_has_a_cpu_fallback() {
    let backend = GpuBackendUnavailable::default();

    assert!(!backend.is_available());
    assert_eq!(backend.name(), "gpu-unavailable");
    assert!(backend.reason().contains("CPU recovery"));

    let cpu = backend.cpu_recovery();
    assert_eq!(cpu.name(), "cpu");

    let chirp = LfmChirp {
        sample_rate_hz: 10_000.0,
        pulse_width_s: 0.001,
        bandwidth_hz: 1_000.0,
        carrier_hz: 0.0,
        initial_phase_rad: 0.0,
    };
    let reference = cpu.chirp(&chirp);
    let compressed = cpu.pulse_compress(&reference, &reference);
    let power = magnitude(&compressed);
    let decisions = cpu.ca_cfar_1d(&power, CfarParams::new(4, 2, 1e-3));

    assert!(decisions.iter().any(|decision| decision.evaluated));
}
