use echoforge_radar::{
    CfarDecision, CfarParams, CpuBackend, LfmChirp, RadarChain, TrackingFusionAdapter,
};

#[test]
fn tracking_fusion_adapter_assigns_dense_track_ids() {
    let adapter = TrackingFusionAdapter;
    let decisions = vec![
        CfarDecision {
            index: 0,
            evaluated: false,
            detected: false,
            statistic: 1.0,
            threshold: 2.0,
            noise_estimate: 1.0,
        },
        CfarDecision {
            index: 12,
            evaluated: true,
            detected: false,
            statistic: 2.0,
            threshold: 4.0,
            noise_estimate: 1.5,
        },
        CfarDecision {
            index: 31,
            evaluated: true,
            detected: true,
            statistic: 9.0,
            threshold: 3.0,
            noise_estimate: 1.0,
        },
        CfarDecision {
            index: 32,
            evaluated: true,
            detected: true,
            statistic: 12.0,
            threshold: 4.0,
            noise_estimate: 1.0,
        },
    ];

    let report = adapter.fuse_decisions("cpu", &decisions);

    assert_eq!(report.backend_name, "cpu");
    assert_eq!(report.evaluated_cells, 3);
    assert_eq!(report.detections, 2);
    assert_eq!(report.tracks.len(), 2);
    assert_eq!(report.tracks[0].track_id, 0);
    assert_eq!(report.tracks[0].detection_index, 31);
    assert_eq!(report.tracks[0].confidence, 3.0);
    assert_eq!(report.tracks[1].track_id, 1);
    assert_eq!(report.tracks[1].detection_index, 32);
    assert_eq!(report.tracks[1].confidence, 3.0);
}

#[test]
fn tracking_fusion_adapter_wraps_radar_chain_output() {
    let adapter = TrackingFusionAdapter;
    let backend = CpuBackend;
    let chain = RadarChain::new(backend, CfarParams::new(4, 2, 1e-3));
    let chirp = LfmChirp {
        sample_rate_hz: 10_000.0,
        pulse_width_s: 0.001,
        bandwidth_hz: 1_000.0,
        carrier_hz: 0.0,
        initial_phase_rad: 0.0,
    };
    let reference = chain.chirp(&chirp);
    let output = chain.detect(&reference, &reference);
    let report = adapter.fuse_decisions(chain.backend_name(), &output.cfar);

    assert_eq!(report.backend_name, "cpu");
    assert_eq!(
        report.evaluated_cells,
        output
            .cfar
            .iter()
            .filter(|decision| decision.evaluated)
            .count()
    );
    assert_eq!(report.detections, report.tracks.len());
}
