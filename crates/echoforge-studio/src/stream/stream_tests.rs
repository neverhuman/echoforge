use std::time::Duration;

use echoforge_radar::{synthesize_scene, EpisodeSeed, RadarSimConfig};

use super::control::{
    builtin_scenarios, resolve_scenario, ControlCommand, RadarParamPatch, SimMode, SimSettings,
};
use super::encode::{decode_scan_frame, encode_scan_frame, quantize_column, quantize_rd};
use super::engine::SimEngine;
use super::frames::{
    ControlFrame, MicroDopplerColumn, OutboundFrame, PpiBlip, RangeDopplerGrid, RdDetection,
    ScanFrame, ScanMeta, SessionInfo, StatusFrame, Telemetry, TrackRow, SCAN_HEADER_LEN,
};
use super::mapping::{episode_to_scan, FrameContext, TrackBook};

fn sample_scan() -> ScanFrame {
    ScanFrame {
        meta: ScanMeta {
            frame_index: 7,
            sim_time_s: 1.5,
            wall_time_ms: 123_456,
            beam_azimuth_deg: 90.0,
            ppi: vec![PpiBlip {
                entity_id: 0,
                range_m: 5000.0,
                azimuth_deg: 62.0,
                amplitude_db: 20.0,
                snr_db: 20.0,
                detected: true,
                class_label: "Shahed-class piston".to_string(),
            }],
            detections: vec![RdDetection {
                range_bin: 10,
                range_m: 5000.0,
                doppler_bin: 5,
                magnitude_db: -12.0,
                snr_db: 14.0,
            }],
            tracks: vec![TrackRow {
                track_id: 1,
                range_m: 5000.0,
                azimuth_deg: 62.0,
                radial_velocity_mps: -22.0,
                snr_db: 14.0,
                confidence: 1.5,
                class_label: "Shahed-class piston".to_string(),
                age_frames: 3,
            }],
            telemetry: Telemetry {
                snr_db: 20.0,
                received_power_dbw: -90.0,
                noise_power_dbw: -130.0,
                free_space_path_loss_db: 140.0,
                atmospheric_loss_db: 1.0,
                rain_loss_db: 0.0,
                propagation_factor_db: 0.0,
                coherent_integration_gain_db: 15.0,
                above_horizon: true,
                detections_this_frame: 1,
                frame_compute_ms: 2.0,
                scan_rate_hz: 15.0,
            },
        },
        range_doppler: RangeDopplerGrid {
            range_bins: 4,
            doppler_bins: 2,
            db_min: -60.0,
            db_max: 0.0,
            cells: vec![1, 2, 3, 4, 5, 6, 7, 8],
        },
        micro_doppler: MicroDopplerColumn {
            bins: 3,
            db_min: -60.0,
            db_max: 0.0,
            doppler_max_hz: 500.0,
            column: vec![9, 10, 11],
        },
    }
}

#[test]
fn scan_frame_binary_round_trip() {
    let frame = sample_scan();
    let bytes = encode_scan_frame(&frame).expect("encode");
    let json_len = serde_json::to_vec(&frame.meta).unwrap().len();
    assert_eq!(bytes.len(), SCAN_HEADER_LEN + json_len + 8 + 3);
    let decoded = decode_scan_frame(&bytes).expect("decode");
    assert_eq!(decoded, frame);
}

#[test]
fn decode_rejects_bad_magic() {
    let mut bytes = encode_scan_frame(&sample_scan()).unwrap();
    bytes[0] ^= 0xFF;
    assert!(decode_scan_frame(&bytes).is_err());
}

#[test]
fn quantize_rd_preserves_peak_location() {
    // [doppler][range] grid, one hot cell in doppler row 2.
    let raw_range = 512;
    let hot_range = 400;
    let mut grid = vec![vec![0.001_f32; raw_range]; 4];
    grid[2][hot_range] = 1000.0;
    let rd = quantize_rd(&grid, 256);
    assert_eq!(rd.doppler_bins, 4);
    assert_eq!(rd.range_bins, 256);
    let (idx, &val) = rd.cells.iter().enumerate().max_by_key(|(_, &v)| v).unwrap();
    assert_eq!(val, 255, "peak cell must saturate to 255");
    assert_eq!(idx / rd.range_bins, 2, "peak must stay in doppler row 2");
}

#[test]
fn quantize_column_resamples_to_target_bins() {
    let spectrum: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let column = quantize_column(&spectrum, 500.0, 256);
    assert_eq!(column.bins, 256);
    assert_eq!(column.column.len(), 256);
}

#[test]
fn control_frame_json_round_trips() {
    let session = ControlFrame::SessionInfo(SessionInfo {
        session_id: 3,
        source: "live".to_string(),
        scenario_id: "shahed-ingress".to_string(),
        scenario_label: "Shahed-class ingress".to_string(),
        frame_rate_hz: 15.0,
        running: true,
        paused: false,
        playback_speed: 1.0,
        range_max_m: 19_000.0,
        doppler_max_hz: 555.0,
        rd_range_bins: 256,
        rd_doppler_bins: 32,
        spectrogram_bins: 256,
        available_scenarios: Vec::new(),
        schema_version: 1,
    });
    let text = serde_json::to_string(&session).unwrap();
    let back: ControlFrame = serde_json::from_str(&text).unwrap();
    assert_eq!(back, session);

    let status = ControlFrame::Status(StatusFrame::warn("lagged", "behind"));
    let text = serde_json::to_string(&status).unwrap();
    assert_eq!(serde_json::from_str::<ControlFrame>(&text).unwrap(), status);
}

#[test]
fn param_patch_overlays_only_some_fields() {
    let mut cfg = RadarSimConfig::default();
    let baseline_gain = cfg.tx_gain_dbi;
    let patch = RadarParamPatch {
        transmit_power_w: Some(2.0e6),
        cfar_pfa: Some(1.0e-4),
        ..RadarParamPatch::default()
    };
    patch.apply_to(&mut cfg);
    assert_eq!(cfg.transmit_power_w, 2.0e6);
    assert_eq!(cfg.cfar_pfa, 1.0e-4);
    assert_eq!(
        cfg.tx_gain_dbi, baseline_gain,
        "untouched field must persist"
    );
}

#[test]
fn builtin_scenarios_resolve_and_are_well_formed() {
    let all = builtin_scenarios();
    assert!(all.len() >= 3);
    for scenario in &all {
        assert!(
            !scenario.scene().targets.is_empty(),
            "scenario {} has no targets",
            scenario.id
        );
    }
    let fallback = resolve_scenario("does-not-exist");
    assert_eq!(fallback.id, all[0].id);
}

#[test]
fn episode_maps_to_a_complete_scan_frame() {
    let scenario = resolve_scenario("shahed-ingress");
    let episode = synthesize_scene(
        scenario.scene(),
        scenario.config.clone(),
        scenario.noise,
        EpisodeSeed(1),
    );
    let mut tracker = TrackBook::default();
    let ctx = FrameContext {
        scenario: &scenario,
        frame_index: 0,
        sim_time_s: 0.0,
        wall_time_ms: 0,
        beam_azimuth_deg: 0.0,
        rd_range_bins: 256,
        spectrogram_bins: 256,
        frame_compute_ms: 1.0,
        scan_rate_hz: 15.0,
    };
    let scan = episode_to_scan(&episode, &ctx, &mut tracker);
    assert_eq!(scan.meta.ppi.len(), scenario.entities.len());
    assert_eq!(scan.micro_doppler.bins, 256);
    assert!(!scan.range_doppler.cells.is_empty());
    assert!(scan.meta.telemetry.snr_db.is_finite());
    // The encoded form must be decodable.
    let bytes = encode_scan_frame(&scan).unwrap();
    assert!(decode_scan_frame(&bytes).is_ok());
}

#[tokio::test]
async fn engine_starts_streams_and_stops() {
    let engine = SimEngine::new(SimSettings {
        default_frame_rate_hz: 60.0,
        autostart: false,
        ..SimSettings::default()
    });
    let mut rx = engine.subscribe();
    engine
        .apply(ControlCommand::Start {
            scenario_id: "shahed-ingress".to_string(),
            mode: SimMode::Live,
        })
        .await;
    assert!(engine.current_status().running);

    let mut scans = 0;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while scans < 4 && tokio::time::Instant::now() < deadline {
        if let Ok(Ok(frame)) = tokio::time::timeout(Duration::from_secs(2), rx.recv()).await {
            if matches!(&*frame, OutboundFrame::Scan(_)) {
                scans += 1;
            }
        }
    }
    assert!(scans >= 4, "expected >=4 scan frames, got {scans}");

    engine.apply(ControlCommand::Stop).await;
    assert!(!engine.current_status().running);
}
