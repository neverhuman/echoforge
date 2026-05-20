//! Per-frame feature/label extraction — extracted from pipeline.rs for LOC compliance.

use echoforge_radar::SyntheticEpisode;

use crate::ml_training::config::MlTrainingDataConfig;
use crate::ml_training::pipeline_writers::sample_state_at_time;
use crate::ml_training::types::{
    MlDetectorEvent, MlEnvelope, MlFrameFeatureRow, MlFrameLabelRow,
    MlRecordPlan, SplitMix64,
};

/// V3 unified-path frame feature extractor.
pub fn build_frame_products(
    config: &MlTrainingDataConfig,
    plan: &MlRecordPlan,
    envelope: &MlEnvelope,
    episode: &SyntheticEpisode,
    frame_count: usize,
    cpi_pulses: usize,
) -> (
    Vec<MlFrameFeatureRow>,
    Vec<MlFrameLabelRow>,
    Vec<MlDetectorEvent>,
    Option<usize>,
) {
    let mut features = Vec::with_capacity(frame_count);
    let mut labels = Vec::with_capacity(frame_count);
    let mut events = Vec::new();
    let mut first_detectable = None;
    let mut tbd_persistence = 0usize;

    let episode_duration_s = if cpi_pulses > 0 {
        cpi_pulses as f64 * episode.config.pri_s
    } else {
        config.time_window_s
    };
    let states = &episode.target_states;

    let mean_cfar_confidence = if episode.detections.is_empty() {
        0.0
    } else {
        episode.detections.iter().map(|d| d.confidence).sum::<f32>()
            / episode.detections.len() as f32
    };

    for frame_index in 0..frame_count {
        let time_s = frame_index as f64 / config.frame_rate_hz;
        let progress = (time_s / config.time_window_s).clamp(0.0, 1.0);
        let mut rng = SplitMix64::new(plan.scenario_seed ^ frame_index as u64 * 0x9d5b);

        let state_t = sample_state_at_time(states, time_s, episode_duration_s);

        let family_noise = rng.range_f32(-1.25, 1.25);
        let rfi_pressure =
            (envelope.rfi_pressure + rng.range_f32(-0.05, 0.09)).clamp(0.0, 1.0);
        let local_noise_floor_db: f32 = (-42.0
            + 13.0 * envelope.clutter_pressure
            + 7.0 * rfi_pressure
            + rng.range_f32(-1.5, 1.5))
        .clamp(-60.0, -12.0);

        let episode_snr = if episode.diagnostic_snr_db.is_finite() {
            episode.diagnostic_snr_db as f32
        } else {
            envelope.base_snr_db
        };
        let snr_db = ((envelope.base_snr_db * 0.4 + episode_snr * 0.6 + family_noise)
            - 3.8 * rfi_pressure
            - 2.2 * envelope.clutter_pressure)
            .clamp(-14.0, 80.0);
        let normalized_snr = ((snr_db + 14.0) / 94.0).clamp(0.0, 1.0);
        let doppler_scr = (snr_db - local_noise_floor_db.abs() * 0.02
            + envelope.speed_mps as f32 * 0.018
            - 4.0 * envelope.clutter_pressure)
            .clamp(-12.0, 38.0);
        let cfar_threshold = (snr_db.abs() * 0.6
            + 8.5
            + 15.0 * envelope.clutter_pressure
            + 12.0 * rfi_pressure
            + rng.range_f32(-1.0, 2.0))
        .clamp(0.0, 70.0);
        let cfar_statistic = snr_db
            + doppler_scr * 0.18
            + rng.range_f32(-2.0, 2.0)
            + 1.5 * (mean_cfar_confidence - 1.0).clamp(-1.0, 4.0);
        let cfar_detected = cfar_statistic >= cfar_threshold;
        if cfar_detected {
            tbd_persistence += 1;
        } else {
            tbd_persistence = tbd_persistence.saturating_sub(1);
        }
        let tbd_track_score = (0.18 * normalized_snr
            + 0.16 * (doppler_scr / 30.0).clamp(0.0, 1.0)
            + 0.12 * (tbd_persistence as f32 / 6.0).clamp(0.0, 1.0)
            - 0.20 * rfi_pressure)
            .clamp(0.0, 1.0);
        let tbd_label = cfar_detected && tbd_persistence >= 2;
        if first_detectable.is_none() && (tbd_label || tbd_track_score >= 0.58) {
            first_detectable = Some(frame_index);
        }

        let range_m = state_t.range_m + rng.range_f64(-5.0, 5.0);
        let range_m = range_m.max(40.0);
        let radial_velocity = state_t.radial_velocity_mps + rng.range_f64(-2.5, 2.5);
        let altitude_jitter = 0.85 + 0.30 * rng.unit_f64();
        let altitude = (state_t.altitude_m * altitude_jitter).max(0.0);
        let dropout_fraction = if rng.unit_f32() < envelope.dropout_probability {
            rng.range_f32(0.1, 0.6)
        } else {
            rng.range_f32(0.0, 0.04)
        };
        let micro_peak = (envelope.micro_peak_hz
            * (0.78 + 0.22 * (2.0 * std::f64::consts::PI * progress).sin() as f32)
            + rng.range_f32(-3.5, 3.5))
        .max(0.0);
        let micro_energy = (normalized_snr * 0.45
            + (micro_peak / 260.0).clamp(0.0, 1.0) * 0.35
            + rng.range_f32(0.0, 0.08))
        .clamp(0.0, 1.0);
        let stft_energy = (micro_energy * (1.0 - 0.3 * rfi_pressure)).clamp(0.0, 1.0);
        let weighted_spectrum_peak =
            (micro_energy * 0.72 + normalized_snr * 0.28).clamp(0.0, 1.0);
        let cepstrum_peak = (micro_energy * 0.55
            + (envelope.micro_bandwidth_hz / 260.0).clamp(0.0, 1.0) * 0.25)
            .clamp(0.0, 1.0);
        let cadence_velocity_peak = (micro_peak / 260.0
            * ((radial_velocity.abs() as f32) / 160.0).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
        let range_time_energy =
            (normalized_snr + envelope.clutter_pressure * 0.2).clamp(0.0, 1.2);
        let doppler_time_energy =
            ((doppler_scr + 12.0) / 50.0 + micro_energy * 0.25).clamp(0.0, 1.2);
        let range_doppler_time_energy: f32 =
            (0.45 * range_time_energy + 0.55 * doppler_time_energy).clamp(0.0, 1.2);

        features.push(MlFrameFeatureRow {
            record_id: plan.record_id.clone(),
            frame_index,
            time_s,
            cpi_pulses,
            range_m,
            radial_velocity_mps: radial_velocity,
            altitude_m: altitude,
            snr_db,
            cfar_statistic,
            cfar_threshold,
            cfar_detected,
            tbd_track_score,
            local_noise_floor_db,
            doppler_scr,
            rfi_pressure,
            dropout_fraction,
            phase_impairment_rad: envelope.phase_impairment_rad,
            amplitude_impairment: envelope.amplitude_impairment,
            micro_doppler_energy: micro_energy,
            micro_doppler_peak_hz_proxy: micro_peak,
            micro_doppler_bandwidth_hz_proxy: envelope.micro_bandwidth_hz,
            stft_energy,
            weighted_spectrum_peak,
            cepstrum_peak,
            cadence_velocity_peak,
            range_time_energy,
            doppler_time_energy,
            range_doppler_time_energy,
            normalized_snr,
        });
        labels.push(MlFrameLabelRow {
            record_id: plan.record_id.clone(),
            frame_index,
            time_s,
            split: plan.split,
            class_label: plan.class.target_family.clone(),
            is_public_proxy_positive: plan.class.is_public_proxy_positive,
            is_hard_negative: plan.class.is_hard_negative,
            hard_negative_family: plan.class.hard_negative_family.clone(),
            cfar_label: cfar_detected,
            tbd_label,
            first_detectable_frame: first_detectable,
            safety_use: "defensive early-detection and false-alarm robustness".to_string(),
        });
        if cfar_detected && events.len() < 4 {
            events.push(MlDetectorEvent {
                record_id: plan.record_id.clone(),
                detector_id: "cfar_tbd_proxy".to_string(),
                frame_index,
                time_s,
                score: cfar_statistic,
                threshold: cfar_threshold,
                event_kind: if tbd_label {
                    "tbd_track_candidate".to_string()
                } else {
                    "cfar_hit".to_string()
                },
            });
        }
    }

    (features, labels, events, first_detectable)
}
