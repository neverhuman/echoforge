//! Per-frame feature/label extraction — extracted from pipeline.rs for LOC compliance.

use echoforge_radar::SyntheticEpisode;

use crate::ml_training::config::MlTrainingDataConfig;
use crate::ml_training::types::{
    MlDetectorEvent, MlFrameFeatureRow, MlFrameLabelRow, MlRecordPlan,
};

fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}

fn lower_quartile_mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = (sorted.len() / 4).max(1);
    sorted.iter().take(n).copied().sum::<f32>() / n as f32
}

/// V3 unified-path frame feature extractor.
pub fn build_frame_products(
    config: &MlTrainingDataConfig,
    plan: &MlRecordPlan,
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

    let mean_cfar_confidence = if episode.detections.is_empty() {
        0.0
    } else {
        episode.detections.iter().map(|d| d.confidence).sum::<f32>()
            / episode.detections.len() as f32
    };
    let doppler_bin_hz = if cpi_pulses > 0 && episode.config.pri_s > 0.0 {
        1.0 / (cpi_pulses as f64 * episode.config.pri_s)
    } else {
        0.0
    };
    let range_profiles = &episode.range_profiles_by_pulse;
    let range_doppler = &episode.range_doppler_proxy;
    let pulse_count = range_profiles.len().max(1);

    for frame_index in 0..frame_count {
        let time_s = frame_index as f64 / config.frame_rate_hz;
        let progress = (time_s / config.time_window_s).clamp(0.0, 1.0);

        let pulse_idx = frame_index % pulse_count;
        let pulse_profile = range_profiles
            .get(pulse_idx)
            .map(|profile| profile.as_slice())
            .unwrap_or(&[]);
        let integrated_peak_bin = argmax(&episode.integrated_range_profile).unwrap_or(0);
        let frame_peak_bin = argmax(pulse_profile).unwrap_or(integrated_peak_bin);
        let frame_peak_range_bin = frame_peak_bin.min(pulse_profile.len().saturating_sub(1));
        let delay = frame_peak_range_bin as isize - pulse_profile.len().saturating_sub(1) as isize;
        let range_m = if delay <= 0 {
            0.0
        } else {
            delay as f64 * 299_792_458.0 / (2.0 * episode.config.sample_rate_hz.max(1.0))
        };

        let lower_quartile = lower_quartile_mean(pulse_profile).max(1e-6);
        let peak_value = pulse_profile
            .get(frame_peak_range_bin)
            .copied()
            .unwrap_or(0.0);
        let snr_linear = (peak_value / lower_quartile).max(1e-6);
        let snr_db = (10.0 * snr_linear.log10()) as f32;
        let local_noise_floor_db: f32 = 10.0 * lower_quartile.log10();

        let rd_row = range_doppler
            .get(frame_peak_range_bin)
            .map(|row| row.as_slice())
            .unwrap_or(&[]);
        let rd_peak_bin = argmax(rd_row).unwrap_or(0);
        let rd_peak_value = rd_row.get(rd_peak_bin).copied().unwrap_or(0.0);
        let rd_energy = if rd_row.is_empty() {
            0.0
        } else {
            rd_row.iter().copied().sum::<f32>() / rd_row.len() as f32
        };
        let doppler_center = rd_row.len() / 2;
        let doppler_offset_bins = rd_peak_bin as isize - doppler_center as isize;
        let doppler_hz = if doppler_bin_hz > 0.0 {
            doppler_offset_bins.unsigned_abs() as f64 * doppler_bin_hz
        } else {
            0.0
        };
        let doppler_scr =
            (((rd_peak_value / rd_energy.max(1e-6)).max(1e-6)).log10() * 10.0).clamp(-12.0, 38.0);
        let rfi_pressure = (episode.noise.rfi_probability
            + 0.25
                * episode
                    .pulse_diagnostics
                    .get(pulse_idx)
                    .map(|diag| {
                        diag.source_diagnostics
                            .iter()
                            .map(|src| src.interference_power_w as f32)
                            .sum::<f32>()
                            .min(1.0)
                    })
                    .unwrap_or(0.0))
        .clamp(0.0, 1.0);
        let clutter_pressure = (episode.noise.clutter_sigma * 4.0).clamp(0.0, 1.0);
        let cfar_threshold =
            (local_noise_floor_db + 7.5 + 11.0 * rfi_pressure + 8.0 * clutter_pressure)
                .clamp(-60.0, 80.0);
        let normalized_snr = ((snr_db + 14.0) / 94.0).clamp(0.0, 1.0);
        let cfar_statistic = (snr_db
            + 0.18 * doppler_scr
            + 0.08 * rd_energy
            + 1.5 * (mean_cfar_confidence - 1.0).clamp(-1.0, 4.0))
        .clamp(-40.0, 90.0);
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

        let radial_velocity = if doppler_hz > 0.0 {
            doppler_hz * 299_792_458.0 / (2.0 * episode.config.carrier_hz.max(1.0))
        } else {
            0.0
        };
        let altitude = (range_m * (0.015 + 0.02 * progress)).clamp(0.0, range_m.max(1.0));
        let dropout_fraction = (1.0
            - (pulse_profile
                .iter()
                .filter(|value| **value > lower_quartile * 1.05)
                .count() as f32
                / pulse_profile.len().max(1) as f32))
            .clamp(0.0, 1.0);
        let micro_peak = if doppler_hz > 0.0 {
            doppler_hz as f32
        } else {
            0.0
        };
        let micro_energy = (rd_peak_value / (rd_energy + lower_quartile)).clamp(0.0, 1.0);
        let stft_energy = (micro_energy * (1.0 - 0.3 * rfi_pressure)).clamp(0.0, 1.0);
        let weighted_spectrum_peak =
            (micro_energy * 0.72 + (snr_db / 40.0).clamp(0.0, 1.0) * 0.28).clamp(0.0, 1.0);
        let cepstrum_peak = (micro_energy * 0.55
            + (episode.noise.phase_noise_std_rad * 3.5).clamp(0.0, 1.0) * 0.25)
            .clamp(0.0, 1.0);
        let cadence_velocity_peak = ((radial_velocity.abs() as f32 / 160.0).clamp(0.0, 1.0)
            * (micro_peak / 260.0).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
        let range_time_energy = ((snr_db + 14.0) / 94.0 + clutter_pressure * 0.2).clamp(0.0, 1.2);
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
            phase_impairment_rad: episode.noise.phase_noise_std_rad,
            amplitude_impairment: episode.noise.amplitude_scintillation_sigma,
            micro_doppler_energy: micro_energy,
            micro_doppler_peak_hz_proxy: micro_peak,
            micro_doppler_bandwidth_hz_proxy: (doppler_bin_hz as f32 * rd_row.len() as f32)
                .max(0.0),
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
