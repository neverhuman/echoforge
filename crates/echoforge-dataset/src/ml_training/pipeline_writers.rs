//! Artifact writers for per-record micro-Doppler, multi-view, learned-window
//! products, the feature summary aggregation, and the phase-tiered detector
//! evaluation used by the pipeline.

use std::fs;
use std::path::Path;

use echoforge_radar::{
    KinematicObservation, KinematicSample, PhaseTieredDetector, SyntheticEpisode, TargetState,
};

use super::types::{
    MicroDopplerDescriptors, MlEnvelope, MlFeatureSummaryRow, MlFrameFeatureRow,
    PerRecordTierObservation,
};
use super::util::{
    cadence_velocity, cepstrum_proxy, entropy, stft_spectrogram, weighted_spectrum,
    write_f32_tensor,
};
use crate::export::write_json_pretty as write_json;
use crate::monte_carlo::DatasetError;

#[path = "pipeline_writers_tensors.rs"]
mod pipeline_writers_tensors;
pub(super) use pipeline_writers_tensors::{write_learned_windows, write_multi_view_products};

/// Run [`PhaseTieredDetector::evaluate_cpi`] against a synthesised
/// episode and aggregate the per-tier counts.
pub(super) fn evaluate_phase_tiered(
    episode: &SyntheticEpisode,
    is_positive: bool,
) -> PerRecordTierObservation {
    let mut detector = PhaseTieredDetector::default();
    let mut observation = PerRecordTierObservation::new(is_positive);

    // Doppler bin spacing for the cruise tier's micro-Doppler check.
    let pulse_count = episode.config.pulse_count.max(1);
    let doppler_bin_hz = if episode.config.pri_s > 0.0 {
        Some(1.0 / (pulse_count as f64 * episode.config.pri_s))
    } else {
        None
    };

    let antenna_height = episode.config.radar_altitude_agl_m;
    let range_m = episode.profile.initial_range_m;

    // 30 frames at 1 Hz so the arbiter has time to walk None → Boost
    // (3-of-5 trailing rule) → ClimbOut → Cruise (10 steady CPIs).
    let n_frames = 30usize;
    let mut sample_buffer: Vec<KinematicSample> = Vec::with_capacity(n_frames);
    for frame_idx in 0..n_frames {
        let t_s = frame_idx as f64;
        let (speed_mps, altitude_m) = synthetic_kinematic_state(episode, is_positive, t_s);
        sample_buffer.push(KinematicSample::new(t_s, speed_mps, altitude_m));

        // Trailing 6-sample observation window.
        let window_start = sample_buffer.len().saturating_sub(6);
        let window_samples = sample_buffer[window_start..].to_vec();
        let obs = KinematicObservation::new(window_samples, range_m, antenna_height);

        let row_idx = frame_idx % pulse_count;
        let mtd_slice: Option<Vec<f32>> = if row_idx < episode.range_doppler_proxy.len() {
            Some(episode.range_doppler_proxy[row_idx].clone())
        } else {
            None
        };

        let decision = detector.evaluate_cpi(&obs, mtd_slice.as_deref(), doppler_bin_hz);
        observation.record(&decision);
    }

    observation
}

/// Build a per-frame (speed, altitude) tuple for the phase-tiered arbiter.
fn synthetic_kinematic_state(
    episode: &SyntheticEpisode,
    is_positive: bool,
    t_s: f64,
) -> (f64, f64) {
    if is_positive {
        let cruise_speed = episode.profile.ground_speed_mps.clamp(45.0, 55.0);
        let cruise_alt = episode.profile.max_altitude_m.clamp(60.0, 1_400.0);
        if t_s <= 3.0 {
            let speed = 5.0 + (32.0 - 5.0) / 3.0 * t_s;
            let altitude = 40.0 + 20.0 * t_s;
            (speed, altitude)
        } else if t_s < 8.0 {
            let progress = (t_s - 3.0) / 5.0;
            let speed = 32.0 + (cruise_speed - 32.0) * progress;
            let altitude = 100.0 + (cruise_alt - 100.0).max(0.0) * progress.clamp(0.0, 1.0);
            (speed, altitude)
        } else {
            let jitter = 0.05 * (t_s * 0.7).sin();
            (cruise_speed + jitter, cruise_alt)
        }
    } else {
        let base_speed = 5.0 + 10.0 * (t_s * 0.3).sin().abs();
        let altitude = 30.0 + 20.0 * (t_s * 0.15).cos();
        (base_speed, altitude)
    }
}

/// Resample the episode's per-pulse `TargetState` vector at a given frame time.
pub(super) fn sample_state_at_time(
    states: &[TargetState],
    time_s: f64,
    episode_duration_s: f64,
) -> TargetState {
    if states.is_empty() {
        return TargetState {
            time_s: 0.0,
            range_m: 0.0,
            altitude_m: 0.0,
            radial_velocity_mps: 0.0,
            pitch_deg: 0.0,
            yaw_deg: 0.0,
            propulsor_phase_rad: 0.0,
        };
    }
    if states.len() == 1 || episode_duration_s <= 0.0 {
        return states[0];
    }
    let normalized = (time_s / episode_duration_s).clamp(0.0, 1.0);
    let scaled = normalized * (states.len() as f64 - 1.0);
    let lower = scaled.floor() as usize;
    let upper = (lower + 1).min(states.len() - 1);
    let t = (scaled - lower as f64) as f32;
    let a = &states[lower];
    let b = &states[upper];
    let lerp64 = |x: f64, y: f64| x + (y - x) * t as f64;
    TargetState {
        time_s: lerp64(a.time_s, b.time_s),
        range_m: lerp64(a.range_m, b.range_m),
        altitude_m: lerp64(a.altitude_m, b.altitude_m),
        radial_velocity_mps: lerp64(a.radial_velocity_mps, b.radial_velocity_mps),
        pitch_deg: lerp64(a.pitch_deg, b.pitch_deg),
        yaw_deg: lerp64(a.yaw_deg, b.yaw_deg),
        propulsor_phase_rad: lerp64(a.propulsor_phase_rad, b.propulsor_phase_rad),
    }
}

pub(super) fn write_micro_doppler_products(
    record_dir: &Path,
    record_id: &str,
    features: &[MlFrameFeatureRow],
    envelope: &MlEnvelope,
) -> Result<(), DatasetError> {
    let dir = record_dir.join("micro_doppler");
    fs::create_dir_all(&dir)?;
    let signal = features
        .iter()
        .map(|feature| {
            (feature.micro_doppler_energy
                * (1.0 + 0.15 * (feature.frame_index as f32 * 0.37).sin()))
            .max(0.0)
        })
        .collect::<Vec<_>>();
    let stft = stft_spectrogram(&signal, 16, 4, 16);
    let stft_windows = stft.len() / 16;
    write_f32_tensor(
        &dir.join("stft_spectrogram.zarr"),
        &[stft_windows, 16],
        stft,
    )?;
    let weighted = weighted_spectrum(&signal, 32);
    write_f32_tensor(&dir.join("weighted_spectrum.zarr"), &[32], weighted.clone())?;
    let cepstrum = cepstrum_proxy(&weighted, 32);
    write_f32_tensor(&dir.join("cepstrum.zarr"), &[32], cepstrum.clone())?;
    let cadence = cadence_velocity(&weighted, envelope.radial_velocity_mps as f32, 16, 16);
    write_f32_tensor(
        &dir.join("cadence_velocity.zarr"),
        &[16, 16],
        cadence.clone(),
    )?;

    let descriptors = MicroDopplerDescriptors {
        record_id: record_id.to_string(),
        peak_hz_proxy: envelope.micro_peak_hz,
        bandwidth_hz_proxy: envelope.micro_bandwidth_hz,
        weighted_spectrum_entropy: entropy(&weighted),
        cepstrum_peak: cepstrum.iter().copied().fold(0.0, f32::max),
        cadence_velocity_peak: cadence.iter().copied().fold(0.0, f32::max),
        representation_note: "Publication-backed proxy formats: STFT, weighted spectrum, cepstrum, and cadence-velocity summary. Values are synthetic public proxies.".to_string(),
    };
    write_json(&dir.join("descriptors.json"), &descriptors)?;
    Ok(())
}

pub(super) fn summarize_features(
    record_id: &str,
    split: crate::split::SplitKind,
    sensor_id: &str,
    phase_target: &str,
    class: &super::types::MlClass,
    features: &[MlFrameFeatureRow],
    first_detectable_frame: Option<usize>,
) -> MlFeatureSummaryRow {
    let len = features.len().max(1) as f32;
    MlFeatureSummaryRow {
        record_id: record_id.to_string(),
        split,
        sensor_id: sensor_id.to_string(),
        phase_target: phase_target.to_string(),
        target_family: class.target_family.clone(),
        hard_negative_family: class.hard_negative_family.clone(),
        is_public_proxy_positive: class.is_public_proxy_positive,
        mean_snr_db: features.iter().map(|row| row.snr_db).sum::<f32>() / len,
        max_snr_db: features
            .iter()
            .map(|row| row.snr_db)
            .fold(f32::MIN, f32::max),
        mean_doppler_scr: features.iter().map(|row| row.doppler_scr).sum::<f32>() / len,
        mean_rfi_pressure: features.iter().map(|row| row.rfi_pressure).sum::<f32>() / len,
        dropout_fraction: features.iter().map(|row| row.dropout_fraction).sum::<f32>() / len,
        mean_micro_doppler_energy: features
            .iter()
            .map(|row| row.micro_doppler_energy)
            .sum::<f32>()
            / len,
        micro_doppler_peak_hz_proxy: features
            .iter()
            .map(|row| row.micro_doppler_peak_hz_proxy)
            .fold(0.0, f32::max),
        micro_doppler_bandwidth_hz_proxy: features
            .iter()
            .map(|row| row.micro_doppler_bandwidth_hz_proxy)
            .sum::<f32>()
            / len,
        mean_track_score: features.iter().map(|row| row.tbd_track_score).sum::<f32>() / len,
        cfar_detection_fraction: features.iter().filter(|row| row.cfar_detected).count() as f32
            / len,
        first_detectable_frame,
    }
}
