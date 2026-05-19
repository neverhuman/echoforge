//! Artifact writers for per-record micro-Doppler, multi-view, learned-window
//! products, the feature summary aggregation, and the phase-tiered detector
//! evaluation used by the pipeline.

use std::fs;
use std::path::Path;

use echoforge_radar::{
    KinematicObservation, KinematicSample, PhaseTieredDetector, SyntheticEpisode, TargetState,
};
use serde_json::json;

use super::types::{
    LearnedWindowEntry, LearnedWindowManifest, MicroDopplerDescriptors,
    MlEnvelope, MlFeatureSummaryRow, MlFrameFeatureRow, PerRecordTierObservation,
};
use crate::export::write_json_pretty as write_json;
use crate::monte_carlo::DatasetError;
use super::util::{
    cadence_velocity, cepstrum_proxy, entropy, stft_spectrogram,
    weighted_spectrum, write_f32_tensor,
};

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

/// Build a 1-D Gaussian-smeared spatial projection tensor of shape
/// `[frames, bins]`. For each frame, `per_feature` returns a `bin → f32`
/// closure that computes the pixel value for that bin. This eliminates the
/// repeated `Vec::with_capacity → for feature → for bin → push` pattern
/// shared by the range-time and Doppler-time projections.
fn build_1d_projection<F, G>(
    frames: usize,
    bins: usize,
    features: &[MlFrameFeatureRow],
    per_feature: F,
) -> Vec<f32>
where
    F: Fn(&MlFrameFeatureRow) -> G,
    G: Fn(usize) -> f32,
{
    let mut out = Vec::with_capacity(frames * bins);
    for feature in features {
        let pixel = per_feature(feature);
        for bin in 0..bins {
            out.push(pixel(bin));
        }
    }
    out
}

pub(super) fn write_multi_view_products(
    record_dir: &Path,
    features: &[MlFrameFeatureRow],
) -> Result<(), DatasetError> {
    let dir = record_dir.join("multi_view");
    fs::create_dir_all(&dir)?;
    let frames = features.len();
    let range_bins = 24usize;
    let doppler_bins = 24usize;
    let rd_range_bins = 12usize;
    let rd_doppler_bins = 12usize;

    let range_time = build_1d_projection(frames, range_bins, features, |f| {
        let center = ((f.range_m / 9_000.0) * (range_bins as f64 - 1.0))
            .clamp(0.0, range_bins as f64 - 1.0) as f32;
        let energy = f.range_time_energy;
        let noise = f.rfi_pressure;
        move |bin: usize| {
            let dist = bin as f32 - center;
            (energy * (-dist * dist / 18.0).exp() + 0.02 * noise).max(0.0)
        }
    });
    write_f32_tensor(&dir.join("range_time.zarr"), &[frames, range_bins], range_time)?;

    let doppler_time = build_1d_projection(frames, doppler_bins, features, |f| {
        let center = (((f.radial_velocity_mps + 160.0) / 320.0) * (doppler_bins as f64 - 1.0))
            .clamp(0.0, doppler_bins as f64 - 1.0) as f32;
        let energy = f.doppler_time_energy;
        let dropout = f.dropout_fraction;
        move |bin: usize| {
            let dist = bin as f32 - center;
            (energy * (-dist * dist / 14.0).exp() + 0.015 * dropout).max(0.0)
        }
    });
    write_f32_tensor(
        &dir.join("doppler_time.zarr"),
        &[frames, doppler_bins],
        doppler_time,
    )?;

    let mut rdt = Vec::with_capacity(frames * rd_range_bins * rd_doppler_bins);
    for feature in features {
        let range_center = ((feature.range_m / 9_000.0) * (rd_range_bins as f64 - 1.0))
            .clamp(0.0, rd_range_bins as f64 - 1.0) as f32;
        let doppler_center = (((feature.radial_velocity_mps + 160.0) / 320.0)
            * (rd_doppler_bins as f64 - 1.0))
            .clamp(0.0, rd_doppler_bins as f64 - 1.0) as f32;
        for d in 0..rd_doppler_bins {
            for r in 0..rd_range_bins {
                let rd = d as f32 - doppler_center;
                let rr = r as f32 - range_center;
                rdt.push(
                    (feature.range_doppler_time_energy * (-(rr * rr + rd * rd) / 12.0).exp()
                        + 0.01 * feature.rfi_pressure)
                        .max(0.0),
                );
            }
        }
    }
    write_f32_tensor(
        &dir.join("range_doppler_time.zarr"),
        &[frames, rd_doppler_bins, rd_range_bins],
        rdt,
    )?;
    write_json(
        &dir.join("range_angle_schema_pending.json"),
        &json!({
            "status": "unavailable",
            "reason": "range-angle and range-azimuth-Doppler tensors require future public-proxy MIMO channel synthesis; v1 records the schema as pending only",
            "reserved_shapes": {
                "range_angle": ["frames", "angle_bins", "range_bins"],
                "range_azimuth_doppler": ["frames", "azimuth_bins", "doppler_bins", "range_bins"]
            }
        }),
    )?;
    Ok(())
}

pub(super) fn write_learned_windows(
    record_dir: &Path,
    record_id: &str,
    features: &[MlFrameFeatureRow],
) -> Result<(), DatasetError> {
    let dir = record_dir.join("learned_windows");
    fs::create_dir_all(&dir)?;
    let feature_order = vec![
        "normalized_snr".to_string(),
        "range_norm".to_string(),
        "velocity_norm".to_string(),
        "clutter_proxy".to_string(),
        "rfi_pressure".to_string(),
        "dropout_fraction".to_string(),
        "micro_doppler_energy".to_string(),
        "tbd_track_score".to_string(),
    ];
    let mut entries = Vec::new();
    for window in [8usize, 16, 32] {
        let (values, windows) = learned_window_values(features, window);
        let path = format!("window_{window}.zarr");
        write_f32_tensor(
            &dir.join(&path),
            &[windows, window, feature_order.len()],
            values,
        )?;
        entries.push(LearnedWindowEntry {
            window_frames: window,
            path,
            shape: vec![windows, window, feature_order.len()],
        });
    }
    write_json(
        &dir.join("manifest.json"),
        &LearnedWindowManifest {
            record_id: record_id.to_string(),
            windows: entries,
            feature_order,
        },
    )?;
    Ok(())
}

fn learned_window_values(features: &[MlFrameFeatureRow], window: usize) -> (Vec<f32>, usize) {
    let stride = (window / 2).max(1);
    let windows = if features.len() <= window {
        1
    } else {
        ((features.len() - window) / stride) + 1
    };
    let mut values = Vec::with_capacity(windows * window * 8);
    for w in 0..windows {
        let start = (w * stride).min(features.len().saturating_sub(1));
        for offset in 0..window {
            let feature = features
                .get((start + offset).min(features.len().saturating_sub(1)))
                .expect("features nonempty");
            values.extend_from_slice(&[
                feature.normalized_snr,
                (feature.range_m as f32 / 10_000.0).clamp(0.0, 1.0),
                ((feature.radial_velocity_mps as f32 + 180.0) / 360.0).clamp(0.0, 1.0),
                ((feature.local_noise_floor_db + 60.0) / 48.0).clamp(0.0, 1.0),
                feature.rfi_pressure,
                feature.dropout_fraction,
                feature.micro_doppler_energy,
                feature.tbd_track_score,
            ]);
        }
    }
    (values, windows)
}

pub(super) fn summarize_features(
    record_id: &str,
    split: crate::split::SplitKind,
    class: &super::types::MlClass,
    features: &[MlFrameFeatureRow],
    first_detectable_frame: Option<usize>,
) -> MlFeatureSummaryRow {
    let len = features.len().max(1) as f32;
    MlFeatureSummaryRow {
        record_id: record_id.to_string(),
        split,
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
