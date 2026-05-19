//! Record generation pipeline: worker orchestration, per-record generation,
//! frame feature/label extraction, and micro-Doppler/multi-view/learned-window
//! artifact writers.

use std::fs;
use std::path::Path;
use std::thread;

use echoforge_radar::{
    synthesize_scene, EpisodeSeed, KinematicObservation, KinematicSample,
    PhaseTieredDetector, RuntimePlan, SyntheticEpisode, TargetState,
};
use serde_json::json;

use crate::export::write_episode_tensors;
use crate::guard::build_radar_sim_config;
use crate::monte_carlo::DatasetError;

use super::config::{MlTrainingDataConfig, NEUTRAL_OBJECT_ID};
use super::envelope::sample_envelope;
use crate::guard::gpu_stage_recovery_note;
use super::report::{feature_family_availability, guardrails};
use super::scene::{adapt_envelope_to_takeoff_profile, build_noise_profile, build_scene_descriptor};
use super::types::{
    MlDetectorEvent, MlEnvelope, MlFeatureSummaryRow, MlFrameFeatureRow, MlFrameLabelRow,
    MlRecordPlan, MlRecordSummary, MlTruthMetadata, PerRecordTierObservation, RecordOutput,
    SplitManifestRow, SplitMix64,
};
use super::types::{
    LearnedWindowEntry, LearnedWindowManifest, MicroDopplerDescriptors,
};
use crate::export::write_json_pretty as write_json;
use super::util::{
    cadence_velocity, cepstrum_proxy, entropy, relative_record_path, stft_spectrogram,
    weighted_spectrum, write_csv, write_f32_tensor,
};

pub(super) fn run_record_workers(
    config: &MlTrainingDataConfig,
    runtime: &RuntimePlan,
    plans: &[MlRecordPlan],
    frame_count: usize,
    worker_count: usize,
) -> Result<Vec<RecordOutput>, DatasetError> {
    let chunk_size = (plans.len() + worker_count - 1) / worker_count;
    thread::scope(|scope| {
        let handles: Vec<_> = plans
            .chunks(chunk_size.max(1))
            .map(|chunk| scope.spawn(move || -> Result<Vec<RecordOutput>, DatasetError> {
                chunk.iter()
                    .map(|plan| generate_record(config, runtime, plan, frame_count))
                    .collect()
            }))
            .collect();
        handles.into_iter().try_fold(
            Vec::with_capacity(plans.len()),
            |mut acc, handle| {
                acc.extend(handle.join().map_err(|_| {
                    DatasetError::InvalidConfig("ML training-data worker panicked".to_string())
                })??);
                Ok(acc)
            },
        )
    })
}

fn generate_record(
    config: &MlTrainingDataConfig,
    runtime: &RuntimePlan,
    plan: &MlRecordPlan,
    frame_count: usize,
) -> Result<RecordOutput, DatasetError> {
    let record_dir = config.output_dir.join("records").join(&plan.record_id);
    let products_dir = record_dir.join("products");
    fs::create_dir_all(&products_dir)?;

    let mut rng = SplitMix64::new(plan.scenario_seed);
    let envelope = sample_envelope(&plan.class, &mut rng);
    let cpi_pulses = rng.range_usize(24, 48);
    let noise = build_noise_profile(&envelope);

    let sim_config = build_radar_sim_config(envelope.base_snr_db, cpi_pulses);
    let profile = adapt_envelope_to_takeoff_profile(&envelope, &mut rng);

    // V3 unified path (Wave 5 Lane K_rust): construct a SceneDescriptor
    // with an explicit `TargetClass` so the truth class is named at the
    // scene level rather than inferred from which generator branch
    // produced the envelope. Lane I's `synthesize_scene` currently
    // accepts only a single `FromTakeoffProfile` entity, so multipath
    // ghosts cannot yet be wired as paired entities; see
    // `confuser_class_for_family` for the Lane J reconciliation items.
    let scene = build_scene_descriptor(&plan.class, profile, &sim_config, &noise);
    let episode = synthesize_scene(
        scene,
        sim_config,
        noise,
        EpisodeSeed(plan.scenario_seed ^ 0x0dd5_136),
    );
    write_episode_tensors(&products_dir, &episode)?;

    // V3 unified path: per-tier Pd/Pfa evaluation using the
    // phase-tiered detector (Lane H/H2). The detector consumes the
    // episode's `target_states` window as kinematic input and
    // optionally a per-CPI Doppler spectrum (Tier 3 cruise check).
    let per_tier_observation =
        evaluate_phase_tiered(&episode, plan.class.is_public_proxy_positive);

    let (frame_features, frame_labels, events, first_detectable_frame) =
        build_frame_products(config, plan, &envelope, &episode, frame_count, cpi_pulses);
    write_csv(&record_dir.join("streaming_features.csv"), &frame_features)?;
    write_csv(&record_dir.join("frame_labels.csv"), &frame_labels)?;
    write_json(&record_dir.join("detector_events.json"), &events)?;

    write_micro_doppler_products(&record_dir, &plan.record_id, &frame_features, &envelope)?;
    write_multi_view_products(&record_dir, &frame_features)?;
    write_learned_windows(&record_dir, &plan.record_id, &frame_features)?;

    let truth = MlTruthMetadata {
        record_id: plan.record_id.clone(),
        neutral_object_id: NEUTRAL_OBJECT_ID.to_string(),
        dataset_id: config.dataset.clone(),
        split: plan.split,
        class_id: plan.class.class_id.clone(),
        target_family: plan.class.target_family.clone(),
        is_public_proxy_positive: plan.class.is_public_proxy_positive,
        is_hard_negative: plan.class.is_hard_negative,
        hard_negative_family: plan.class.hard_negative_family.clone(),
        scenario_seed: plan.scenario_seed,
        object_seed: plan.object_seed,
        dimensions_m: envelope.dimensions_m.clone(),
        rcs_dbsm_proxy: envelope.rcs_dbsm,
        speed_mps_proxy: envelope.speed_mps,
        time_window_s: config.time_window_s,
        frame_rate_hz: config.frame_rate_hz,
        frame_count,
        guardrails: guardrails(),
    };
    write_json(&record_dir.join("truth_metadata.json"), &truth)?;
    write_json(
        &record_dir.join("runtime_stage.json"),
        &json!({
            "selected_backend": runtime.selected_backend.to_string(),
            "kernel_backend": "cpu-scaffold",
            "gpu_stage_recovery": gpu_stage_recovery_note(runtime),
        }),
    )?;
    write_json(
        &record_dir.join("feature_family_availability.json"),
        &feature_family_availability(&plan.record_id),
    )?;

    let feature_summary = summarize_features(
        &plan.record_id,
        plan.split,
        &plan.class,
        &frame_features,
        first_detectable_frame,
    );
    let summary = MlRecordSummary {
        record_id: plan.record_id.clone(),
        record_index: plan.record_index,
        split: plan.split,
        class_id: plan.class.class_id.clone(),
        target_family: plan.class.target_family.clone(),
        is_public_proxy_positive: plan.class.is_public_proxy_positive,
        is_hard_negative: plan.class.is_hard_negative,
        scenario_seed: plan.scenario_seed,
        object_seed: plan.object_seed,
        hard_negative_family: plan.class.hard_negative_family.clone(),
        frame_count,
        cpi_pulses,
        tensor_dir: relative_record_path(&plan.record_id, "products"),
        streaming_features_path: relative_record_path(
            &plan.record_id,
            "streaming_features.csv",
        ),
        frame_labels_path: relative_record_path(&plan.record_id, "frame_labels.csv"),
        truth_metadata_path: relative_record_path(&plan.record_id, "truth_metadata.json"),
        detector_events_path: relative_record_path(&plan.record_id, "detector_events.json"),
        feature_family_availability_path: relative_record_path(
            &plan.record_id,
            "feature_family_availability.json",
        ),
        micro_doppler_dir: relative_record_path(&plan.record_id, "micro_doppler"),
        multi_view_dir: relative_record_path(&plan.record_id, "multi_view"),
        learned_windows_dir: relative_record_path(&plan.record_id, "learned_windows"),
    };
    let split_row = SplitManifestRow {
        record_id: plan.record_id.clone(),
        split: plan.split,
        split_key_kind: "scenario_object_seed".to_string(),
        split_key: format!(
            "{:016x}:{:016x}",
            plan.scenario_seed, plan.object_seed
        ),
        scenario_seed: plan.scenario_seed,
        object_seed: plan.object_seed,
        class_id: plan.class.class_id.clone(),
        target_family: plan.class.target_family.clone(),
        hard_negative_family: plan.class.hard_negative_family.clone(),
    };

    Ok(RecordOutput {
        summary,
        features: feature_summary,
        split: split_row,
        per_tier_observation,
    })
}

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
fn sample_state_at_time(
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

/// V3 unified-path frame feature extractor.
pub(super) fn build_frame_products(
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
        let local_noise_floor_db = (-42.0
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
        let range_doppler_time_energy =
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

fn write_micro_doppler_products(
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

fn write_multi_view_products(
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

fn write_learned_windows(
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
