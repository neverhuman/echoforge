//! Record generation pipeline: worker orchestration, per-record generation,
//! and frame feature/label extraction.
//!
//! Artifact writers (micro-Doppler, multi-view, learned-window) and feature
//! summarisation live in the sibling `pipeline_writers` module.

use std::fs;
use std::thread;

use echoforge_radar::{synthesize_scene, EpisodeSeed, RuntimePlan, SyntheticEpisode};
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
    MlDetectorEvent, MlEnvelope, MlFrameFeatureRow, MlFrameLabelRow,
    MlRecordPlan, MlRecordSummary, MlTruthMetadata, RecordOutput,
    SplitManifestRow, SplitMix64,
};
use crate::export::write_json_pretty as write_json;
use super::util::{relative_record_path, write_csv};
use super::pipeline_writers::{
    sample_state_at_time, summarize_features, write_learned_windows,
    write_micro_doppler_products, write_multi_view_products,
};
pub(super) use super::pipeline_writers::evaluate_phase_tiered;

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

