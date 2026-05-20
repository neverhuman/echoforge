//! Record generation pipeline: worker orchestration, per-record generation,
//! and frame feature/label extraction.
//!
//! Artifact writers (micro-Doppler, multi-view, learned-window) and feature
//! summarisation live in the sibling `pipeline_writers` module.

use std::fs;
use std::thread;

use echoforge_radar::{synthesize_scene, EpisodeSeed, RuntimePlan};
use serde_json::json;

use crate::export::write_episode_tensors;
use crate::guard::build_radar_sim_config;
use crate::monte_carlo::DatasetError;

use super::config::{MlTrainingDataConfig, NEUTRAL_OBJECT_ID};
use super::envelope::sample_envelope;
pub(super) use super::pipeline_writers::evaluate_phase_tiered;
use super::pipeline_writers::{
    summarize_features, write_learned_windows, write_micro_doppler_products,
    write_multi_view_products,
};
use super::report::{feature_family_availability, guardrails};
use super::scene::{
    adapt_envelope_to_takeoff_profile, build_noise_profile, build_scene_descriptor,
};
use super::types::{
    MlRecordPlan, MlRecordSummary, MlTruthMetadata, RecordOutput, SplitManifestRow, SplitMix64,
};
use super::util::{relative_record_path, write_csv};
use crate::export::write_json_pretty as write_json;
use crate::guard::gpu_stage_recovery_note;

#[path = "pipeline_frame.rs"]
mod pipeline_frame;
pub(super) use pipeline_frame::build_frame_products;

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
            .map(|chunk| {
                scope.spawn(move || -> Result<Vec<RecordOutput>, DatasetError> {
                    chunk
                        .iter()
                        .map(|plan| generate_record(config, runtime, plan, frame_count))
                        .collect()
                })
            })
            .collect();
        handles
            .into_iter()
            .try_fold(Vec::with_capacity(plans.len()), |mut acc, handle| {
                acc.extend(handle.join().map_err(|_| {
                    DatasetError::InvalidConfig("ML training-data worker panicked".to_string())
                })??);
                Ok(acc)
            })
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
    let per_tier_observation = evaluate_phase_tiered(&episode, plan.class.is_public_proxy_positive);

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
        sensor_id: plan.sensor_id.clone(),
        phase_target: plan.phase_target.clone(),
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
        &plan.sensor_id,
        &plan.phase_target,
        &plan.class,
        &frame_features,
        first_detectable_frame,
    );
    let summary = MlRecordSummary {
        record_id: plan.record_id.clone(),
        record_index: plan.record_index,
        sensor_id: plan.sensor_id.clone(),
        phase_target: plan.phase_target.clone(),
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
        streaming_features_path: relative_record_path(&plan.record_id, "streaming_features.csv"),
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
        split_key: format!("{:016x}:{:016x}", plan.scenario_seed, plan.object_seed),
        scenario_seed: plan.scenario_seed,
        object_seed: plan.object_seed,
        class_id: plan.class.class_id.clone(),
        sensor_id: plan.sensor_id.clone(),
        phase_target: plan.phase_target.clone(),
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
