use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use echoforge_radar::{
    synthesize_takeoff_episode, EpisodeSeed, NoiseProfile, RuntimePlan, TakeoffProfile,
};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::json;

use super::reporting::campaign_guardrails;
use super::rng::SplitMix64;
use super::simulation::{
    apply_campaign_stressors, build_frame_features, build_frame_labels, class_envelope,
    run_streaming_models,
};
use super::types::{
    CampaignConfig, CampaignRecordOutput, CampaignRecordPlan, CampaignRecordSummary,
    RecordTruthMetadata,
};
use crate::export::{write_episode_tensors, write_json_pretty};
use crate::guard::build_radar_sim_config;
use crate::guard::gpu_stage_recovery_note;
use crate::monte_carlo::DatasetError;

use super::reporting::write_csv;

// ── Worker pool ───────────────────────────────────────────────────────────────

pub(super) fn run_campaign_workers(
    config: &CampaignConfig,
    runtime: &RuntimePlan,
    plans: &[CampaignRecordPlan],
    frame_count: usize,
    worker_count: usize,
    progress_enabled: bool,
) -> Result<Vec<CampaignRecordOutput>, DatasetError> {
    let chunk_size = plans.len().div_ceil(worker_count);
    let progress_bar = if progress_enabled {
        let bar = ProgressBar::new(plans.len() as u64);
        let style = ProgressStyle::with_template(
            "{spinner:.green} {pos}/{len} records positives={msg} [{elapsed_precise}] {per_sec} ETA {eta_precise}",
        )
        .expect("progress template is a compile-time constant and always valid")
        .progress_chars("=> ");
        bar.set_style(style);
        Some(bar)
    } else {
        None
    };
    let positive_done = Arc::new(AtomicUsize::new(0));

    let result = thread::scope(|scope| -> Result<Vec<CampaignRecordOutput>, DatasetError> {
        let mut handles = Vec::new();
        for chunk in plans.chunks(chunk_size.max(1)) {
            let progress = progress_bar.clone();
            let positive_done = Arc::clone(&positive_done);
            handles.push(scope.spawn(move || {
                run_campaign_chunk(config, runtime, chunk, frame_count, progress, positive_done)
            }));
        }
        let mut outputs = Vec::with_capacity(plans.len());
        for handle in handles {
            let mut chunk = handle.join().map_err(|_| {
                DatasetError::InvalidConfig("campaign worker panicked".to_string())
            })??;
            outputs.append(&mut chunk);
        }
        Ok(outputs)
    });

    if let Some(bar) = progress_bar {
        bar.finish_and_clear();
    }
    result
}

fn run_campaign_chunk(
    config: &CampaignConfig,
    runtime: &RuntimePlan,
    plans: &[CampaignRecordPlan],
    frame_count: usize,
    progress_bar: Option<ProgressBar>,
    positive_done: Arc<AtomicUsize>,
) -> Result<Vec<CampaignRecordOutput>, DatasetError> {
    let mut outputs = Vec::with_capacity(plans.len());
    for plan in plans {
        let output = generate_campaign_record(config, runtime, plan, frame_count)?;
        if plan.class.is_shahed_public_proxy {
            positive_done.fetch_add(1, Ordering::SeqCst);
        }
        if let Some(bar) = progress_bar.as_ref() {
            let positives = positive_done.load(Ordering::SeqCst);
            bar.set_message(format!(
                "{positives} class={} backend={}",
                plan.class.target_family, runtime.selected_backend
            ));
            bar.inc(1);
        }
        outputs.push(output);
    }
    Ok(outputs)
}

// ── Single record generation ──────────────────────────────────────────────────

fn generate_campaign_record(
    config: &CampaignConfig,
    runtime: &RuntimePlan,
    plan: &CampaignRecordPlan,
    frame_count: usize,
) -> Result<CampaignRecordOutput, DatasetError> {
    let record_dir = config.output_dir.join("records").join(&plan.record_id);
    let products_dir = record_dir.join("products");
    fs::create_dir_all(&products_dir)?;
    let mut rng = SplitMix64::new(plan.seed);
    let envelope = class_envelope(&plan.class, &mut rng);
    let cpi_pulses = rng.range_usize(32, 96);

    let sim_config = build_radar_sim_config(envelope.base_snr_db, cpi_pulses);
    let profile = TakeoffProfile {
        initial_range_m: envelope.initial_range_m,
        runway_heading_deg: envelope.heading_deg,
        ground_speed_mps: envelope.speed_mps,
        acceleration_mps2: envelope.acceleration_mps2,
        climb_rate_mps: envelope.climb_rate_mps,
        max_altitude_m: envelope.max_altitude_m,
        radial_velocity_bias_mps: envelope.radial_velocity_bias_mps,
        pitch_jitter_deg: envelope.attitude_jitter_deg,
        yaw_jitter_deg: envelope.attitude_jitter_deg,
        propulsor_hz: envelope.propulsor_hz,
        micro_doppler_hz: envelope.micro_doppler_hz,
        rcs_scalar: 10f64.powf(envelope.rcs_dbsm / 10.0).max(0.01),
        blade_count: None,
        blade_length_m: None,
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.awgn_sigma = 0.035 + 0.055 * envelope.clutter_profile.false_alarm_pressure();
    noise.rfi_probability = envelope.rfi_profile.burst_probability.min(0.12);
    noise.rfi_amplitude = envelope.rfi_profile.burst_amplitude;
    noise.clutter_sigma = 0.025 + 0.12 * envelope.clutter_profile.false_alarm_pressure();
    noise.ground_glint_count =
        (2.0f32 + 12.0f32 * envelope.clutter_profile.false_alarm_pressure()).round() as usize;

    let mut episode = synthesize_takeoff_episode(
        sim_config,
        profile,
        noise,
        EpisodeSeed(plan.seed ^ 0x05ee_dcaf_e136),
    );
    apply_campaign_stressors(&mut episode, &envelope, plan.seed);
    write_episode_tensors(&products_dir, &episode)?;

    let (features, first_detectable_frame) =
        build_frame_features(config, plan, &envelope, frame_count, cpi_pulses);
    let model_run = run_streaming_models(&plan.record_id, &features, config.trigger_confidence)?;
    let first_model_trigger_frame = model_run.events.iter().map(|event| event.frame_index).min();
    let labels = build_frame_labels(
        config,
        plan,
        &features,
        first_detectable_frame,
        first_model_trigger_frame,
    );
    let truth = RecordTruthMetadata {
        record_id: plan.record_id.clone(),
        neutral_object_id: plan.class.id.clone(),
        target_family: plan.class.target_family.clone(),
        is_shahed_public_proxy: plan.class.is_shahed_public_proxy,
        is_hard_negative: plan.class.is_hard_negative,
        hard_negative_family: plan.class.hard_negative_family.clone(),
        source_dossier_ref: super::SOURCE_DOSSIER_REF.to_string(),
        dimensions_m: envelope.dimensions_m.clone(),
        rcs_dbsm_proxy: envelope.rcs_dbsm,
        cruise_speed_mps: envelope.speed_mps,
        cpi_pulses,
        frame_count,
        time_window_s: config.time_window_s,
        first_detectable_frame,
        confidence_threshold: config.trigger_confidence,
        guardrails: campaign_guardrails(),
    };

    write_csv(&record_dir.join("streaming_features.csv"), &features)?;
    write_csv(&record_dir.join("frame_labels.csv"), &labels)?;
    write_json_pretty(&record_dir.join("truth_metadata.json"), &truth)?;
    write_json_pretty(&record_dir.join("detector_events.json"), &model_run.events)?;
    write_csv(
        &record_dir.join("model_predictions.csv"),
        &model_run.predictions,
    )?;
    write_json_pretty(
        &record_dir.join("runtime_stage.json"),
        &json!({
            "selected_backend": runtime.selected_backend.to_string(),
            "kernel_backend": "cpu-scaffold",
            "gpu_stage_recovery": gpu_stage_recovery_note(runtime),
        }),
    )?;

    Ok(CampaignRecordOutput {
        summary: CampaignRecordSummary {
            record_id: plan.record_id.clone(),
            record_index: plan.index,
            target_family: plan.class.target_family.clone(),
            class_id: plan.class.id.clone(),
            bucket: plan.class.bucket,
            is_shahed_public_proxy: plan.class.is_shahed_public_proxy,
            is_hard_negative: plan.class.is_hard_negative,
            hard_negative_family: plan.class.hard_negative_family.clone(),
            first_detectable_frame,
            first_model_trigger_frame,
            cpi_pulses,
            tensor_dir: relative_record_path(&plan.record_id, "products"),
            frame_labels_path: relative_record_path(&plan.record_id, "frame_labels.csv"),
            truth_metadata_path: relative_record_path(&plan.record_id, "truth_metadata.json"),
            model_predictions_path: relative_record_path(&plan.record_id, "model_predictions.csv"),
            detector_events_path: relative_record_path(&plan.record_id, "detector_events.json"),
            max_confidence_by_model: model_run.max_confidence_by_model,
            first_trigger_by_model: model_run.first_trigger_by_model,
        },
        events: model_run.events,
        predictions: model_run.predictions,
    })
}

fn relative_record_path(record_id: &str, suffix: &str) -> String {
    format!("records/{record_id}/{suffix}")
}
