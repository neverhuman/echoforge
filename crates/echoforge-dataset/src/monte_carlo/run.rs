use std::collections::BTreeMap;
use std::fs;
use std::thread;
use std::time::Instant;

use echoforge_radar::{BackendSignals, RuntimePlan};

use crate::export::{write_episode_tensors, write_json_pretty, write_text};
use crate::guard::guard_output_dir;
use crate::leakage::{build_leakage_report, LeakageReport};
use crate::split::{assign_split, DatasetRecord, SplitKind};

use super::config::{
    MonteCarloBenchmarkReport, MonteCarloDemoConfig, MonteCarloDemoReport, StageTiming,
    DEFAULT_NOISE_PROFILE,
};
use super::error::DatasetError;
use super::helpers::{
    build_benchmark_report, child_seed, dataset_card_model, elapsed_ns, records_jsonl,
    EpisodeManifestEntry, SplitManifest, SplitMix64,
};
use super::sampling::{
    radar_episode_model, sample_episode, synthesize_episode, write_episode_json_products,
    write_static_cards, ResolvedPreset,
};
use super::scene_config::embedded_airspace_config;

#[path = "run_report.rs"]
mod run_report;
use run_report::{monte_carlo_split_policy, run_manifest};

// ── public entry point ────────────────────────────────────────────────────────

pub fn run_monte_carlo_demo(
    config: MonteCarloDemoConfig,
) -> Result<MonteCarloDemoReport, DatasetError> {
    validate_run_config(&config)?;
    guard_output_dir(&config.output_dir)?;

    let overall_start = Instant::now();
    let library = embedded_airspace_config()?;
    let resolved = ResolvedPreset::resolve(&library, &config.preset)?;
    let runtime_probe = BackendSignals::detect();
    let runtime = RuntimePlan::from_signals(config.runtime.backend, runtime_probe)?;
    fs::create_dir_all(&config.output_dir)?;

    let mut stage_timings = Vec::new();
    let static_cards_start = Instant::now();
    write_static_cards(&config, &resolved)?;
    stage_timings.push(StageTiming {
        stage: "static_cards".to_string(),
        elapsed_ns: elapsed_ns(static_cards_start),
    });

    let policy = monte_carlo_split_policy();
    let worker_count = runtime
        .recommended_worker_budget
        .min(config.episodes.max(1));
    let episode_generation_start = Instant::now();
    let mut episode_outputs = run_episode_workers(&config, &resolved, worker_count)?;
    episode_outputs.sort_by_key(|entry| entry.index);
    stage_timings.push(StageTiming {
        stage: "episode_generation".to_string(),
        elapsed_ns: elapsed_ns(episode_generation_start),
    });

    let postprocess_start = Instant::now();
    let mut records = Vec::with_capacity(config.episodes);
    let mut episode_manifests = Vec::with_capacity(config.episodes);
    let mut split_counts: BTreeMap<SplitKind, usize> = BTreeMap::new();

    for output in episode_outputs {
        let EpisodeOutcome {
            index,
            record,
            split,
            manifest,
        } = output;
        let _ = index;
        *split_counts.entry(split).or_insert(0) += 1;
        records.push(record);
        episode_manifests.push(manifest);
    }

    let leakage_report = build_leakage_report(&records, &policy);
    write_text(
        &config.output_dir.join("records.jsonl"),
        &records_jsonl(&records)?,
    )?;
    write_json_pretty(
        &config.output_dir.join("split_manifest.json"),
        &SplitManifest {
            policy: policy.clone(),
            counts: split_counts.clone(),
            records: records.clone(),
        },
    )?;
    write_json_pretty(
        &config.output_dir.join("leakage_report.json"),
        &leakage_report,
    )?;

    let dataset_card = dataset_card_model(
        &config,
        &resolved.object.id,
        &resolved.preset.id,
        &split_counts,
    )?;
    write_json_pretty(&config.output_dir.join("dataset_card.json"), &dataset_card)?;

    let manifest = run_manifest(
        &config,
        &library,
        &resolved,
        &episode_manifests,
        &leakage_report,
    );
    write_json_pretty(&config.output_dir.join("manifest.json"), &manifest)?;
    stage_timings.push(StageTiming {
        stage: "postprocess".to_string(),
        elapsed_ns: elapsed_ns(postprocess_start),
    });

    let benchmark = build_benchmark_report(
        &config,
        &runtime,
        worker_count,
        &stage_timings,
        overall_start.elapsed(),
    );
    let benchmark_report_path = if config.runtime.benchmark {
        let path = config.output_dir.join("benchmark_report.json");
        write_json_pretty(&path, &benchmark)?;
        Some(path)
    } else {
        None
    };
    write_text(
        &config.output_dir.join("benchmark_report.md"),
        &benchmark_report_markdown(
            &config,
            &resolved,
            &split_counts,
            &leakage_report,
            &benchmark,
        ),
    )?;

    Ok(MonteCarloDemoReport {
        output_dir: config.output_dir.clone(),
        preset: config.preset,
        episode_count: config.episodes,
        split_counts,
        leakage_clean: leakage_report.is_clean(),
        validation_status: "pass".to_string(),
        manifest_path: config.output_dir.join("manifest.json"),
        dataset_card_path: config.output_dir.join("dataset_card.json"),
        runtime,
        benchmark_report_path,
    })
}

// ── config validation ─────────────────────────────────────────────────────────

fn validate_run_config(config: &MonteCarloDemoConfig) -> Result<(), DatasetError> {
    if !(1..=10_000).contains(&config.episodes) {
        return Err(DatasetError::InvalidConfig(
            "episodes must be in the range 1..=10000".to_string(),
        ));
    }
    if config.generated_at.trim().is_empty()
        || !config.generated_at.contains('T')
        || !config.generated_at.ends_with('Z')
    {
        return Err(DatasetError::InvalidConfig(
            "generated_at must be an RFC3339-like UTC timestamp ending in Z".to_string(),
        ));
    }
    if config.noise_profile != DEFAULT_NOISE_PROFILE {
        return Err(DatasetError::InvalidConfig(format!(
            "unknown noise profile {}; expected {DEFAULT_NOISE_PROFILE}",
            config.noise_profile
        )));
    }
    if config.sample_rate_hz <= 0.0 || config.pulse_count == 0 {
        return Err(DatasetError::InvalidConfig(
            "sample_rate_hz and pulse_count must be positive".to_string(),
        ));
    }
    Ok(())
}

// ── episode worker threads ────────────────────────────────────────────────────

#[derive(Debug)]
struct EpisodeOutcome {
    index: usize,
    split: SplitKind,
    record: DatasetRecord,
    manifest: EpisodeManifestEntry,
}

fn run_episode_workers(
    config: &MonteCarloDemoConfig,
    resolved: &ResolvedPreset<'_>,
    worker_count: usize,
) -> Result<Vec<EpisodeOutcome>, DatasetError> {
    let worker_count = worker_count.max(1).min(config.episodes.max(1));
    let chunk_size = (config.episodes + worker_count - 1) / worker_count;

    thread::scope(|scope| -> Result<Vec<EpisodeOutcome>, DatasetError> {
        let mut handles = Vec::new();
        for chunk_start in (0..config.episodes).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(config.episodes);
            handles.push(
                scope.spawn(move || run_episode_range(config, resolved, chunk_start, chunk_end)),
            );
        }

        let mut outputs = Vec::with_capacity(config.episodes);
        for handle in handles {
            let mut chunk = handle.join().map_err(|_| {
                DatasetError::InvalidConfig("episode worker panicked".to_string())
            })??;
            outputs.append(&mut chunk);
        }
        Ok(outputs)
    })
}

fn run_episode_range(
    config: &MonteCarloDemoConfig,
    resolved: &ResolvedPreset<'_>,
    start: usize,
    end: usize,
) -> Result<Vec<EpisodeOutcome>, DatasetError> {
    let mut outputs = Vec::with_capacity(end.saturating_sub(start));
    for index in start..end {
        outputs.push(run_episode(config, resolved, index)?);
    }
    Ok(outputs)
}

fn run_episode(
    config: &MonteCarloDemoConfig,
    resolved: &ResolvedPreset<'_>,
    index: usize,
) -> Result<EpisodeOutcome, DatasetError> {
    let episode_seed = child_seed(config.seed, index as u64);
    let mut rng = SplitMix64::new(episode_seed);
    let episode_id = format!("episode_{:06}", index + 1);
    let episode_dir = config.output_dir.join("episodes").join(&episode_id);
    let products_dir = episode_dir.join("products");
    fs::create_dir_all(&products_dir)?;

    let sampled = sample_episode(config, resolved, &mut rng, episode_seed);
    let episode = synthesize_episode(&sampled, episode_seed);

    write_episode_tensors(&products_dir, &episode)?;
    write_episode_json_products(&products_dir, &episode, &sampled)?;

    let mut record = DatasetRecord {
        sample_id: episode_id.clone(),
        split_hint: None,
        object_family: resolved.object.object_family.clone(),
        geometry_hash: format!("geometry-{:016x}", child_seed(episode_seed, 1)),
        material_sample_hash: format!("material-{:016x}", child_seed(episode_seed, 2)),
        scenario_seed: episode_seed,
        sensor_archetype: resolved.sensor.id.clone(),
        hard_negative_family: resolved.environment.id.clone(),
    };
    let policy = monte_carlo_split_policy();
    let split = assign_split(&record, &policy);
    record.split_hint = Some(split);

    let radar_episode = radar_episode_model(config, &episode_id, &sampled)?;
    write_json_pretty(&episode_dir.join("radar_episode.json"), &radar_episode)?;

    Ok(EpisodeOutcome {
        index,
        split,
        record,
        manifest: EpisodeManifestEntry {
            episode_id,
            seed: episode_seed,
            split,
            path: format!("episodes/episode_{:06}/radar_episode.json", index + 1),
            detections: episode.detections.len(),
            target_label: config.target_label.clone(),
        },
    })
}

// ── manifest + report builders ────────────────────────────────────────────────

fn benchmark_report_markdown(
    config: &MonteCarloDemoConfig,
    resolved: &ResolvedPreset<'_>,
    split_counts: &BTreeMap<SplitKind, usize>,
    leakage_report: &LeakageReport,
    benchmark: &MonteCarloBenchmarkReport,
) -> String {
    run_report::benchmark_report_markdown(config, resolved, split_counts, leakage_report, benchmark)
}
