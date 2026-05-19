//! Integration tests for the hard-negative mining loop.
//!
//! These tests build small synthetic fixtures on disk under `tempfile` roots
//! so they don't depend on the multi-megabyte `outputs/campaigns/...`
//! artifacts. The fixture shapes are byte-compatible with the real
//! `model_eval_*.json` and `airspace-objects.json` shapes — see the
//! `synthetic_model_eval` and `synthetic_airspace_library` helpers below.

use std::fs;
use std::path::Path;

use echoforge_hardneg_mining::{
    cluster::FailureClusterStore,
    loop_driver::{run_one_iteration, MiningLoopConfig},
    synthesize::{compute_widen_factor, synthesize_batch, synthesize_variant, WIDEN_MAX},
    types::{
        AirspaceObjectsConfig, KinematicsBounds, MicroMotionBounds, ModelEvalRollup, ObjectClass,
        SensorObservableBounds,
    },
    MiningError,
};
use serde_json::json;
use tempfile::TempDir;

fn synthetic_model_eval(
    model_id: &str,
    positive_records: usize,
    negative_records: usize,
    family_fps: &[(&str, usize)],
) -> ModelEvalRollup {
    let mut false_alarm = std::collections::BTreeMap::new();
    for (family, count) in family_fps {
        false_alarm.insert((*family).to_string(), *count);
    }
    ModelEvalRollup {
        model_id: model_id.to_string(),
        positive_records,
        negative_records,
        pd: 0.8,
        pfa: family_fps.iter().map(|(_, c)| *c).sum::<usize>() as f64
            / negative_records.max(1) as f64,
        missed_positive_records: Vec::new(),
        false_alarm_by_hard_negative_family: false_alarm,
        mean_first_detection_latency_frames: Some(3.0),
        extra: std::collections::BTreeMap::new(),
    }
}

fn write_model_eval_to_qa(qa_dir: &Path, rollup: &ModelEvalRollup) {
    fs::create_dir_all(qa_dir).expect("create qa");
    let path = qa_dir.join(format!("model_eval_{}.json", rollup.model_id));
    fs::write(&path, serde_json::to_string_pretty(rollup).unwrap()).expect("write");
}

fn synthetic_object_class(id: &str, family: &str) -> ObjectClass {
    ObjectClass {
        id: id.to_string(),
        display_name: format!("Synthetic {id}"),
        object_family: family.to_string(),
        role_tags: vec!["public_proxy".to_string(), "hard_negative".to_string()],
        dimensions_m: json!({
            "length": [1.0, 2.0],
            "wingspan": [1.0, 2.0],
            "height": [0.5, 1.0]
        }),
        rcs_dbsm: Some([-20.0, -5.0]),
        material_mix: json!({"composite": [0.1, 0.9]}),
        kinematics: KinematicsBounds {
            ground_speed_mps: Some([10.0, 30.0]),
            radial_velocity_mps: Some([-20.0, 20.0]),
            climb_rate_mps: Some([-2.0, 2.0]),
            max_altitude_m: Some([100.0, 500.0]),
            turn_rate_deg_s: Some([0.0, 10.0]),
            acceleration_mps2: Some([0.0, 2.0]),
            altitude_agl_m: Some([0.0, 500.0]),
            extra: Default::default(),
        },
        micro_motion: MicroMotionBounds {
            propulsor_hz: Some([10.0, 50.0]),
            micro_doppler_hz: Some([5.0, 40.0]),
            amplitude_modulation: Some([0.0, 0.2]),
            attitude_jitter_deg: Some([0.0, 2.0]),
            extra: Default::default(),
        },
        behavior: json!({"phases": ["transit"]}),
        sensor_observables: SensorObservableBounds {
            doppler_spread_bins: Some([1, 4]),
            scintillation_sigma: Some([0.05, 0.3]),
            classification_prior: Some(0.25),
            extra: Default::default(),
        },
        extra: Default::default(),
    }
}

fn synthetic_airspace_library() -> AirspaceObjectsConfig {
    AirspaceObjectsConfig {
        config_id: "synthetic-airspace".to_string(),
        display_name: "synthetic".to_string(),
        purpose: "tests".to_string(),
        guardrails: vec!["test fixture".to_string()],
        object_classes: vec![
            synthetic_object_class(
                "commercial-aircraft-corridor-clutter",
                "commercial_aircraft_corridor",
            ),
            synthetic_object_class("wind-turbine-industrial-glint", "infrastructure_glint"),
            synthetic_object_class("bird-flock-dense", "bird_flock"),
        ],
        extra: Default::default(),
    }
}

fn write_airspace_library(path: &Path, lib: &AirspaceObjectsConfig) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, serde_json::to_string_pretty(lib).unwrap()).unwrap();
}

#[test]
fn cluster_store_extracts_trouble_clusters_from_synthetic_eval() {
    // Two detectors, both flag the corridor heavily and one flags the glint
    // moderately. negative_records = 920 per detector.
    let r1 = synthetic_model_eval(
        "feature_tree_classifier",
        80,
        920,
        &[
            ("commercial_aircraft_corridor", 220),
            ("infrastructure_glint", 80),
        ],
    );
    let r2 = synthetic_model_eval(
        "cfar_tracker_baseline",
        80,
        920,
        &[("commercial_aircraft_corridor", 180)],
    );

    let store = FailureClusterStore::from_rollups(&[r1, r2], 0.15).expect("cluster store");
    // Aggregated: corridor has 400 FPs over 1840 neg evaluations = 0.217 >0.15.
    // Glint has 80 / 1840 = 0.043 <0.15 => filtered.
    assert_eq!(store.clusters.len(), 1);
    let corridor = &store.clusters[0];
    assert_eq!(corridor.class_id, "commercial_aircraft_corridor");
    assert_eq!(corridor.fp_count, 400);
    assert_eq!(corridor.frame_count, 1840);
    assert!((corridor.fp_rate - 0.217).abs() < 0.01);
    assert_eq!(corridor.contributing_models.len(), 2);
    // representative_features should contain both overall and per-model.
    assert!(corridor.representative_features.contains_key("fp_rate"));
    assert!(corridor
        .representative_features
        .contains_key("fp_rate__feature_tree_classifier"));
    assert!(corridor
        .representative_features
        .contains_key("fp_rate__cfar_tracker_baseline"));
}

#[test]
fn cluster_store_rejects_invalid_threshold() {
    let r = synthetic_model_eval("m", 1, 100, &[("x", 50)]);
    let one = std::slice::from_ref(&r);
    let err = FailureClusterStore::from_rollups(one, 0.0).unwrap_err();
    assert!(matches!(err, MiningError::InvalidThreshold { .. }));
    let err = FailureClusterStore::from_rollups(one, 1.5).unwrap_err();
    assert!(matches!(err, MiningError::InvalidThreshold { .. }));
    let err = FailureClusterStore::from_rollups(one, f64::NAN).unwrap_err();
    assert!(matches!(err, MiningError::InvalidThreshold { .. }));
}

#[test]
fn cluster_store_empty_when_zero_failures() {
    // No false alarms -> no clusters. Must not panic, must succeed.
    let r = synthetic_model_eval("m", 80, 920, &[]);
    let store = FailureClusterStore::from_rollups(&[r], 0.15).expect("store");
    assert!(store.clusters.is_empty());
    assert!((store.fp_threshold - 0.15).abs() < 1e-9);
}

#[test]
fn cluster_store_reads_from_disk_qa_dir() {
    let tmp = TempDir::new().unwrap();
    let campaign_root = tmp.path().join("campaign");
    let qa = campaign_root.join("qa");
    write_model_eval_to_qa(
        &qa,
        &synthetic_model_eval("m1", 80, 920, &[("commercial_aircraft_corridor", 200)]),
    );
    write_model_eval_to_qa(
        &qa,
        &synthetic_model_eval(
            "m2",
            80,
            920,
            &[("commercial_aircraft_corridor", 150), ("bird_flock", 200)],
        ),
    );
    // Sneak a non-rollup JSON file in to make sure the loader filters it out.
    fs::write(qa.join("something_else.json"), b"{}").unwrap();

    let store = FailureClusterStore::from_campaign_root(&campaign_root, 0.10).expect("disk load");
    // corridor: (200+150) FPs / (920+920) neg = 0.190.
    // bird_flock: 200 FPs / (920+920) neg = 0.109.
    // Both exceed 0.10 → both clusters; ordered by fp_rate desc.
    assert_eq!(store.clusters.len(), 2);
    assert_eq!(store.clusters[0].class_id, "commercial_aircraft_corridor");
    assert_eq!(store.clusters[1].class_id, "bird_flock");
    assert_eq!(store.source_rollups.len(), 2); // not 3 — the foreign JSON was skipped
}

#[test]
fn synthesize_variant_widens_envelopes_and_tags() {
    let base = synthetic_object_class(
        "commercial-aircraft-corridor-clutter",
        "commercial_aircraft_corridor",
    );
    let cluster = echoforge_hardneg_mining::FailureCluster {
        class_id: "commercial_aircraft_corridor".to_string(),
        frame_count: 1000,
        fp_count: 250,
        fp_rate: 0.25,
        sample_record_ids: Vec::new(),
        representative_features: Default::default(),
        contributing_models: vec!["m1".to_string()],
    };
    let delta = synthesize_variant(&cluster, &base, 0.15, 7);
    let v = &delta.variant;
    assert_eq!(v.id, "commercial-aircraft-corridor-clutter-mined-variant-7");
    assert!(v.role_tags.contains(&"mined_variant".to_string()));
    // Envelope must have widened (not shrunk).
    let base_doppler = base.micro_motion.micro_doppler_hz.unwrap();
    let mined_doppler = v.micro_motion.micro_doppler_hz.unwrap();
    let base_span = base_doppler[1] - base_doppler[0];
    let mined_span = mined_doppler[1] - mined_doppler[0];
    assert!(mined_span >= base_span, "{mined_span} >= {base_span}");
    // Lower bound must be clamped >= 0 (non-negative widen).
    assert!(mined_doppler[0] >= 0.0);
    // RCS upper bound should have lifted by the dB bump.
    let mined_rcs = v.rcs_dbsm.unwrap();
    assert!(mined_rcs[1] > base.rcs_dbsm.unwrap()[1]);
    // Provenance recorded.
    let prov = v.extra.get("mined_provenance").expect("provenance");
    assert_eq!(
        prov["base_class_id"],
        "commercial-aircraft-corridor-clutter"
    );
}

#[test]
fn synthesize_batch_caps_at_max_variants_and_skips_unmapped() {
    let base_classes = synthetic_airspace_library().object_classes;
    let clusters = vec![
        echoforge_hardneg_mining::FailureCluster {
            class_id: "commercial_aircraft_corridor".to_string(),
            frame_count: 1000,
            fp_count: 250,
            fp_rate: 0.25,
            sample_record_ids: vec![],
            representative_features: Default::default(),
            contributing_models: vec![],
        },
        echoforge_hardneg_mining::FailureCluster {
            class_id: "infrastructure_glint".to_string(),
            frame_count: 1000,
            fp_count: 200,
            fp_rate: 0.20,
            sample_record_ids: vec![],
            representative_features: Default::default(),
            contributing_models: vec![],
        },
        echoforge_hardneg_mining::FailureCluster {
            class_id: "bird_flock".to_string(),
            frame_count: 1000,
            fp_count: 180,
            fp_rate: 0.18,
            sample_record_ids: vec![],
            representative_features: Default::default(),
            contributing_models: vec![],
        },
        // No base class for this one — should be skipped, not error.
        echoforge_hardneg_mining::FailureCluster {
            class_id: "phantom_clutter_family_with_no_base".to_string(),
            frame_count: 100,
            fp_count: 50,
            fp_rate: 0.5,
            sample_record_ids: vec![],
            representative_features: Default::default(),
            contributing_models: vec![],
        },
    ];

    let deltas = synthesize_batch(&clusters, &base_classes, 0.15, 2, 0);
    assert_eq!(deltas.len(), 2, "must respect max_variants_per_iteration");
    assert_eq!(deltas[0].variant_index, 0);
    assert_eq!(deltas[1].variant_index, 1);
    // Unmapped phantom cluster must NOT appear.
    assert!(deltas
        .iter()
        .all(|d| d.base_class_id != "phantom_clutter_family_with_no_base"));
}

#[test]
fn compute_widen_factor_clamps_within_bounds() {
    // Very small headroom -> WIDEN_MIN.
    let w = compute_widen_factor(0.151, 0.15, 1.5);
    assert!((w - echoforge_hardneg_mining::WIDEN_MIN).abs() < 1e-6);
    // Very large headroom -> WIDEN_MAX.
    let w = compute_widen_factor(1.0, 0.0, 100.0);
    assert!((w - WIDEN_MAX).abs() < 1e-6);
    // At threshold -> WIDEN_MIN.
    let w = compute_widen_factor(0.15, 0.15, 1.5);
    assert!((w - echoforge_hardneg_mining::WIDEN_MIN).abs() < 1e-6);
}

#[test]
fn run_one_iteration_writes_expected_outputs() {
    let tmp = TempDir::new().unwrap();
    let campaign = tmp.path().join("campaign");
    let qa = campaign.join("qa");
    write_model_eval_to_qa(
        &qa,
        &synthetic_model_eval(
            "feature_tree_classifier",
            80,
            920,
            &[
                ("commercial_aircraft_corridor", 220),
                ("infrastructure_glint", 80),
            ],
        ),
    );
    write_model_eval_to_qa(
        &qa,
        &synthetic_model_eval(
            "cfar_tracker_baseline",
            80,
            920,
            &[("commercial_aircraft_corridor", 180)],
        ),
    );
    let airspace_path = tmp.path().join("airspace.json");
    write_airspace_library(&airspace_path, &synthetic_airspace_library());
    let output_dir = tmp.path().join("hard-negative-runs");

    let cfg = MiningLoopConfig {
        campaign_root: campaign,
        airspace_objects_path: airspace_path,
        fp_threshold: 0.15,
        max_variants_per_iteration: 4,
        output_dir: output_dir.clone(),
        iteration_label: Some("ut-iter-1".to_string()),
        iteration_number: 1,
    };
    let report = run_one_iteration(&cfg).expect("run iteration");

    assert_eq!(report.iteration_number, 1);
    assert!(report.cluster_count >= 1);
    assert_eq!(report.variant_count, report.cluster_count.min(4));
    assert!(report.clusters_path.exists());
    assert!(report.variants_path.exists());
    assert!(report.manifest_path.exists());

    // Read back the manifest and check fields.
    let manifest_text = fs::read_to_string(&report.manifest_path).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).unwrap();
    assert_eq!(manifest["iteration_number"], 1);
    assert_eq!(manifest["fp_threshold"], 0.15);
    assert_eq!(manifest["max_variants_per_iteration"], 4);
    assert_eq!(manifest["clusters_filename"], "failure_clusters.json");

    // Variants file must contain the expected outer shape and a derived id.
    let variants_text = fs::read_to_string(&report.variants_path).unwrap();
    let variants_doc: serde_json::Value = serde_json::from_str(&variants_text).unwrap();
    let variants = variants_doc["variants"].as_array().unwrap();
    assert!(!variants.is_empty());
    let first_id = variants[0]["variant"]["id"].as_str().unwrap();
    assert!(first_id.contains("-mined-variant-"), "id={first_id}");
    // iteration_number=1 means starting_index=100.
    assert!(first_id.ends_with("-mined-variant-100"), "id={first_id}");
}

#[test]
fn run_one_iteration_handles_zero_failures_without_panic() {
    let tmp = TempDir::new().unwrap();
    let campaign = tmp.path().join("campaign");
    let qa = campaign.join("qa");
    // Zero false alarms -> zero clusters.
    write_model_eval_to_qa(&qa, &synthetic_model_eval("m1", 80, 920, &[]));
    let airspace_path = tmp.path().join("airspace.json");
    write_airspace_library(&airspace_path, &synthetic_airspace_library());

    let cfg = MiningLoopConfig {
        campaign_root: campaign,
        airspace_objects_path: airspace_path,
        fp_threshold: 0.15,
        max_variants_per_iteration: 4,
        output_dir: tmp.path().join("runs"),
        iteration_label: None,
        iteration_number: 0,
    };
    let report = run_one_iteration(&cfg).expect("zero-failure iteration must succeed");
    assert_eq!(report.cluster_count, 0);
    assert_eq!(report.variant_count, 0);
    assert!(report.no_failures);
    assert!(report.clusters_path.exists());
    assert!(report.variants_path.exists());
    assert!(report.manifest_path.exists());
}

/// Optional smoke test against the real on-disk campaign artifacts. Marked
/// `#[ignore]` because `outputs/` is gitignored and not guaranteed to be
/// present on CI — run locally with:
///   `cargo test -p echoforge-hardneg-mining -- --ignored --nocapture real_campaign_smoke`
#[test]
#[ignore]
fn real_campaign_smoke() {
    let campaign =
        std::path::PathBuf::from("../../outputs/campaigns/shahed136-public-proxy-early-detection");
    if !campaign.exists() {
        eprintln!(
            "skipping: real campaign not present at {}",
            campaign.display()
        );
        return;
    }
    let airspace = std::path::PathBuf::from("../../configs/monte-carlo/airspace-objects.json");
    let out = TempDir::new().unwrap();
    let cfg = MiningLoopConfig {
        campaign_root: campaign,
        airspace_objects_path: airspace,
        // Real-campaign FP rates are small relative to 920 negatives × 3
        // detectors; use a low threshold so the smoke test demonstrates
        // cluster extraction. Production runs would keep the default 0.15
        // for an early-bench dataset and tighten over time.
        fp_threshold: 0.001,
        max_variants_per_iteration: 4,
        output_dir: out.path().to_path_buf(),
        iteration_label: Some("smoke".to_string()),
        iteration_number: 0,
    };
    let report = run_one_iteration(&cfg).expect("real-campaign run");
    eprintln!(
        "clusters={} variants={}",
        report.cluster_count, report.variant_count
    );
    let clusters_text = fs::read_to_string(&report.clusters_path).unwrap();
    eprintln!("\nfailure_clusters.json:\n{}", clusters_text);
}

#[test]
fn missing_campaign_root_is_a_typed_error() {
    let tmp = TempDir::new().unwrap();
    let airspace_path = tmp.path().join("airspace.json");
    write_airspace_library(&airspace_path, &synthetic_airspace_library());
    let cfg = MiningLoopConfig {
        campaign_root: tmp.path().join("does-not-exist"),
        airspace_objects_path: airspace_path,
        fp_threshold: 0.15,
        max_variants_per_iteration: 4,
        output_dir: tmp.path().join("runs"),
        iteration_label: None,
        iteration_number: 0,
    };
    let err = run_one_iteration(&cfg).unwrap_err();
    assert!(matches!(err, MiningError::MissingCampaignRoot { .. }));
}
