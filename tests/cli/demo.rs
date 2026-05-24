use clap::Parser;

#[test]
fn demo_monte_carlo_args_are_parseable() {
    let cli = echoforge_cli::cli::Cli::try_parse_from([
        "ef",
        "demo",
        "monte-carlo",
        "--preset",
        "low-altitude-fixed-wing-takeoff-v1",
        "--episodes",
        "2",
        "--seed",
        "20260518",
        "--generated-at",
        "2026-05-18T00:00:00Z",
        "--out",
        "outputs/demo/low-altitude-fixed-wing-takeoff-v1",
    ]);

    assert!(cli.is_ok());
}

#[test]
fn demo_ml_training_args_are_parseable() {
    let cli = echoforge_cli::cli::Cli::try_parse_from([
        "ef",
        "demo",
        "ml-training-data",
        "--scenario",
        "best-final-scenario-v1",
        "--smoke",
        "--backend",
        "cpu",
        "--workers",
        "2",
        "--out",
        "outputs/training-data/best-final-scenario-v1-smoke",
    ]);

    assert!(cli.is_ok());
}

#[test]
fn invalid_episode_count_fails() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out = temp.path().join("bad");
    let args = vec![
        "ef".into(),
        "demo".into(),
        "monte-carlo".into(),
        "--episodes".into(),
        "0".into(),
        "--out".into(),
        out.into_os_string(),
    ];

    let result = echoforge_cli::run(args);
    assert!(result.is_err());
    assert!(result
        .expect_err("expected invalid episodes")
        .contains("episodes must be in the range"));
}

#[test]
fn forbidden_output_under_source_path_fails() {
    let result = echoforge_cli::run(vec![
        "ef".into(),
        "demo".into(),
        "monte-carlo".into(),
        "--episodes".into(),
        "1".into(),
        "--out".into(),
        "crates/generated-demo".into(),
    ]);

    assert!(result.is_err());
    assert!(result
        .expect_err("expected forbidden output")
        .contains("refusing output path"));
}

#[test]
fn small_tempdir_run_succeeds() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out = temp.path().join("run");
    let args = vec![
        "ef".into(),
        "demo".into(),
        "monte-carlo".into(),
        "--episodes".into(),
        "2".into(),
        "--pulse-count".into(),
        "6".into(),
        "--seed".into(),
        "20260518".into(),
        "--generated-at".into(),
        "2026-05-18T00:00:00Z".into(),
        "--out".into(),
        out.clone().into_os_string(),
    ];

    let code = echoforge_cli::run(args).expect("run succeeds");
    assert_eq!(code, 0);
    assert!(out.join("dataset_card.json").exists());
    assert!(out
        .join("episodes/episode_000001/radar_episode.json")
        .exists());
}

#[test]
fn small_ml_training_smoke_run_succeeds() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out = temp.path().join("ml-training");
    let args = vec![
        "ef".into(),
        "demo".into(),
        "ml-training-data".into(),
        "--scenario".into(),
        "best-final-scenario-v1".into(),
        "--smoke".into(),
        "--backend".into(),
        "cpu".into(),
        "--workers".into(),
        "2".into(),
        "--time-window-s".into(),
        "4".into(),
        "--frame-rate-hz".into(),
        "1".into(),
        "--out".into(),
        out.clone().into_os_string(),
    ];

    let code = echoforge_cli::run(args).expect("run succeeds");
    assert_eq!(code, 0);
    assert!(out.join("records.csv").exists());
    assert!(out.join("features.csv").exists());
    assert!(out.join("dataset_card.json").exists());
    assert!(out.join("per_tier_pd_pfa.json").exists());
}
