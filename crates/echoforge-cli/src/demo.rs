use clap::{Args, Subcommand};
use echoforge_radar::BackendMode;
use std::path::PathBuf;

#[derive(Debug, Args)]
pub struct DemoArgs {
    #[command(subcommand)]
    pub command: DemoCommand,
}

#[derive(Debug, Subcommand)]
pub enum DemoCommand {
    /// Generate a reproducible public-proxy Monte Carlo radar dataset.
    MonteCarlo(MonteCarloArgs),
    /// Generate a reproducible ML training dataset.
    MlTrainingData(MlTrainingDataArgs),
}

#[derive(Debug, Args)]
pub struct MonteCarloArgs {
    #[arg(long, default_value = "low-altitude-fixed-wing-takeoff-v1")]
    pub preset: String,

    #[arg(long, default_value_t = 32)]
    pub episodes: usize,

    #[arg(long, default_value_t = 20_260_518)]
    pub seed: u64,

    #[arg(long = "generated-at", default_value = "2026-05-18T00:00:00Z")]
    pub generated_at: String,

    #[arg(long = "out", value_name = "DIR")]
    pub out: PathBuf,

    #[arg(long = "noise-profile", default_value = "real-world-proxy-v1")]
    pub noise_profile: String,

    #[arg(long = "sample-rate-hz", default_value_t = 2_000_000.0)]
    pub sample_rate_hz: f64,

    #[arg(long = "pulse-count", default_value_t = 32)]
    pub pulse_count: usize,
}

#[derive(Debug, Args)]
pub struct MlTrainingDataArgs {
    #[arg(long, default_value = "best-final-scenario-v1")]
    pub scenario: String,

    #[arg(long)]
    pub smoke: bool,

    #[arg(long = "out", value_name = "DIR")]
    pub out: Option<PathBuf>,

    #[arg(long, default_value_t = 20_260_520_390_001)]
    pub seed: u64,

    #[arg(long = "generated-at", default_value = "2026-05-20T00:00:00Z")]
    pub generated_at: String,

    #[arg(long, default_value = "cpu")]
    pub backend: String,

    #[arg(long)]
    pub workers: Option<usize>,

    #[arg(long = "time-window-s")]
    pub time_window_s: Option<f64>,

    #[arg(long = "frame-rate-hz")]
    pub frame_rate_hz: Option<f64>,
}

pub fn run_demo(args: DemoArgs) -> Result<u8, String> {
    match args.command {
        DemoCommand::MonteCarlo(args) => run_monte_carlo(args),
        DemoCommand::MlTrainingData(args) => run_ml_training_data(args),
    }
}

fn run_monte_carlo(args: MonteCarloArgs) -> Result<u8, String> {
    let known = echoforge_dataset::known_presets().map_err(|err| err.to_string())?;
    if !known.iter().any(|preset| preset == &args.preset) {
        return Err(format!(
            "unknown preset {}; known presets: {}",
            args.preset,
            known.join(", ")
        ));
    }

    let config = echoforge_dataset::MonteCarloDemoConfig {
        preset: args.preset,
        episodes: args.episodes,
        seed: args.seed,
        generated_at: args.generated_at,
        output_dir: args.out,
        target_label: "Iranian Public-Proxy Fixed-Wing UAV Takeoff".to_string(),
        noise_profile: args.noise_profile,
        sample_rate_hz: args.sample_rate_hz,
        pulse_count: args.pulse_count,
        runtime: echoforge_dataset::MonteCarloRuntimePolicy::default(),
    };
    let report = echoforge_dataset::run_monte_carlo_demo(config).map_err(|err| err.to_string())?;

    println!("EchoForge Monte Carlo demo receipt");
    println!("output_dir: {}", report.output_dir.display());
    println!("preset: {}", report.preset);
    println!("episodes: {}", report.episode_count);
    println!(
        "splits: train={}, validation={}, test={}",
        report
            .split_counts
            .get(&echoforge_dataset::SplitKind::Train)
            .unwrap_or(&0),
        report
            .split_counts
            .get(&echoforge_dataset::SplitKind::Validation)
            .unwrap_or(&0),
        report
            .split_counts
            .get(&echoforge_dataset::SplitKind::Test)
            .unwrap_or(&0)
    );
    println!(
        "leakage_status: {}",
        if report.leakage_clean {
            "clean"
        } else {
            "findings"
        }
    );
    println!("validation_status: {}", report.validation_status);
    println!(
        "limitation: public-proxy statistical noise proxy; not measured truth; not proprietary-equivalent"
    );

    Ok(0)
}

fn run_ml_training_data(args: MlTrainingDataArgs) -> Result<u8, String> {
    let mut config = match args.scenario.as_str() {
        "best-final-scenario-v1" if args.smoke => {
            echoforge_dataset::MlTrainingDataConfig::best_final_smoke()
        }
        "best-final-scenario-v1" => echoforge_dataset::MlTrainingDataConfig::best_final_default(),
        "shahed136-public-proxy-ml-training-v1" if !args.smoke => {
            echoforge_dataset::MlTrainingDataConfig::shahed_public_proxy_default()
        }
        "shahed136-public-proxy-ml-training-v1" => {
            let mut config = echoforge_dataset::MlTrainingDataConfig::shahed_public_proxy_default();
            config.records = 30;
            config.positive_fraction = 0.20;
            config.time_window_s = 6.0;
            config.frame_rate_hz = 2.0;
            config.backend = BackendMode::Cpu;
            config.workers = Some(4);
            config.output_dir =
                PathBuf::from("outputs/training-data/shahed136-public-proxy-ml-training-smoke");
            config
        }
        other => {
            return Err(format!(
                "unknown ML training scenario {other}; expected best-final-scenario-v1 or shahed136-public-proxy-ml-training-v1"
            ));
        }
    };

    config.seed = args.seed;
    config.generated_at = args.generated_at;
    config.backend = parse_backend(&args.backend)?;
    if let Some(workers) = args.workers {
        config.workers = Some(workers);
    }
    if let Some(time_window_s) = args.time_window_s {
        config.time_window_s = time_window_s;
    }
    if let Some(frame_rate_hz) = args.frame_rate_hz {
        config.frame_rate_hz = frame_rate_hz;
    }
    if let Some(out) = args.out {
        config.output_dir = out;
    }

    let report = echoforge_dataset::run_ml_training_data(config).map_err(|err| err.to_string())?;

    println!("EchoForge ML training-data receipt");
    println!("output_dir: {}", report.output_dir.display());
    println!("dataset_id: {}", report.dataset_id);
    println!("records: {}", report.records);
    println!("positive_records: {}", report.positive_records);
    println!("hard_negative_families: {}", report.hard_negative_families);
    println!("frame_count: {}", report.frame_count);
    println!("worker_count: {}", report.worker_count);
    println!("runtime_backend: {}", report.runtime.selected_backend);
    println!("records_csv: {}", report.records_path.display());
    println!("features_csv: {}", report.features_path.display());
    println!("dataset_card: {}", report.dataset_card_path.display());
    println!("quality_report: {}", report.quality_report_path.display());
    println!("per_tier_pd_pfa: {}", report.per_tier_pd_pfa_path.display());
    println!(
        "limitation: synthetic public-proxy benchmark data only; not measured truth; not proprietary-equivalent"
    );

    Ok(0)
}

fn parse_backend(input: &str) -> Result<BackendMode, String> {
    match input {
        "auto" => Ok(BackendMode::Auto),
        "cpu" => Ok(BackendMode::Cpu),
        "gpu" => Ok(BackendMode::Gpu),
        other => Err(format!(
            "unknown backend {other}; expected auto, cpu, or gpu"
        )),
    }
}
