use clap::{Args, Subcommand};
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

pub fn run_demo(args: DemoArgs) -> Result<u8, String> {
    match args.command {
        DemoCommand::MonteCarlo(args) => run_monte_carlo(args),
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
