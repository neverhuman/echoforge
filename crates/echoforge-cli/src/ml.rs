use clap::{Args, Subcommand};
use std::path::PathBuf;

use echoforge_dataset::{
    inspect_pipeline, list_pipelines, run_pipeline, run_suite, PipelineRunRequest,
    PipelineRunResult, DEFAULT_DATA_ROOT, DEFAULT_OUT_ROOT, DEFAULT_VALIDATION_TIER,
    MAX_PIPELINE_WORKERS, MAX_SUITE_CONCURRENCY,
};

#[derive(Debug, Args)]
pub struct MlArgs {
    #[arg(long, value_name = "PATH")]
    pub repo_root: Option<PathBuf>,

    #[command(subcommand)]
    pub command: MlCommand,
}

#[derive(Debug, Subcommand)]
pub enum MlCommand {
    /// List discoverable ML pipelines.
    List,
    /// Inspect one pipeline spec.
    Inspect(MlInspectArgs),
    /// Run one pipeline end-to-end.
    Run(MlRunArgs),
    /// Run the full evidence ladder suite.
    RunSuite(MlSuiteArgs),
}

#[derive(Debug, Args)]
pub struct MlInspectArgs {
    #[arg(value_name = "PIPELINE_ID")]
    pub pipeline: Option<String>,
}

#[derive(Debug, Args)]
pub struct MlRunArgs {
    #[arg(long)]
    pub pipeline: String,

    #[arg(long, value_name = "PATH", default_value = DEFAULT_DATA_ROOT)]
    pub data_root: PathBuf,

    #[arg(long, value_name = "PATH", default_value = DEFAULT_OUT_ROOT)]
    pub out_root: PathBuf,

    #[arg(long, default_value_t = MAX_PIPELINE_WORKERS)]
    pub workers: usize,

    #[arg(long, default_value_t = 20_260_520_390_001)]
    pub seed: u64,

    #[arg(long)]
    pub smoke: bool,

    #[arg(long = "validation-tier", default_value = DEFAULT_VALIDATION_TIER)]
    pub validation_tier: String,
}

#[derive(Debug, Args)]
pub struct MlSuiteArgs {
    #[arg(long, default_value = "evidence-ladder-v1")]
    pub suite: String,

    #[arg(long, value_name = "PATH", default_value = DEFAULT_DATA_ROOT)]
    pub data_root: PathBuf,

    #[arg(long, value_name = "PATH", default_value = DEFAULT_OUT_ROOT)]
    pub out_root: PathBuf,

    #[arg(long = "workers-per-pipeline", default_value_t = MAX_PIPELINE_WORKERS)]
    pub workers_per_pipeline: usize,

    #[arg(long = "max-concurrent", default_value_t = MAX_SUITE_CONCURRENCY)]
    pub max_concurrent: usize,

    #[arg(long, default_value_t = 20_260_520_390_001)]
    pub seed: u64,

    #[arg(long)]
    pub smoke: bool,

    #[arg(long = "validation-tier", default_value = DEFAULT_VALIDATION_TIER)]
    pub validation_tier: String,
}

pub fn run_ml(args: MlArgs) -> Result<u8, String> {
    let MlArgs { repo_root, command } = args;
    match command {
        MlCommand::List => run_list(repo_root),
        MlCommand::Inspect(args) => run_inspect(args, repo_root),
        MlCommand::Run(args) => run_run(args, repo_root),
        MlCommand::RunSuite(args) => run_suite_cmd(args, repo_root),
    }
}

fn run_list(repo_root: Option<PathBuf>) -> Result<u8, String> {
    let specs = list_pipelines(repo_root).map_err(|err| err.to_string())?;
    println!("EchoForge ML pipelines");
    for spec in specs {
        println!(
            "- {} | {} | {} | worker_budget={} | research_only={}",
            spec.id, spec.version, spec.title, spec.worker_budget, spec.research_only
        );
    }
    Ok(0)
}

fn run_inspect(args: MlInspectArgs, repo_root: Option<PathBuf>) -> Result<u8, String> {
    let pipeline_id = args
        .pipeline
        .unwrap_or_else(|| "physics_cfar_track_fusion_v1".to_string());
    let spec = inspect_pipeline(&pipeline_id, repo_root).map_err(|err| err.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&spec).map_err(|err| err.to_string())?
    );
    Ok(0)
}

fn run_run(args: MlRunArgs, repo_root: Option<PathBuf>) -> Result<u8, String> {
    let request = PipelineRunRequest {
        pipeline_id: args.pipeline,
        data_root: args.data_root,
        out_root: args.out_root,
        workers: args.workers,
        seed: args.seed,
        smoke: args.smoke,
        validation_tier: args.validation_tier,
        repo_root,
    };
    let result = run_pipeline(request).map_err(|err| err.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|err| err.to_string())?
    );
    Ok(exit_code_for_result(&result))
}

fn run_suite_cmd(args: MlSuiteArgs, repo_root: Option<PathBuf>) -> Result<u8, String> {
    let result = run_suite(
        &args.suite,
        repo_root,
        args.data_root,
        args.out_root,
        args.workers_per_pipeline,
        args.max_concurrent,
        args.seed,
        args.smoke,
        args.validation_tier,
    )
    .map_err(|err| err.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|err| err.to_string())?
    );
    Ok(
        if result.results.iter().any(|item| item.status != "completed") {
            1
        } else {
            0
        },
    )
}

fn exit_code_for_result(result: &PipelineRunResult) -> u8 {
    if result.status == "completed" {
        0
    } else {
        1
    }
}
