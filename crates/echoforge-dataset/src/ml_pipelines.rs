use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MAX_PIPELINE_WORKERS: usize = 20;
pub const MAX_SUITE_CONCURRENCY: usize = 3;
pub const DEFAULT_DATA_ROOT: &str = "outputs/training-data/best-final-scenario-v1";
pub const DEFAULT_OUT_ROOT: &str = "outputs/ml-pipelines";
pub const DEFAULT_VALIDATION_TIER: &str = "evidence_ladder_v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineSpec {
    pub id: String,
    pub version: String,
    pub title: String,
    pub summary: String,
    pub validation_tier: String,
    pub worker_budget: usize,
    pub input_contract: Vec<String>,
    pub output_contract: Vec<String>,
    pub feature_banks: Vec<String>,
    pub detector_heads: Vec<String>,
    pub deciders: Vec<String>,
    pub references: Vec<String>,
    #[serde(default)]
    pub research_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineArtifact {
    pub id: String,
    pub kind: String,
    pub path: String,
    pub ready: bool,
    #[serde(default)]
    pub incomplete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineGate {
    pub gate: String,
    pub status: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineRunResult {
    pub pipeline_id: String,
    pub run_id: String,
    pub status: String,
    pub message: String,
    pub output_dir: String,
    #[serde(default)]
    pub artifacts: Vec<PipelineArtifact>,
    #[serde(default)]
    pub metrics: Value,
    #[serde(default)]
    pub gates: Vec<PipelineGate>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub missing_input_kind: Option<String>,
    pub worker_count: usize,
    #[serde(default)]
    pub error_code: Option<String>,
    #[serde(default)]
    pub details: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineSuiteResult {
    pub suite: String,
    pub max_concurrent: usize,
    pub workers_per_pipeline: usize,
    pub results: Vec<PipelineRunResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineRunRequest {
    pub pipeline_id: String,
    pub data_root: PathBuf,
    pub out_root: PathBuf,
    pub workers: usize,
    pub seed: u64,
    pub smoke: bool,
    pub validation_tier: String,
    #[serde(default)]
    pub repo_root: Option<PathBuf>,
}

#[derive(Debug)]
pub enum MlPipelineError {
    Io(std::io::Error),
    Json(serde_json::Error),
    PythonFailed { status: Option<i32>, stderr: String },
    InvalidRepoRoot(String),
    InvalidWorkers(String),
}

impl fmt::Display for MlPipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Json(err) => write!(f, "json error: {err}"),
            Self::PythonFailed { status, stderr } => {
                write!(f, "python runner failed with status {status:?}: {stderr}")
            }
            Self::InvalidRepoRoot(msg) => write!(f, "invalid repo root: {msg}"),
            Self::InvalidWorkers(msg) => write!(f, "invalid worker budget: {msg}"),
        }
    }
}

impl std::error::Error for MlPipelineError {}

impl From<std::io::Error> for MlPipelineError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for MlPipelineError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

pub fn discover_repo_root() -> Result<PathBuf, MlPipelineError> {
    let start = std::env::current_dir()?;
    repo_root_from(start).ok_or_else(|| {
        MlPipelineError::InvalidRepoRoot(
            "could not find Cargo.toml + detection/ from the current directory".to_string(),
        )
    })
}

pub fn repo_root_from(start: PathBuf) -> Option<PathBuf> {
    let mut current = start;
    loop {
        if is_repo_root(&current) {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

fn is_repo_root(path: &Path) -> bool {
    path.join("Cargo.toml").exists() && path.join("detection").exists()
}

pub fn resolve_repo_root(repo_root: Option<PathBuf>) -> Result<PathBuf, MlPipelineError> {
    if let Some(repo_root) = repo_root {
        if repo_root.is_absolute() && is_repo_root(&repo_root) {
            return Ok(repo_root);
        }
        let absolute = if repo_root.is_absolute() {
            repo_root
        } else {
            std::env::current_dir()?.join(repo_root)
        };
        if is_repo_root(&absolute) {
            return Ok(absolute);
        }
        return Err(MlPipelineError::InvalidRepoRoot(format!(
            "{} does not look like the EchoForge repo root",
            absolute.display()
        )));
    }
    discover_repo_root()
}

fn resolve_path(repo_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    }
}

fn thread_env(workers: usize) -> Vec<(String, String)> {
    let capped = workers.clamp(1, MAX_PIPELINE_WORKERS);
    vec![
        ("OMP_NUM_THREADS".to_string(), capped.to_string()),
        ("MKL_NUM_THREADS".to_string(), capped.to_string()),
        ("OPENBLAS_NUM_THREADS".to_string(), capped.to_string()),
        ("NUMEXPR_NUM_THREADS".to_string(), capped.to_string()),
        ("RAYON_NUM_THREADS".to_string(), capped.to_string()),
        ("PYTHONHASHSEED".to_string(), "0".to_string()),
    ]
}

fn run_python_json(repo_root: &Path, args: &[String]) -> Result<Value, MlPipelineError> {
    let mut cmd = Command::new("python3");
    cmd.current_dir(repo_root);
    cmd.arg("-m").arg("detection.pipeline_contracts.runner");
    for arg in args {
        cmd.arg(arg);
    }
    for (key, value) in thread_env(MAX_PIPELINE_WORKERS) {
        cmd.env(key, value);
    }
    let output = cmd.output()?;
    if !output.status.success() {
        return Err(MlPipelineError::PythonFailed {
            status: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn validate_workers(workers: usize) -> Result<(), MlPipelineError> {
    if workers == 0 {
        return Err(MlPipelineError::InvalidWorkers(
            "workers must be at least 1".to_string(),
        ));
    }
    if workers > MAX_PIPELINE_WORKERS {
        return Err(MlPipelineError::InvalidWorkers(format!(
            "workers must be <= {MAX_PIPELINE_WORKERS}"
        )));
    }
    Ok(())
}

pub fn list_pipelines(repo_root: Option<PathBuf>) -> Result<Vec<PipelineSpec>, MlPipelineError> {
    let repo_root = resolve_repo_root(repo_root)?;
    let payload = run_python_json(&repo_root, &[String::from("list")])?;
    Ok(serde_json::from_value(payload)?)
}

pub fn inspect_pipeline(
    pipeline_id: &str,
    repo_root: Option<PathBuf>,
) -> Result<PipelineSpec, MlPipelineError> {
    let repo_root = resolve_repo_root(repo_root)?;
    let payload = run_python_json(
        &repo_root,
        &[String::from("inspect"), pipeline_id.to_string()],
    )?;
    Ok(serde_json::from_value(payload)?)
}

pub fn run_pipeline(request: PipelineRunRequest) -> Result<PipelineRunResult, MlPipelineError> {
    validate_workers(request.workers)?;
    let repo_root = resolve_repo_root(request.repo_root)?;
    let data_root = resolve_path(&repo_root, &request.data_root);
    let out_root = resolve_path(&repo_root, &request.out_root);
    let mut args = vec![
        String::from("run"),
        String::from("--pipeline"),
        request.pipeline_id,
        String::from("--repo-root"),
        repo_root.display().to_string(),
        String::from("--data-root"),
        data_root.display().to_string(),
        String::from("--out-root"),
        out_root.display().to_string(),
        String::from("--workers"),
        request.workers.to_string(),
        String::from("--seed"),
        request.seed.to_string(),
        String::from("--validation-tier"),
        request.validation_tier,
    ];
    if request.smoke {
        args.push(String::from("--smoke"));
    }
    let payload = run_python_json(&repo_root, &args)?;
    Ok(serde_json::from_value(payload)?)
}

pub fn run_suite(
    suite: &str,
    repo_root: Option<PathBuf>,
    data_root: PathBuf,
    out_root: PathBuf,
    workers_per_pipeline: usize,
    max_concurrent: usize,
    seed: u64,
    smoke: bool,
    validation_tier: String,
) -> Result<PipelineSuiteResult, MlPipelineError> {
    validate_workers(workers_per_pipeline)?;
    if max_concurrent == 0 || max_concurrent > MAX_SUITE_CONCURRENCY {
        return Err(MlPipelineError::InvalidWorkers(format!(
            "max_concurrent must be in 1..={MAX_SUITE_CONCURRENCY}"
        )));
    }
    let repo_root = resolve_repo_root(repo_root)?;
    let data_root = resolve_path(&repo_root, &data_root);
    let out_root = resolve_path(&repo_root, &out_root);
    let mut args = vec![
        String::from("run-suite"),
        String::from("--suite"),
        suite.to_string(),
        String::from("--repo-root"),
        repo_root.display().to_string(),
        String::from("--data-root"),
        data_root.display().to_string(),
        String::from("--out-root"),
        out_root.display().to_string(),
        String::from("--workers-per-pipeline"),
        workers_per_pipeline.to_string(),
        String::from("--max-concurrent"),
        max_concurrent.to_string(),
        String::from("--seed"),
        seed.to_string(),
        String::from("--validation-tier"),
        validation_tier,
    ];
    if smoke {
        args.push(String::from("--smoke"));
    }
    let payload = run_python_json(&repo_root, &args)?;
    Ok(serde_json::from_value(payload)?)
}

pub fn default_ml_pipeline_request(
    pipeline_id: &str,
    repo_root: Option<PathBuf>,
) -> Result<PipelineRunRequest, MlPipelineError> {
    let repo_root = resolve_repo_root(repo_root)?;
    Ok(PipelineRunRequest {
        pipeline_id: pipeline_id.to_string(),
        data_root: repo_root.join(DEFAULT_DATA_ROOT),
        out_root: repo_root.join(DEFAULT_OUT_ROOT),
        workers: MAX_PIPELINE_WORKERS,
        seed: 20_260_520_390_001,
        smoke: true,
        validation_tier: DEFAULT_VALIDATION_TIER.to_string(),
        repo_root: Some(repo_root),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_root_search_finds_workspace() {
        let root =
            repo_root_from(PathBuf::from(env!("CARGO_MANIFEST_DIR"))).expect("workspace root");
        assert!(root.join("Cargo.toml").exists());
        assert!(root.join("detection").exists());
    }
}
