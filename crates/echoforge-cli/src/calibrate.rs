//! `ef calibrate empirical-pfa` subcommand. Runs the Lane G_b empirical Pfa
//! calibrator from `echoforge_radar::empirical_pfa` and writes a JSON Lines
//! receipt to disk. Summary line goes to stderr; the JSONL file is the
//! consumable artifact a radar engineer audits.

use clap::{Args, Subcommand};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Args)]
pub struct CalibrateArgs {
    #[command(subcommand)]
    pub command: CalibrateCommand,
}

#[derive(Debug, Subcommand)]
pub enum CalibrateCommand {
    /// Empirical Pfa calibration sweep (Lane G_b / C12). For each
    /// (ClutterRegime, CfarVariant) pair, generates `--trials` target-free
    /// cells, counts false alarms, compares observed vs nominal via 95% Wilson CI.
    EmpiricalPfa(EmpiricalPfaArgs),
}

#[derive(Debug, Args)]
pub struct EmpiricalPfaArgs {
    /// Number of CFAR decisions to evaluate per (regime, variant) combo.
    /// Production gate value is 1e7; smaller values run faster but yield
    /// wider Wilson CIs.
    #[arg(long, default_value_t = 10_000_000)]
    pub trials: u64,

    /// Master seed; XOR'd with combo index so each combo is independent.
    #[arg(long, default_value_t = 0xC12_A001)]
    pub seed: u64,

    /// Output JSON Lines file path (one observation per line).
    #[arg(long = "out", value_name = "PATH")]
    pub out: PathBuf,
}

pub fn run_calibrate(args: CalibrateArgs) -> Result<u8, String> {
    match args.command {
        CalibrateCommand::EmpiricalPfa(args) => run_empirical_pfa(args),
    }
}

fn run_empirical_pfa(args: EmpiricalPfaArgs) -> Result<u8, String> {
    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create output parent {}: {err}",
                    parent.display()
                )
            })?;
        }
    }
    let observations = echoforge_radar::empirical_pfa::calibrate_standard_table(
        args.trials,
        args.seed,
    );
    let jsonl = echoforge_radar::empirical_pfa::render_jsonl(&observations);
    let mut file = fs::File::create(&args.out)
        .map_err(|err| format!("failed to create {}: {err}", args.out.display()))?;
    file.write_all(jsonl.as_bytes())
        .map_err(|err| format!("failed to write {}: {err}", args.out.display()))?;
    let pass_count = observations.iter().filter(|o| o.passes).count();
    let fail_count = observations.len() - pass_count;
    eprintln!(
        "empirical-pfa: wrote {} observations to {} ({} pass / {} fail at 95% Wilson CI)",
        observations.len(),
        args.out.display(),
        pass_count,
        fail_count,
    );
    Ok(if fail_count == 0 { 0 } else { 1 })
}
