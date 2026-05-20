use crate::calibrate::{run_calibrate, CalibrateArgs};
use crate::core::{Health, StatusCheck, StatusSummary};
use crate::demo::{run_demo, DemoArgs};
use crate::ml::{run_ml, MlArgs};
use crate::manifest::EchoSigManifest;
use crate::schema::{validate_inputs, SchemaValidationReport};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "ef", version, about = "EchoForge CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Validate JSON schema files or schema directories.
    Schema(SchemaArgs),
    /// Inspect a single EchoSig manifest.
    EchoSig(ManifestArgs),
    /// Print a doctor-style status summary.
    Doctor(DoctorArgs),
    /// Run the V-tier validation gate against an EchoSig bundle directory.
    Validate(ValidateArgs),
    /// Run generated data demos.
    Demo(DemoArgs),
    /// Run ML pipeline discovery and evidence ladder jobs.
    Ml(MlArgs),
    /// Run calibration / credibility sweeps (e.g. empirical Pfa for C12).
    Calibrate(CalibrateArgs),
}

#[derive(Debug, Args)]
pub struct ValidateArgs {
    /// Path to an EchoSig bundle directory containing manifest.json (+ qa/).
    #[arg(value_name = "BUNDLE")]
    pub bundle: PathBuf,
    /// Canonical primitive to validate against (auto-detect by default).
    #[arg(long, value_name = "PRIMITIVE", default_value = "auto")]
    pub primitive: String,
    /// Target V-tier (v0, v1, v2).
    #[arg(long = "target-tier", value_name = "TIER", default_value = "v1")]
    pub target_tier: String,
    /// Optional path to write the validation report JSON.
    #[arg(long = "write-report", value_name = "PATH")]
    pub write_report: Option<PathBuf>,
    /// Upgrade warnings to failures.
    #[arg(long)]
    pub strict: bool,
}

#[derive(Debug, Args)]
pub struct SchemaArgs {
    #[arg(value_name = "PATH")]
    pub inputs: Vec<PathBuf>,
}

#[derive(Debug, Args)]
pub struct ManifestArgs {
    #[arg(value_name = "MANIFEST")]
    pub manifest: PathBuf,
}

#[derive(Debug, Args)]
pub struct DoctorArgs {
    #[arg(long, value_name = "MANIFEST")]
    pub manifest: Option<PathBuf>,

    #[arg(long = "schema", value_name = "PATH")]
    pub schemas: Vec<PathBuf>,
}

pub fn run(args: impl IntoIterator<Item = std::ffi::OsString>) -> Result<u8, String> {
    let cli = Cli::try_parse_from(args).map_err(|err| err.to_string())?;

    let summary = match cli.command {
        Command::Validate(args) => {
            let v_args = echoforge_validate::cli::ValidateArgs {
                bundle: args.bundle,
                primitive: Some(args.primitive),
                target_tier: args.target_tier,
                write_report: args.write_report,
                strict: args.strict,
            };
            return match echoforge_validate::cli::run(v_args) {
                Ok(code) => Ok(code.clamp(0, 255) as u8),
                Err(err) => {
                    eprintln!("{err}");
                    let rc: u8 = match err {
                        echoforge_validate::ValidateError::BadArgs(_) => 3,
                        echoforge_validate::ValidateError::Schema(_)
                        | echoforge_validate::ValidateError::Io(_)
                        | echoforge_validate::ValidateError::Json(_) => 2,
                        echoforge_validate::ValidateError::Failed(_) => 1,
                    };
                    Ok(rc)
                }
            };
        }
        Command::Demo(args) => return run_demo(args),
        Command::Ml(args) => return run_ml(args),
        Command::Calibrate(args) => {
            return run_calibrate(args).map(|code| code.clamp(0, 255) as u8);
        }
        Command::Schema(args) => {
            let report = validate_inputs(&args.inputs);
            println!("{}", report.render());
            report.summary
        }
        Command::EchoSig(args) => {
            let manifest = EchoSigManifest::load(args.manifest);
            let report = manifest.inspect();
            println!("{}", report.render());
            report.status
        }
        Command::Doctor(args) => {
            let mut summary = StatusSummary::new("EchoForge doctor");

            if let Some(manifest_path) = args.manifest.as_ref() {
                let manifest = EchoSigManifest::load(manifest_path);
                let report = manifest.inspect();
                summary.push(StatusCheck::new(
                    "EchoSig manifest",
                    report.status.overall(),
                    format!("manifest: {}", manifest_path.display()),
                ));
                println!("{}", report.render());
            } else {
                summary.push(StatusCheck::new(
                    "EchoSig manifest",
                    Health::Warn,
                    "no manifest supplied",
                ));
            }

            let schema_report: SchemaValidationReport = validate_inputs(&args.schemas);
            let schema_detail = if args.schemas.is_empty() {
                "no schema inputs supplied".to_string()
            } else {
                format!("{} schema input(s)", args.schemas.len())
            };
            summary.push(StatusCheck::new(
                "Schema validation",
                schema_report.summary.overall(),
                schema_detail,
            ));

            summary.push(StatusCheck::new(
                "CLI wiring",
                Health::Ok,
                "argument parsing and reporting are working",
            ));

            println!("{}", summary.render());
            summary
        }
    };

    Ok(if summary.overall() == Health::Fail {
        1
    } else {
        0
    })
}
