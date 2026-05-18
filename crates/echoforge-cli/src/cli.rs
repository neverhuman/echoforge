use crate::core::{Health, StatusCheck, StatusSummary};
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

    Ok(if summary.overall() == Health::Fail { 1 } else { 0 })
}
