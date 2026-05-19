//! `ef pack` subcommand — discover, inspect, and validate object packs.
//!
//! Subcommands:
//!
//! - `ef pack list [--packs-root <DIR>]` — list every discovered pack and
//!   its card count by kind.
//! - `ef pack show <KIND> <SLUG> [--packs-root <DIR>]` — print the full
//!   card payload for a given kind+slug match.
//! - `ef pack validate [--packs-root <DIR>] [--schemas-root <DIR>]` —
//!   JSON-schema-validate every card in every discovered pack. Exit
//!   code 0 if all pass, 1 otherwise.

use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};

use echoforge_packs::validation::SchemaCatalog;
use echoforge_packs::Registry;

#[derive(Debug)]
enum PackError {
    CardNotFound { kind: String, slug: String },
    Registry(String),
    Json(String),
}

impl std::fmt::Display for PackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CardNotFound { kind, slug } => write!(f, "no card found for kind={kind} slug={slug}"),
            Self::Registry(e) => write!(f, "{e}"),
            Self::Json(e) => write!(f, "json serialize: {e}"),
        }
    }
}

#[derive(Debug, Args)]
pub struct PackArgs {
    #[command(subcommand)]
    pub command: PackCommand,
}

#[derive(Debug, Subcommand)]
pub enum PackCommand {
    /// List discovered packs.
    List(ListArgs),
    /// Show a single card payload by (kind, slug).
    Show(ShowArgs),
    /// JSON-schema-validate every card in every discovered pack.
    Validate(ValidateArgs),
}

#[derive(Debug, Args)]
pub struct ListArgs {
    /// Override the default `<repo>/object-packs` discovery root.
    #[arg(long, value_name = "DIR")]
    pub packs_root: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct ShowArgs {
    /// Card kind discriminator (e.g. `object_card`, `radar_platform_card`).
    #[arg(value_name = "KIND")]
    pub kind: String,
    /// Card slug (the `public_proxy_id` field).
    #[arg(value_name = "SLUG")]
    pub slug: String,
    #[arg(long, value_name = "DIR")]
    pub packs_root: Option<PathBuf>,
    /// Emit raw JSON (default: human-readable).
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct ValidateArgs {
    #[arg(long, value_name = "DIR")]
    pub packs_root: Option<PathBuf>,
    #[arg(long, value_name = "DIR")]
    pub schemas_root: Option<PathBuf>,
}

pub fn run_pack(args: PackArgs) -> Result<u8, String> {
    match args.command {
        PackCommand::List(list) => run_list(list),
        PackCommand::Show(show) => run_show(show).map_err(|e| e.to_string()),
        PackCommand::Validate(val) => run_validate(val),
    }
}

fn run_list(args: ListArgs) -> Result<u8, String> {
    let packs_root = resolve_packs_root(args.packs_root)?;
    let registry = Registry::discover(&packs_root).map_err(|e| e.to_string())?;
    println!(
        "Discovered {} pack(s) under {}",
        registry.pack_count(),
        packs_root.display()
    );
    for (slug, entry) in registry.packs() {
        println!("- pack `{slug}` ({} cards)", entry.card_count());
        println!("    manifest_version: {:?}", entry.manifest.manifest_version);
        println!("    validation_tier:  {}", entry.manifest.validation_tier);
        println!("    purpose:          {}", entry.manifest.purpose);
        for (kind, by_slug) in &entry.cards {
            println!("    [{kind}] {} card(s)", by_slug.len());
            for card_slug in by_slug.keys() {
                println!("      - {card_slug}");
            }
        }
    }
    Ok(0)
}

fn run_show(args: ShowArgs) -> Result<u8, PackError> {
    let packs_root = resolve_packs_root(args.packs_root)
        .map_err(PackError::Registry)?;
    let registry = Registry::discover(&packs_root)
        .map_err(|e| PackError::Registry(e.to_string()))?;
    let card = registry
        .find_card(&args.kind, &args.slug)
        .ok_or(PackError::CardNotFound { kind: args.kind.clone(), slug: args.slug.clone() })?;
    if args.json {
        let text = serde_json::to_string_pretty(&*card.value)
            .map_err(|e| PackError::Json(e.to_string()))?;
        println!("{text}");
    } else {
        println!("kind:  {}", card.kind);
        println!("slug:  {}", card.slug);
        println!("path:  {}", card.path.display());
        if let Some(name) = card.value.get("display_name").and_then(|v| v.as_str()) {
            println!("name:  {name}");
        }
        if let Some(tier) = card
            .value
            .get("validation")
            .and_then(|v| v.get("tier"))
            .and_then(|v| v.as_str())
        {
            println!("tier:  {tier}");
        }
        if let Some(fid) = card
            .value
            .get("validation")
            .and_then(|v| v.get("fidelity_class"))
            .and_then(|v| v.as_str())
        {
            println!("fidelity_class: {fid}");
        }
    }
    Ok(0)
}

fn run_validate(args: ValidateArgs) -> Result<u8, String> {
    let packs_root = resolve_packs_root(args.packs_root)?;
    let schemas_root = match args.schemas_root {
        Some(p) => p,
        None => discover_repo_root()?,
    };
    let registry = Registry::discover(&packs_root).map_err(|e| e.to_string())?;
    let mut catalog = SchemaCatalog::from_repo(&schemas_root).map_err(|e| e.to_string())?;
    match registry.validate_with_catalog(&mut catalog) {
        Ok(()) => {
            println!(
                "OK: {} card(s) across {} pack(s) validated against {} schema(s)",
                registry.total_card_count(),
                registry.pack_count(),
                catalog.cached_validator_count()
            );
            Ok(0)
        }
        Err(errs) => {
            for err in &errs {
                eprintln!("FAIL {err}");
            }
            eprintln!("validation failed: {} error(s)", errs.len());
            Ok(1)
        }
    }
}

/// Resolve the packs root: honor the explicit flag, else `<repo>/object-packs/`.
fn resolve_packs_root(explicit: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(p) = explicit {
        return Ok(p);
    }
    let repo = discover_repo_root()?;
    Ok(repo.join("object-packs"))
}

/// Find the EchoForge repo root by walking up from CWD looking for
/// `Cargo.toml` + `object-packs/`.
fn discover_repo_root() -> Result<PathBuf, String> {
    let mut cur = std::env::current_dir().map_err(|e| format!("cwd: {e}"))?;
    loop {
        if has_marker(&cur) {
            return Ok(cur);
        }
        match cur.parent() {
            Some(p) => cur = p.to_path_buf(),
            None => break,
        }
    }
    Err("could not find EchoForge repo root (Cargo.toml + object-packs/)".to_string())
}

fn has_marker(dir: &Path) -> bool {
    dir.join("Cargo.toml").exists() && dir.join("object-packs").exists() && dir.join("schemas").exists()
}
