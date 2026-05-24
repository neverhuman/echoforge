use crate::core::{Health, StatusCheck, StatusSummary};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaFileReport {
    pub path: PathBuf,
    pub health: Health,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaValidationReport {
    pub summary: StatusSummary,
    pub files: Vec<SchemaFileReport>,
}

impl SchemaValidationReport {
    pub fn render(&self) -> String {
        self.summary.render()
    }
}

pub fn validate_inputs(paths: &[PathBuf]) -> SchemaValidationReport {
    let mut files = Vec::new();
    for path in paths {
        collect_paths(path, &mut files, false);
    }

    if files.is_empty() {
        let mut summary = StatusSummary::new("Schema validation");
        summary.push(StatusCheck::new(
            "inputs",
            Health::Warn,
            "no schema inputs provided",
        ));
        return SchemaValidationReport {
            summary,
            files: Vec::new(),
        };
    }

    let mut summary = StatusSummary::new("Schema validation");
    let mut reports = Vec::new();

    for path in files {
        let report = validate_file(&path);
        let health = report.health;
        summary.push(StatusCheck::new(
            path.display().to_string(),
            health,
            report.detail.clone(),
        ));
        reports.push(report);
    }

    SchemaValidationReport {
        summary,
        files: reports,
    }
}

fn collect_paths(path: &Path, files: &mut Vec<PathBuf>, from_dir: bool) {
    if path.is_dir() {
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                collect_paths(&entry.path(), files, true);
            }
        }
    } else if !from_dir || is_jsonish(path) {
        files.push(path.to_path_buf());
    }
}

fn is_jsonish(path: &Path) -> bool {
    matches!(path.extension().and_then(|ext| ext.to_str()), Some("json"))
}

fn validate_file(path: &Path) -> SchemaFileReport {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            return SchemaFileReport {
                path: path.to_path_buf(),
                health: Health::Fail,
                detail: format!("unable to read file: {err}"),
            }
        }
    };

    let json: Result<Value, _> = serde_json::from_str(&text);
    let json = match json {
        Ok(json) => json,
        Err(err) => {
            return SchemaFileReport {
                path: path.to_path_buf(),
                health: Health::Fail,
                detail: format!("invalid JSON: {err}"),
            }
        }
    };

    if !json.is_object() {
        return SchemaFileReport {
            path: path.to_path_buf(),
            health: Health::Fail,
            detail: "schema root is not a JSON object".to_string(),
        };
    }

    let has_schema_marker = json.get("$schema").is_some()
        || json.get("title").is_some()
        || json.get("type").is_some()
        || json.get("$id").is_some();

    if !has_schema_marker {
        return SchemaFileReport {
            path: path.to_path_buf(),
            health: Health::Warn,
            detail: "parsed successfully, but no obvious schema markers were found".to_string(),
        };
    }

    SchemaFileReport {
        path: path.to_path_buf(),
        health: Health::Ok,
        detail: "parsed successfully".to_string(),
    }
}
