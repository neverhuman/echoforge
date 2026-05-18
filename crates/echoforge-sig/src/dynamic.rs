//! Dynamic state-sequence I/O for EchoSig bundles.
//!
//! Per Packet 5 of the plan, the long-term target is Parquet (via `arrow` /
//! `parquet`). The first-cut fallback shipped here is a plain CSV writer that
//! the Python side (`echoforge_signatures.bundle`) can read with stdlib
//! `csv`, keeping cross-language reads dependency-free. Promoting to Parquet
//! is a drop-in replacement of this module.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::{SigError, SigResult};

/// Write a state-sequence CSV.
///
/// `columns` is an ordered list of `(name, values)` pairs; every column must
/// have the same number of rows.
pub fn write_state_sequence_csv(path: &Path, columns: &[(&str, Vec<f64>)]) -> SigResult<()> {
    if columns.is_empty() {
        return Err(SigError::InvalidTensor(
            "state_sequence must have at least one column".into(),
        ));
    }
    let row_count = columns[0].1.len();
    for (name, values) in columns {
        if values.len() != row_count {
            return Err(SigError::InvalidTensor(format!(
                "column {name} has {} rows, expected {row_count}",
                values.len()
            )));
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut buf = String::new();
    let names: Vec<&str> = columns.iter().map(|(n, _)| *n).collect();
    buf.push_str(&names.join(","));
    buf.push('\n');
    for row in 0..row_count {
        let mut first = true;
        for (_, values) in columns {
            if !first {
                buf.push(',');
            }
            first = false;
            // Use a stable, lossless float representation.
            buf.push_str(&format!("{:.17e}", values[row]));
        }
        buf.push('\n');
    }
    fs::write(path, buf)?;
    Ok(())
}

/// Read a state-sequence CSV. Returns `Vec<(column_name, values)>`.
pub fn read_state_sequence_csv(path: &Path) -> SigResult<Vec<(String, Vec<f64>)>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let header = lines
        .next()
        .ok_or_else(|| SigError::InvalidTensor("empty state_sequence csv".into()))??;
    let names: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
    let mut columns: Vec<(String, Vec<f64>)> =
        names.iter().map(|n| (n.clone(), Vec::new())).collect();

    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != columns.len() {
            return Err(SigError::InvalidTensor(format!(
                "state_sequence row has {} cols, header has {}",
                parts.len(),
                columns.len()
            )));
        }
        for (i, p) in parts.iter().enumerate() {
            let v: f64 = p
                .trim()
                .parse()
                .map_err(|e| SigError::InvalidTensor(format!("parse error: {e}")))?;
            columns[i].1.push(v);
        }
    }
    Ok(columns)
}
