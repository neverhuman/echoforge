//! Tensor I/O for EchoSig bundles.
//!
//! This module implements a minimal "Zarr-like" layout chosen as the first-cut
//! recovery per Packet 5 of the EchoForge plan. Each tensor directory contains:
//!
//! ```text
//! tensors/<name>/
//!   .zarray            JSON metadata (zarr v2 / v3 compatible subset)
//!   0.raw              little-endian raw bytes for the single chunk (whole array)
//! ```
//!
//! Multi-chunk support is intentionally deferred to a follow-up packet; the
//! current writer always emits a single chunk equal to the array shape. The
//! `.zarray` metadata records the declared chunk shape so a future
//! multi-chunk writer/reader can replace this module without breaking the
//! on-disk envelope. The Python side (`echoforge_signatures.bundle`) mirrors
//! this exact layout so cross-language reads work without any extra deps.

use std::fs;
use std::path::Path;

use ndarray::ArrayD;
use num_complex::Complex;
use serde::{Deserialize, Serialize};

use crate::error::{SigError, SigResult};

/// Logical element type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    F32,
    F64,
    Complex64,
    Bool,
}

impl Dtype {
    /// Zarr-style dtype code. Uses the v2 string vocabulary which is also
    /// understood by zarr-python v3 for the subset we need.
    pub fn zarr_code(self) -> &'static str {
        match self {
            Dtype::F32 => "<f4",
            Dtype::F64 => "<f8",
            Dtype::Complex64 => "<c8",
            Dtype::Bool => "|b1",
        }
    }

    pub fn element_size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F64 => 8,
            Dtype::Complex64 => 8,
            Dtype::Bool => 1,
        }
    }
}

/// On-disk metadata for a tensor directory. Mirrors a subset of zarr v2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZarrayMetadata {
    pub shape: Vec<usize>,
    pub chunks: Vec<usize>,
    pub dtype: String,
    #[serde(default)]
    pub fill_value: serde_json::Value,
    #[serde(default = "default_order")]
    pub order: String,
    #[serde(default)]
    pub compressor: Option<serde_json::Value>,
    #[serde(default)]
    pub filters: Option<Vec<serde_json::Value>>,
    #[serde(default = "default_zarr_format")]
    pub zarr_format: u8,
}

fn default_order() -> String {
    "C".to_string()
}

fn default_zarr_format() -> u8 {
    2
}

fn write_metadata(dir: &Path, meta: &ZarrayMetadata) -> SigResult<()> {
    fs::create_dir_all(dir)?;
    let bytes = serde_json::to_vec_pretty(meta)?;
    fs::write(dir.join(".zarray"), bytes)?;
    Ok(())
}

fn read_metadata(dir: &Path) -> SigResult<ZarrayMetadata> {
    let bytes = fs::read(dir.join(".zarray"))?;
    let meta: ZarrayMetadata = serde_json::from_slice(&bytes)?;
    Ok(meta)
}

fn chunk_path(dir: &Path) -> std::path::PathBuf {
    // Single-chunk recovery — chunk 0 along every axis. We use a flat name
    // (`0.raw`) for zero-rank tensors. The naming intentionally matches the
    // Python reference implementation.
    dir.join("0.raw")
}

/// Write an `f32` tensor.
pub fn write_f32(dir: &Path, array: &ArrayD<f32>) -> SigResult<()> {
    let shape = array.shape().to_vec();
    let meta = ZarrayMetadata {
        shape: shape.clone(),
        chunks: shape,
        dtype: Dtype::F32.zarr_code().to_string(),
        fill_value: serde_json::Value::from(0.0),
        order: "C".to_string(),
        compressor: None,
        filters: None,
        zarr_format: 2,
    };
    write_metadata(dir, &meta)?;
    let contiguous = array.as_standard_layout().to_owned();
    let slice = match contiguous.as_slice() {
        Some(v) => v,
        None => return Err(SigError::InvalidTensor("f32 tensor not contiguous".into())),
    };
    let mut bytes = Vec::with_capacity(slice.len() * 4);
    for v in slice {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(chunk_path(dir), bytes)?;
    Ok(())
}

/// Write an `f64` tensor.
pub fn write_f64(dir: &Path, array: &ArrayD<f64>) -> SigResult<()> {
    let shape = array.shape().to_vec();
    let meta = ZarrayMetadata {
        shape: shape.clone(),
        chunks: shape,
        dtype: Dtype::F64.zarr_code().to_string(),
        fill_value: serde_json::Value::from(0.0),
        order: "C".to_string(),
        compressor: None,
        filters: None,
        zarr_format: 2,
    };
    write_metadata(dir, &meta)?;
    let contiguous = array.as_standard_layout().to_owned();
    let slice = match contiguous.as_slice() {
        Some(v) => v,
        None => return Err(SigError::InvalidTensor("f64 tensor not contiguous".into())),
    };
    let mut bytes = Vec::with_capacity(slice.len() * 8);
    for v in slice {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(chunk_path(dir), bytes)?;
    Ok(())
}

/// Write a `complex64` (== two `f32`) tensor. Layout: re, im, re, im, ... in C order.
pub fn write_complex64(dir: &Path, array: &ArrayD<Complex<f32>>) -> SigResult<()> {
    let shape = array.shape().to_vec();
    let meta = ZarrayMetadata {
        shape: shape.clone(),
        chunks: shape,
        dtype: Dtype::Complex64.zarr_code().to_string(),
        fill_value: serde_json::Value::Array(vec![
            serde_json::Value::from(0.0),
            serde_json::Value::from(0.0),
        ]),
        order: "C".to_string(),
        compressor: None,
        filters: None,
        zarr_format: 2,
    };
    write_metadata(dir, &meta)?;
    let contiguous = array.as_standard_layout().to_owned();
    let slice = match contiguous.as_slice() {
        Some(v) => v,
        None => return Err(SigError::InvalidTensor("complex64 tensor not contiguous".into())),
    };
    let mut bytes = Vec::with_capacity(slice.len() * 8);
    for c in slice {
        bytes.extend_from_slice(&c.re.to_le_bytes());
        bytes.extend_from_slice(&c.im.to_le_bytes());
    }
    fs::write(chunk_path(dir), bytes)?;
    Ok(())
}

/// Write a `bool` tensor (1 byte per element, 0/1).
pub fn write_bool(dir: &Path, array: &ArrayD<bool>) -> SigResult<()> {
    let shape = array.shape().to_vec();
    let meta = ZarrayMetadata {
        shape: shape.clone(),
        chunks: shape,
        dtype: Dtype::Bool.zarr_code().to_string(),
        fill_value: serde_json::Value::from(false),
        order: "C".to_string(),
        compressor: None,
        filters: None,
        zarr_format: 2,
    };
    write_metadata(dir, &meta)?;
    let contiguous = array.as_standard_layout().to_owned();
    let slice = match contiguous.as_slice() {
        Some(v) => v,
        None => return Err(SigError::InvalidTensor("bool tensor not contiguous".into())),
    };
    let bytes: Vec<u8> = slice.iter().map(|b| if *b { 1u8 } else { 0u8 }).collect();
    fs::write(chunk_path(dir), bytes)?;
    Ok(())
}

/// Read an `f32` tensor.
pub fn read_f32(dir: &Path) -> SigResult<ArrayD<f32>> {
    let meta = read_metadata(dir)?;
    if meta.dtype != Dtype::F32.zarr_code() {
        return Err(SigError::SchemaMismatch(format!(
            "expected dtype {}, got {}",
            Dtype::F32.zarr_code(),
            meta.dtype
        )));
    }
    let bytes = fs::read(chunk_path(dir))?;
    let count: usize = meta.shape.iter().product();
    if bytes.len() != count * 4 {
        return Err(SigError::InvalidTensor(format!(
            "f32 tensor byte length {} != expected {}",
            bytes.len(),
            count * 4
        )));
    }
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
        values.push(f32::from_le_bytes(buf));
    }
    ArrayD::from_shape_vec(meta.shape, values)
        .map_err(|e| SigError::InvalidTensor(format!("shape error: {e}")))
}

/// Read an `f64` tensor.
pub fn read_f64(dir: &Path) -> SigResult<ArrayD<f64>> {
    let meta = read_metadata(dir)?;
    if meta.dtype != Dtype::F64.zarr_code() {
        return Err(SigError::SchemaMismatch(format!(
            "expected dtype {}, got {}",
            Dtype::F64.zarr_code(),
            meta.dtype
        )));
    }
    let bytes = fs::read(chunk_path(dir))?;
    let count: usize = meta.shape.iter().product();
    if bytes.len() != count * 8 {
        return Err(SigError::InvalidTensor(format!(
            "f64 tensor byte length {} != expected {}",
            bytes.len(),
            count * 8
        )));
    }
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
        values.push(f64::from_le_bytes(buf));
    }
    ArrayD::from_shape_vec(meta.shape, values)
        .map_err(|e| SigError::InvalidTensor(format!("shape error: {e}")))
}

/// Read a `complex64` tensor.
pub fn read_complex64(dir: &Path) -> SigResult<ArrayD<Complex<f32>>> {
    let meta = read_metadata(dir)?;
    if meta.dtype != Dtype::Complex64.zarr_code() {
        return Err(SigError::SchemaMismatch(format!(
            "expected dtype {}, got {}",
            Dtype::Complex64.zarr_code(),
            meta.dtype
        )));
    }
    let bytes = fs::read(chunk_path(dir))?;
    let count: usize = meta.shape.iter().product();
    if bytes.len() != count * 8 {
        return Err(SigError::InvalidTensor(format!(
            "complex64 tensor byte length {} != expected {}",
            bytes.len(),
            count * 8
        )));
    }
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let mut re = [0u8; 4];
        let mut im = [0u8; 4];
        re.copy_from_slice(&bytes[i * 8..i * 8 + 4]);
        im.copy_from_slice(&bytes[i * 8 + 4..i * 8 + 8]);
        values.push(Complex::new(f32::from_le_bytes(re), f32::from_le_bytes(im)));
    }
    ArrayD::from_shape_vec(meta.shape, values)
        .map_err(|e| SigError::InvalidTensor(format!("shape error: {e}")))
}

/// Read a `bool` tensor.
pub fn read_bool(dir: &Path) -> SigResult<ArrayD<bool>> {
    let meta = read_metadata(dir)?;
    if meta.dtype != Dtype::Bool.zarr_code() {
        return Err(SigError::SchemaMismatch(format!(
            "expected dtype {}, got {}",
            Dtype::Bool.zarr_code(),
            meta.dtype
        )));
    }
    let bytes = fs::read(chunk_path(dir))?;
    let count: usize = meta.shape.iter().product();
    if bytes.len() != count {
        return Err(SigError::InvalidTensor(format!(
            "bool tensor byte length {} != expected {}",
            bytes.len(),
            count
        )));
    }
    let values: Vec<bool> = bytes.iter().map(|b| *b != 0).collect();
    ArrayD::from_shape_vec(meta.shape, values)
        .map_err(|e| SigError::InvalidTensor(format!("shape error: {e}")))
}
