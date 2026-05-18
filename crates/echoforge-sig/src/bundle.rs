//! EchoSig bundle writer / reader (Packet 5).
//!
//! On-disk layout produced by [`EchosigBundleWriter`]:
//!
//! ```text
//! <root>/
//!   manifest.json            EchosigManifest from echoforge-core
//!   provenance.json          (mirrors manifest.provenance for fast access)
//!   license.json             (mirrors manifest.license)
//!   object_card.yaml         optional (attach_card_yaml)
//!   material_card.yaml       optional
//!   solver_card.yaml         optional
//!   tensors/<name>/.zarray   minimal zarr-v2 metadata
//!   tensors/<name>/0.raw     single-chunk little-endian payload
//!   qa/<name>.json           qa documents (e.g. canonical_validation.json)
//!   dynamic/state_sequence.csv   optional state sequence (CSV fallback)
//! ```
//!
//! All tensor and QA paths must be declared in `manifest.tensor_paths` /
//! `manifest.qa_paths` ahead of time — writes to undeclared paths are
//! rejected. Each tensor must additionally be declared via
//! [`EchosigBundleWriter::declare_tensor`] so the writer can validate axis
//! counts at write time and verify completeness at `finalize`.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use ndarray::ArrayD;
use num_complex::Complex;
use serde::{Deserialize, Serialize};

use echoforge_core::{EchosigManifest, LicenseInfo, Provenance};

use crate::dynamic;
use crate::error::{SigError, SigResult};
use crate::tensor::{self, Dtype};

const MANIFEST_FILE: &str = "manifest.json";
const PROVENANCE_FILE: &str = "provenance.json";
const LICENSE_FILE: &str = "license.json";
const TENSORS_DIR: &str = "tensors";
const QA_DIR: &str = "qa";
const DYNAMIC_DIR: &str = "dynamic";
const STATE_SEQUENCE_FILE: &str = "state_sequence.csv";

/// Identifies one of the three optional YAML cards that can be attached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardKind {
    ObjectCard,
    MaterialCard,
    SolverCard,
}

impl CardKind {
    fn filename(self) -> &'static str {
        match self {
            CardKind::ObjectCard => "object_card.yaml",
            CardKind::MaterialCard => "material_card.yaml",
            CardKind::SolverCard => "solver_card.yaml",
        }
    }
}

/// Declared shape / dtype / chunking for one tensor. Tracked by the writer so
/// that `write_*` calls can validate shape rank against the manifest's axis
/// declaration before writing bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDecl {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub chunks: Vec<usize>,
}

/// Writer for an EchoSig bundle. Drop without calling [`finalize`] is allowed
/// but does not perform the completeness check.
pub struct EchosigBundleWriter {
    root: PathBuf,
    manifest: EchosigManifest,
    declared_tensors: HashMap<String, TensorDecl>,
    written_tensors: HashSet<String>,
    written_qa: HashSet<String>,
}

/// Handle returned by [`EchosigBundleWriter::finalize`].
#[derive(Debug)]
pub struct EchosigBundle {
    pub root: PathBuf,
    pub manifest: EchosigManifest,
}

impl EchosigBundleWriter {
    /// Create a new writer at `root`. The directory is created if missing.
    /// `manifest` is validated via `EchosigManifest::validate` before any
    /// directory work happens.
    pub fn create(root: &Path, manifest: EchosigManifest) -> SigResult<Self> {
        manifest.validate()?;
        fs::create_dir_all(root)?;
        fs::create_dir_all(root.join(TENSORS_DIR))?;
        fs::create_dir_all(root.join(QA_DIR))?;
        fs::create_dir_all(root.join(DYNAMIC_DIR))?;

        write_json(&root.join(MANIFEST_FILE), &manifest)?;
        write_json(&root.join(PROVENANCE_FILE), &manifest.provenance)?;
        write_json(&root.join(LICENSE_FILE), &manifest.license)?;

        Ok(Self {
            root: root.to_path_buf(),
            manifest,
            declared_tensors: HashMap::new(),
            written_tensors: HashSet::new(),
            written_qa: HashSet::new(),
        })
    }

    /// Borrow the bundle manifest.
    pub fn manifest(&self) -> &EchosigManifest {
        &self.manifest
    }

    /// Declare a tensor's dtype/shape/chunks. Must be called before any
    /// `write_tensor_*` call for that name. `name` is the manifest tensor
    /// path (e.g. `tensors/sigma_m2.zarr`).
    pub fn declare_tensor(&mut self, name: &str, decl: TensorDecl) -> SigResult<()> {
        if !self.manifest.tensor_paths.iter().any(|p| p == name) {
            return Err(SigError::UndeclaredPath(name.to_string()));
        }
        if decl.shape.len() != self.manifest.tensor_axes.len() {
            return Err(SigError::AxisMismatch {
                expected: self.manifest.tensor_axes.len(),
                actual: decl.shape.len(),
            });
        }
        self.declared_tensors.insert(name.to_string(), decl);
        Ok(())
    }

    fn tensor_dir(&self, name: &str) -> SigResult<PathBuf> {
        if !self.manifest.tensor_paths.iter().any(|p| p == name) {
            return Err(SigError::UndeclaredPath(name.to_string()));
        }
        // Strip leading "tensors/" if present so callers can pass either the
        // bare logical name ("sigma_m2.zarr") or the manifest path
        // ("tensors/sigma_m2.zarr").
        let stripped = name
            .strip_prefix(&format!("{TENSORS_DIR}/"))
            .unwrap_or(name);
        Ok(self.root.join(TENSORS_DIR).join(stripped))
    }

    fn check_declared(&self, name: &str, dtype: Dtype, shape: &[usize]) -> SigResult<()> {
        if let Some(decl) = self.declared_tensors.get(name) {
            if decl.dtype != dtype {
                return Err(SigError::SchemaMismatch(format!(
                    "tensor {name}: declared dtype {:?}, got {:?}",
                    decl.dtype, dtype
                )));
            }
            if decl.shape.as_slice() != shape {
                return Err(SigError::SchemaMismatch(format!(
                    "tensor {name}: declared shape {:?}, got {:?}",
                    decl.shape, shape
                )));
            }
        } else if shape.len() != self.manifest.tensor_axes.len() {
            return Err(SigError::AxisMismatch {
                expected: self.manifest.tensor_axes.len(),
                actual: shape.len(),
            });
        }
        Ok(())
    }

    /// Write an `f32` tensor. The tensor path must appear in
    /// `manifest.tensor_paths`.
    pub fn write_tensor_f32(&mut self, name: &str, array: &ArrayD<f32>) -> SigResult<()> {
        self.check_declared(name, Dtype::F32, array.shape())?;
        let dir = self.tensor_dir(name)?;
        tensor::write_f32(&dir, array)?;
        self.written_tensors.insert(name.to_string());
        Ok(())
    }

    /// Write a `complex64` tensor.
    pub fn write_tensor_complex64(
        &mut self,
        name: &str,
        array: &ArrayD<Complex<f32>>,
    ) -> SigResult<()> {
        self.check_declared(name, Dtype::Complex64, array.shape())?;
        let dir = self.tensor_dir(name)?;
        tensor::write_complex64(&dir, array)?;
        self.written_tensors.insert(name.to_string());
        Ok(())
    }

    /// Write a `bool` tensor.
    pub fn write_tensor_bool(&mut self, name: &str, array: &ArrayD<bool>) -> SigResult<()> {
        self.check_declared(name, Dtype::Bool, array.shape())?;
        let dir = self.tensor_dir(name)?;
        tensor::write_bool(&dir, array)?;
        self.written_tensors.insert(name.to_string());
        Ok(())
    }

    /// Write a QA JSON document at `qa/<name>.json`. The manifest must
    /// declare `qa/<name>.json` in `qa_paths`.
    pub fn write_qa_json<T: Serialize>(&mut self, name: &str, payload: &T) -> SigResult<()> {
        let qa_path = format!("{QA_DIR}/{name}.json");
        if !self.manifest.qa_paths.iter().any(|p| p == &qa_path) {
            return Err(SigError::UndeclaredPath(qa_path));
        }
        let abs = self.root.join(&qa_path);
        write_json(&abs, payload)?;
        self.written_qa.insert(qa_path);
        Ok(())
    }

    /// Attach an `object_card.yaml`, `material_card.yaml`, or
    /// `solver_card.yaml` at the bundle root.
    pub fn attach_card_yaml(&mut self, kind: CardKind, body: &str) -> SigResult<()> {
        let path = self.root.join(kind.filename());
        fs::write(path, body.as_bytes())?;
        Ok(())
    }

    /// Write `dynamic/state_sequence.csv` from a list of `(column_name, values)` pairs.
    pub fn write_state_sequence_csv(&mut self, columns: &[(&str, Vec<f64>)]) -> SigResult<()> {
        let path = self.root.join(DYNAMIC_DIR).join(STATE_SEQUENCE_FILE);
        dynamic::write_state_sequence_csv(&path, columns)
    }

    /// Cross-check that every declared tensor has been written and return a
    /// bundle handle.
    pub fn finalize(self) -> SigResult<EchosigBundle> {
        for path in &self.manifest.tensor_paths {
            if !self.written_tensors.contains(path) {
                return Err(SigError::MissingTensor(path.clone()));
            }
        }
        Ok(EchosigBundle {
            root: self.root,
            manifest: self.manifest,
        })
    }
}

/// Reader for an EchoSig bundle written by [`EchosigBundleWriter`].
pub struct EchosigBundleReader {
    root: PathBuf,
    manifest: EchosigManifest,
}

impl EchosigBundleReader {
    /// Open a bundle directory. Validates the manifest against the core schema.
    pub fn open(root: &Path) -> SigResult<Self> {
        let manifest: EchosigManifest = read_json(&root.join(MANIFEST_FILE))?;
        manifest.validate()?;
        Ok(Self {
            root: root.to_path_buf(),
            manifest,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &EchosigManifest {
        &self.manifest
    }

    /// Read `provenance.json` (mirror written by the writer).
    pub fn read_provenance(&self) -> SigResult<Provenance> {
        read_json(&self.root.join(PROVENANCE_FILE))
    }

    /// Read `license.json`.
    pub fn read_license(&self) -> SigResult<LicenseInfo> {
        read_json(&self.root.join(LICENSE_FILE))
    }

    /// Read a QA document at `qa/<name>.json` as type `T`.
    pub fn read_qa_json<T: for<'de> Deserialize<'de>>(&self, name: &str) -> SigResult<T> {
        let path = self.root.join(QA_DIR).join(format!("{name}.json"));
        read_json(&path)
    }

    /// Read `dynamic/state_sequence.csv` if present.
    pub fn read_state_sequence_csv(&self) -> SigResult<Vec<(String, Vec<f64>)>> {
        let path = self.root.join(DYNAMIC_DIR).join(STATE_SEQUENCE_FILE);
        dynamic::read_state_sequence_csv(&path)
    }

    /// List the on-disk tensor directory names under `tensors/`.
    pub fn list_tensor_files(&self) -> SigResult<Vec<String>> {
        let dir = self.root.join(TENSORS_DIR);
        let mut out = Vec::new();
        if !dir.exists() {
            return Ok(out);
        }
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                out.push(format!("{TENSORS_DIR}/{name}"));
            }
        }
        out.sort();
        Ok(out)
    }

    fn tensor_dir(&self, name: &str) -> PathBuf {
        let stripped = name
            .strip_prefix(&format!("{TENSORS_DIR}/"))
            .unwrap_or(name);
        self.root.join(TENSORS_DIR).join(stripped)
    }

    /// Load an entire `f32` tensor.
    pub fn read_tensor_f32(&self, name: &str) -> SigResult<ArrayD<f32>> {
        tensor::read_f32(&self.tensor_dir(name))
    }

    /// Load an entire `complex64` tensor.
    pub fn read_tensor_complex64(&self, name: &str) -> SigResult<ArrayD<Complex<f32>>> {
        tensor::read_complex64(&self.tensor_dir(name))
    }

    /// Load an entire `bool` tensor.
    pub fn read_tensor_bool(&self, name: &str) -> SigResult<ArrayD<bool>> {
        tensor::read_bool(&self.tensor_dir(name))
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> SigResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(path, bytes)?;
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> SigResult<T> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}
