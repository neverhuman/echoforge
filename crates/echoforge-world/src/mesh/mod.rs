//! Analytic parametric mesh generators for canonical calibration scatterers.
//!
//! This module produces triangulated surface meshes for the five canonical
//! shapes used by the radar-primitive sanity lane: sphere, plate, cylinder,
//! dihedral, and trihedral corner reflectors. Each generator is deterministic,
//! pure-Rust, and free of external mesh dependencies; the resulting meshes
//! match the analytic primitives in `echoforge-validate` so they can be
//! cross-checked against closed-form RCS oracles before user-provided meshes
//! are accepted.
//!
//! Output triangles are right-handed with outward-pointing normals (CCW
//! winding when viewed from outside the surface). Coordinates are in metres.
//! Meshes are emitted as binary STL via [`write_stl`], and accompanied by a
//! lightweight manifest (see [`MeshManifestFile`]) recording the primitive
//! id, dimensions, triangle count, sha256 of the STL payload, and the
//! generator name + version.
//!
//! # Determinism
//! Triangle ordering and vertex coordinates are a deterministic function of
//! the input parameters; this is required for reproducible artifact hashing.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub mod generators;
mod math;
#[cfg(test)]
mod tests;

pub use generators::{cylinder_mesh, dihedral_mesh, plate_mesh, sphere_mesh, trihedral_mesh};

/// Name of this generator, embedded in mesh manifests for provenance.
pub const GENERATOR_NAME: &str = "echoforge-world::mesh";

/// Version of this generator, embedded in mesh manifests for provenance.
pub const GENERATOR_VERSION: &str = "0.1.0";

/// A single triangle in 3-space, vertices ordered counter-clockwise as seen
/// from the outward-facing side of the surface.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeshTriangle {
    pub v0: [f64; 3],
    pub v1: [f64; 3],
    pub v2: [f64; 3],
}

impl MeshTriangle {
    pub fn new(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> Self {
        Self { v0, v1, v2 }
    }

    /// Outward-facing unit normal computed from the vertex ordering.
    pub fn normal(&self) -> [f64; 3] {
        let e1 = math::sub(self.v1, self.v0);
        let e2 = math::sub(self.v2, self.v0);
        let n = math::cross(e1, e2);
        let mag = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if mag == 0.0 || !mag.is_finite() {
            [0.0, 0.0, 0.0]
        } else {
            [n[0] / mag, n[1] / mag, n[2] / mag]
        }
    }

    /// Triangle area in square metres.
    pub fn area(&self) -> f64 {
        let e1 = math::sub(self.v1, self.v0);
        let e2 = math::sub(self.v2, self.v0);
        let n = math::cross(e1, e2);
        0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
    }

    /// `true` when every coordinate is finite (no NaN, no Inf).
    pub fn all_finite(&self) -> bool {
        self.v0.iter().all(|c| c.is_finite())
            && self.v1.iter().all(|c| c.is_finite())
            && self.v2.iter().all(|c| c.is_finite())
    }
}

/// A parametric mesh: a tagged list of triangles plus the dimensions used to
/// generate it. `dimensions` is shape-specific; see the per-primitive
/// constructor docs for the slot order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParametricMesh {
    pub primitive_id: String,
    pub dimensions: Vec<f64>,
    pub triangles: Vec<MeshTriangle>,
}

impl ParametricMesh {
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn bounding_box(&self) -> ([f64; 3], [f64; 3]) {
        if self.triangles.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for t in &self.triangles {
            for v in [t.v0, t.v1, t.v2] {
                for i in 0..3 {
                    if v[i] < lo[i] {
                        lo[i] = v[i];
                    }
                    if v[i] > hi[i] {
                        hi[i] = v[i];
                    }
                }
            }
        }
        (lo, hi)
    }

    pub fn all_finite(&self) -> bool {
        self.triangles.iter().all(MeshTriangle::all_finite)
    }
}

/// Writes a [`ParametricMesh`] as a binary STL file to the given path and
/// returns the SHA-256 of the on-disk bytes as a lowercase 64-character hex
/// string.
pub fn write_stl(mesh: &ParametricMesh, path: &Path) -> io::Result<String> {
    let bytes = encode_binary_stl(mesh);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, &bytes)?;
    Ok(sha256_hex(&bytes))
}

/// Encodes a mesh as binary STL into a new `Vec<u8>`. Pure function: useful
/// for hashing without touching disk.
pub fn encode_binary_stl(mesh: &ParametricMesh) -> Vec<u8> {
    let mut out = Vec::with_capacity(84 + mesh.triangles.len() * 50);
    let mut header = [0u8; 80];
    let tag = format!(
        "echoforge-world parametric mesh: {} v{}",
        mesh.primitive_id, GENERATOR_VERSION
    );
    let tag_bytes = tag.as_bytes();
    let copy_len = tag_bytes.len().min(80);
    header[..copy_len].copy_from_slice(&tag_bytes[..copy_len]);
    out.extend_from_slice(&header);
    out.extend_from_slice(&(mesh.triangles.len() as u32).to_le_bytes());
    for t in &mesh.triangles {
        let n = t.normal();
        for v in [n, t.v0, t.v1, t.v2] {
            for c in v {
                out.extend_from_slice(&(c as f32).to_le_bytes());
            }
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut s = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(s, "{:02x}", b);
    }
    s
}

/// Lightweight per-file manifest written alongside a parametric STL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshManifestFile {
    pub primitive_id: String,
    pub dimensions: Vec<f64>,
    pub triangle_count: usize,
    pub sha256_of_stl: String,
    pub generator_name: String,
    pub generator_version: String,
}

impl MeshManifestFile {
    pub fn from_mesh(mesh: &ParametricMesh, sha256_of_stl: impl Into<String>) -> Self {
        Self {
            primitive_id: mesh.primitive_id.clone(),
            dimensions: mesh.dimensions.clone(),
            triangle_count: mesh.triangles.len(),
            sha256_of_stl: sha256_of_stl.into(),
            generator_name: GENERATOR_NAME.to_string(),
            generator_version: GENERATOR_VERSION.to_string(),
        }
    }
}

/// Emits the mesh to `<dir>/<primitive>.stl` and its companion manifest to
/// `<dir>/<primitive>_manifest.json`. Returns the two paths.
pub fn emit_mesh_bundle(mesh: &ParametricMesh, dir: &Path) -> io::Result<(PathBuf, PathBuf)> {
    std::fs::create_dir_all(dir)?;
    let stl_path = dir.join(format!("{}.stl", mesh.primitive_id));
    let manifest_path = dir.join(format!("{}_manifest.json", mesh.primitive_id));
    let sha = write_stl(mesh, &stl_path)?;
    let manifest = MeshManifestFile::from_mesh(mesh, sha);
    let json = serde_json::to_string_pretty(&manifest).map_err(io::Error::other)?;
    let mut f = std::fs::File::create(&manifest_path)?;
    f.write_all(json.as_bytes())?;
    f.write_all(b"\n")?;
    Ok((stl_path, manifest_path))
}
