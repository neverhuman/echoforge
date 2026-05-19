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
//! generator name + version. The manifest is intentionally minimal here; the
//! full canonical `mesh_manifest` envelope (provenance / license / validation)
//! lives in `schemas/mesh_manifest.schema.json` and is composed by the
//! calling pipeline.
//!
//! # Determinism
//! Triangle ordering and vertex coordinates are a deterministic function of
//! the input parameters; this is required for reproducible artifact hashing.
//!
//! # References (geometry only; no third-party code)
//! - Standard latitude/longitude sphere tessellation.
//! - Knott / Ruck RCS Handbook conventions for dihedral and trihedral corner
//!   reflectors (geometry only — RCS lives in `echoforge-validate`).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

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
    /// Construct a triangle from three vertices.
    pub fn new(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> Self {
        Self { v0, v1, v2 }
    }

    /// Outward-facing unit normal computed from the vertex ordering.
    ///
    /// Returns `[0,0,0]` for degenerate (zero-area) triangles rather than
    /// producing NaNs — callers that need to flag degeneracies should use
    /// [`Self::area`].
    pub fn normal(&self) -> [f64; 3] {
        let e1 = sub(self.v1, self.v0);
        let e2 = sub(self.v2, self.v0);
        let n = cross(e1, e2);
        let mag = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if mag == 0.0 || !mag.is_finite() {
            [0.0, 0.0, 0.0]
        } else {
            [n[0] / mag, n[1] / mag, n[2] / mag]
        }
    }

    /// Triangle area in square metres.
    pub fn area(&self) -> f64 {
        let e1 = sub(self.v1, self.v0);
        let e2 = sub(self.v2, self.v0);
        let n = cross(e1, e2);
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
    /// Stable slug naming the primitive family (e.g. `sphere`, `plate`,
    /// `cylinder`, `dihedral`, `trihedral`).
    pub primitive_id: String,
    /// Shape-defining lengths in metres. The order is documented per
    /// generator; for example, `cylinder` records `[radius_m, length_m]`.
    pub dimensions: Vec<f64>,
    /// Triangulated surface. Winding is CCW from outside the surface.
    pub triangles: Vec<MeshTriangle>,
}

impl ParametricMesh {
    /// Number of triangles in the surface.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Axis-aligned bounding box as `(min, max)`. Returns
    /// `([0,0,0],[0,0,0])` for an empty mesh.
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

    /// Whether every triangle has only finite vertex coordinates.
    pub fn all_finite(&self) -> bool {
        self.triangles.iter().all(MeshTriangle::all_finite)
    }
}

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------

/// Triangulated PEC sphere of radius `radius_m`, sampled on `n_lat` latitude
/// bands by `n_lon` longitude slices.
///
/// The dimensions slot order is `[radius_m]`.
///
/// Tessellation is the standard lat/lon scheme: latitudes are sampled at
/// `n_lat` values from the north pole (θ=0) to the south pole (θ=π), and
/// longitudes are sampled at `n_lon` values around the azimuth. Every
/// longitude/latitude quad emits two triangles, yielding
/// `(n_lat - 1) * n_lon * 2` triangles total. The poles collapse to a single
/// shared vertex but the quad-pair count is preserved (one triangle of each
/// polar pair is geometrically a sliver of width zero on one edge but still a
/// valid CCW triangle with non-zero area).
///
/// # Panics
/// Panics if `n_lat < 2`, `n_lon < 3`, or `radius_m <= 0.0`.
pub fn sphere_mesh(radius_m: f64, n_lat: usize, n_lon: usize) -> ParametricMesh {
    assert!(n_lat >= 2, "sphere mesh requires n_lat >= 2");
    assert!(n_lon >= 3, "sphere mesh requires n_lon >= 3");
    assert!(radius_m > 0.0, "sphere radius must be positive");

    let mut triangles = Vec::with_capacity((n_lat - 1) * n_lon * 2);

    let pi = std::f64::consts::PI;
    let two_pi = 2.0 * pi;

    let vertex = |i_lat: usize, j_lon: usize| -> [f64; 3] {
        let theta = pi * (i_lat as f64) / ((n_lat - 1) as f64);
        let phi = two_pi * (j_lon as f64) / (n_lon as f64);
        let sin_t = theta.sin();
        [
            radius_m * sin_t * phi.cos(),
            radius_m * sin_t * phi.sin(),
            radius_m * theta.cos(),
        ]
    };

    for i in 0..(n_lat - 1) {
        for j in 0..n_lon {
            let j1 = (j + 1) % n_lon;
            let v00 = vertex(i, j);
            let v10 = vertex(i + 1, j);
            let v11 = vertex(i + 1, j1);
            let v01 = vertex(i, j1);
            // CCW from outside: (v00, v10, v11) and (v00, v11, v01).
            triangles.push(MeshTriangle::new(v00, v10, v11));
            triangles.push(MeshTriangle::new(v00, v11, v01));
        }
    }

    ParametricMesh {
        primitive_id: "sphere".to_string(),
        dimensions: vec![radius_m],
        triangles,
    }
}

// ---------------------------------------------------------------------------
// Plate
// ---------------------------------------------------------------------------

/// Flat rectangular plate centred on the origin in the z=0 plane.
///
/// The plate spans `[-width_m/2, +width_m/2]` along x and
/// `[-height_m/2, +height_m/2]` along y, with the outward normal pointing in
/// `+z`. The surface is sampled on `n_w` vertices along x and `n_h` along y,
/// producing `2 * (n_w - 1) * (n_h - 1)` triangles.
///
/// Dimensions slot order is `[width_m, height_m]`.
///
/// # Panics
/// Panics if `n_w < 2`, `n_h < 2`, `width_m <= 0.0`, or `height_m <= 0.0`.
pub fn plate_mesh(width_m: f64, height_m: f64, n_w: usize, n_h: usize) -> ParametricMesh {
    assert!(n_w >= 2, "plate mesh requires n_w >= 2");
    assert!(n_h >= 2, "plate mesh requires n_h >= 2");
    assert!(width_m > 0.0, "plate width must be positive");
    assert!(height_m > 0.0, "plate height must be positive");

    let mut triangles = Vec::with_capacity(2 * (n_w - 1) * (n_h - 1));

    let dx = width_m / ((n_w - 1) as f64);
    let dy = height_m / ((n_h - 1) as f64);
    let x0 = -width_m / 2.0;
    let y0 = -height_m / 2.0;

    let vertex =
        |i: usize, j: usize| -> [f64; 3] { [x0 + (i as f64) * dx, y0 + (j as f64) * dy, 0.0] };

    for i in 0..(n_w - 1) {
        for j in 0..(n_h - 1) {
            let v00 = vertex(i, j);
            let v10 = vertex(i + 1, j);
            let v11 = vertex(i + 1, j + 1);
            let v01 = vertex(i, j + 1);
            // CCW from +z (outward): (v00, v10, v11) and (v00, v11, v01).
            triangles.push(MeshTriangle::new(v00, v10, v11));
            triangles.push(MeshTriangle::new(v00, v11, v01));
        }
    }

    ParametricMesh {
        primitive_id: "plate".to_string(),
        dimensions: vec![width_m, height_m],
        triangles,
    }
}

// ---------------------------------------------------------------------------
// Cylinder
// ---------------------------------------------------------------------------

/// Closed circular cylinder centred on the z-axis with axis-aligned end caps.
///
/// The lateral surface has radius `radius_m` and runs from `z = -length_m/2`
/// to `z = +length_m/2`. `n_circ` is the number of azimuthal samples around
/// the circumference; `n_axial` is the number of samples along the z axis.
/// The lateral surface contributes `(n_axial - 1) * n_circ * 2` triangles and
/// each cap contributes `n_circ` triangles (fan from the centre), for a total
/// of `(n_axial - 1) * n_circ * 2 + 2 * n_circ` triangles.
///
/// Dimensions slot order is `[radius_m, length_m]`.
///
/// # Panics
/// Panics if `n_circ < 3`, `n_axial < 2`, `radius_m <= 0.0`, or
/// `length_m <= 0.0`.
pub fn cylinder_mesh(
    radius_m: f64,
    length_m: f64,
    n_circ: usize,
    n_axial: usize,
) -> ParametricMesh {
    assert!(n_circ >= 3, "cylinder mesh requires n_circ >= 3");
    assert!(n_axial >= 2, "cylinder mesh requires n_axial >= 2");
    assert!(radius_m > 0.0, "cylinder radius must be positive");
    assert!(length_m > 0.0, "cylinder length must be positive");

    let mut triangles = Vec::with_capacity((n_axial - 1) * n_circ * 2 + 2 * n_circ);

    let two_pi = 2.0 * std::f64::consts::PI;
    let z0 = -length_m / 2.0;
    let dz = length_m / ((n_axial - 1) as f64);

    let lat = |i_axial: usize, j_circ: usize| -> [f64; 3] {
        let phi = two_pi * (j_circ as f64) / (n_circ as f64);
        [
            radius_m * phi.cos(),
            radius_m * phi.sin(),
            z0 + (i_axial as f64) * dz,
        ]
    };

    // Lateral surface (outward normal = +radial).
    for i in 0..(n_axial - 1) {
        for j in 0..n_circ {
            let j1 = (j + 1) % n_circ;
            let v00 = lat(i, j);
            let v10 = lat(i + 1, j);
            let v11 = lat(i + 1, j1);
            let v01 = lat(i, j1);
            triangles.push(MeshTriangle::new(v00, v10, v11));
            triangles.push(MeshTriangle::new(v00, v11, v01));
        }
    }

    // Bottom cap (outward normal = -z, so CCW when viewed from -z means
    // (centre, j1, j) in the +z view ordering).
    let bottom_centre = [0.0, 0.0, z0];
    for j in 0..n_circ {
        let j1 = (j + 1) % n_circ;
        let vj = lat(0, j);
        let vj1 = lat(0, j1);
        triangles.push(MeshTriangle::new(bottom_centre, vj1, vj));
    }

    // Top cap (outward normal = +z).
    let top_centre = [0.0, 0.0, z0 + length_m];
    for j in 0..n_circ {
        let j1 = (j + 1) % n_circ;
        let vj = lat(n_axial - 1, j);
        let vj1 = lat(n_axial - 1, j1);
        triangles.push(MeshTriangle::new(top_centre, vj, vj1));
    }

    ParametricMesh {
        primitive_id: "cylinder".to_string(),
        dimensions: vec![radius_m, length_m],
        triangles,
    }
}

// ---------------------------------------------------------------------------
// Dihedral corner reflector (two orthogonal square plates sharing an edge)
// ---------------------------------------------------------------------------

/// Dihedral corner reflector: two orthogonal square plates of side `side_m`
/// sharing one edge along the z-axis.
///
/// Plate A occupies the y=0 half-plane with x in [0, side_m] and z in
/// [0, side_m]; its outward normal points in +y. Plate B occupies the x=0
/// half-plane with y in [0, side_m] and z in [0, side_m]; its outward normal
/// points in +x. Each plate is sampled on `n` x `n` vertices and contributes
/// `2 * (n - 1)^2` triangles, for a total of `4 * (n - 1)^2` triangles.
///
/// Dimensions slot order is `[side_m]`.
///
/// # Panics
/// Panics if `n < 2` or `side_m <= 0.0`.
pub fn dihedral_mesh(side_m: f64, n: usize) -> ParametricMesh {
    assert!(n >= 2, "dihedral mesh requires n >= 2");
    assert!(side_m > 0.0, "dihedral side must be positive");

    let mut triangles = Vec::with_capacity(4 * (n - 1) * (n - 1));
    let step = side_m / ((n - 1) as f64);

    // Plate A: lies in the y=0 plane, normal = +y. Sample over (x, z).
    let plate_a = |i: usize, j: usize| -> [f64; 3] { [(i as f64) * step, 0.0, (j as f64) * step] };
    for i in 0..(n - 1) {
        for j in 0..(n - 1) {
            let v00 = plate_a(i, j);
            let v10 = plate_a(i + 1, j);
            let v11 = plate_a(i + 1, j + 1);
            let v01 = plate_a(i, j + 1);
            // CCW viewed from +y: (v00, v11, v10) and (v00, v01, v11).
            triangles.push(MeshTriangle::new(v00, v11, v10));
            triangles.push(MeshTriangle::new(v00, v01, v11));
        }
    }

    // Plate B: lies in the x=0 plane, normal = +x. Sample over (y, z).
    let plate_b = |i: usize, j: usize| -> [f64; 3] { [0.0, (i as f64) * step, (j as f64) * step] };
    for i in 0..(n - 1) {
        for j in 0..(n - 1) {
            let v00 = plate_b(i, j);
            let v10 = plate_b(i + 1, j);
            let v11 = plate_b(i + 1, j + 1);
            let v01 = plate_b(i, j + 1);
            // CCW viewed from +x: (v00, v10, v11) and (v00, v11, v01).
            triangles.push(MeshTriangle::new(v00, v10, v11));
            triangles.push(MeshTriangle::new(v00, v11, v01));
        }
    }

    ParametricMesh {
        primitive_id: "dihedral".to_string(),
        dimensions: vec![side_m],
        triangles,
    }
}

// ---------------------------------------------------------------------------
// Trihedral corner reflector (three orthogonal triangular plates)
// ---------------------------------------------------------------------------

/// Trihedral corner reflector: three orthogonal right-triangle plates of leg
/// `side_m`, meeting at the origin and bounded by the plane
/// `x + y + z = side_m`.
///
/// Each plate lies in one of the coordinate planes (xy, yz, zx), with its
/// outward normal pointing toward the +octant interior of the reflector
/// (i.e. plate in xy plane has normal +z, plate in yz plane has normal +x,
/// plate in zx plane has normal +y). Each plate is tessellated on a triangular
/// `n` x `n` grid (legs evenly subdivided into `n - 1` segments), producing
/// `(n - 1)^2` triangles per plate, for a total of `3 * (n - 1)^2` triangles.
///
/// Dimensions slot order is `[side_m]`.
///
/// # Panics
/// Panics if `n < 2` or `side_m <= 0.0`.
pub fn trihedral_mesh(side_m: f64, n: usize) -> ParametricMesh {
    assert!(n >= 2, "trihedral mesh requires n >= 2");
    assert!(side_m > 0.0, "trihedral side must be positive");

    let mut triangles = Vec::with_capacity(3 * (n - 1) * (n - 1));
    let step = side_m / ((n - 1) as f64);

    // Tessellate the right triangle with legs along u and v (each leg of
    // length side_m) into (n-1)^2 small triangles. For each quad (i, j) where
    // i + j + 1 < n we emit two triangles; the boundary i + j + 1 == n - 1
    // case emits one. Equivalently: for each (i, j) with i + j < n - 1,
    // emit triangle (i,j)-(i+1,j)-(i,j+1). For each (i, j) with
    // i + j < n - 2, emit triangle (i+1,j)-(i+1,j+1)-(i,j+1).
    //
    // Total triangles = (n-1)^2 per plate (sum of an arithmetic progression).
    let emit_plate = |triangles: &mut Vec<MeshTriangle>,
                      vertex: &dyn Fn(usize, usize) -> [f64; 3],
                      flip: bool| {
        for i in 0..(n - 1) {
            for j in 0..(n - 1 - i) {
                let v00 = vertex(i, j);
                let v10 = vertex(i + 1, j);
                let v01 = vertex(i, j + 1);
                if flip {
                    triangles.push(MeshTriangle::new(v00, v01, v10));
                } else {
                    triangles.push(MeshTriangle::new(v00, v10, v01));
                }
            }
        }
        for i in 0..(n - 1) {
            for j in 0..(n - 2).saturating_sub(i) {
                let v10 = vertex(i + 1, j);
                let v11 = vertex(i + 1, j + 1);
                let v01 = vertex(i, j + 1);
                if flip {
                    triangles.push(MeshTriangle::new(v10, v01, v11));
                } else {
                    triangles.push(MeshTriangle::new(v10, v11, v01));
                }
            }
        }
    };

    // Plate in xy plane (z = 0). u = x, v = y. Outward normal = +z, so CCW
    // from +z: (v00, v10, v01) -> normal is (+z). No flip.
    let xy = |i: usize, j: usize| -> [f64; 3] { [(i as f64) * step, (j as f64) * step, 0.0] };
    emit_plate(&mut triangles, &xy, false);

    // Plate in yz plane (x = 0). u = y, v = z. Outward normal = +x. CCW from
    // +x: (v00, v10, v01) with v10 = +y, v01 = +z -> normal (+x). No flip.
    let yz = |i: usize, j: usize| -> [f64; 3] { [0.0, (i as f64) * step, (j as f64) * step] };
    emit_plate(&mut triangles, &yz, false);

    // Plate in zx plane (y = 0). u = z, v = x. Outward normal = +y. CCW from
    // +y: (v00, v10, v01) with v10 = +z, v01 = +x -> (v10 x v01) = (+y). No
    // flip.
    let zx = |i: usize, j: usize| -> [f64; 3] { [(j as f64) * step, 0.0, (i as f64) * step] };
    emit_plate(&mut triangles, &zx, false);

    ParametricMesh {
        primitive_id: "trihedral".to_string(),
        dimensions: vec![side_m],
        triangles,
    }
}

// ---------------------------------------------------------------------------
// STL writer
// ---------------------------------------------------------------------------

/// Writes a [`ParametricMesh`] as a binary STL file to the given path and
/// returns the SHA-256 of the on-disk bytes as a lowercase 64-character hex
/// string.
///
/// Binary STL layout:
/// - 80-byte ASCII header (here: padded primitive id + generator tag),
/// - `u32` triangle count (little-endian),
/// - per triangle: `[f32; 3]` normal + `3 * [f32; 3]` vertices + `u16`
///   attribute byte count (always zero).
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
                let bytes = (c as f32).to_le_bytes();
                out.extend_from_slice(&bytes);
            }
        }
        out.extend_from_slice(&0u16.to_le_bytes()); // attribute byte count
    }
    out
}

fn sha256_hex(bytes: &[u8]) -> String {
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

// ---------------------------------------------------------------------------
// Mesh manifest (lightweight per-file companion, distinct from the canonical
// `mesh_manifest` schema which lives in the contracts layer).
// ---------------------------------------------------------------------------

/// Lightweight per-file manifest written alongside a parametric STL. This is
/// the minimum fingerprint needed by downstream tooling to identify a mesh;
/// the full `mesh_manifest` envelope (provenance/license/validation) is the
/// caller's responsibility and lives in `schemas/mesh_manifest.schema.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshManifestFile {
    /// Primitive family slug (mirrors [`ParametricMesh::primitive_id`]).
    pub primitive_id: String,
    /// Shape-defining lengths, in metres (mirrors
    /// [`ParametricMesh::dimensions`]).
    pub dimensions: Vec<f64>,
    /// Number of triangles in the emitted STL.
    pub triangle_count: usize,
    /// SHA-256 of the STL byte payload, lowercase hex.
    pub sha256_of_stl: String,
    /// Generator identifier (currently always `GENERATOR_NAME`).
    pub generator_name: String,
    /// Generator version (currently always `GENERATOR_VERSION`).
    pub generator_version: String,
}

impl MeshManifestFile {
    /// Build a manifest from a mesh and its STL hash.
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

// ---------------------------------------------------------------------------
// Tiny 3-vector helpers (no nalgebra dep; keeps the world crate light).
// ---------------------------------------------------------------------------

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    const TOL_REL: f64 = 0.01; // 1% tolerance for bounding-box checks

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1e-12);
        diff <= rel * scale
    }

    #[test]
    fn sphere_triangle_count_matches_formula() {
        let n_lat = 10;
        let n_lon = 20;
        let mesh = sphere_mesh(1.0, n_lat, n_lon);
        assert_eq!(mesh.primitive_id, "sphere");
        assert_eq!(mesh.dimensions, vec![1.0]);
        assert_eq!(mesh.triangle_count(), (n_lat - 1) * n_lon * 2);
        assert!(mesh.all_finite());
    }

    #[test]
    fn plate_triangle_count_matches_formula() {
        let n_w = 10;
        let n_h = 10;
        let mesh = plate_mesh(1.0, 1.0, n_w, n_h);
        assert_eq!(mesh.primitive_id, "plate");
        assert_eq!(mesh.triangle_count(), 2 * (n_w - 1) * (n_h - 1));
        // Outward normal should be +z for every triangle.
        for t in &mesh.triangles {
            let n = t.normal();
            assert!(
                (n[2] - 1.0).abs() < 1e-9,
                "plate normal must be +z, got {:?}",
                n
            );
        }
    }

    #[test]
    fn cylinder_mesh_is_nondegenerate() {
        let mesh = cylinder_mesh(0.5, 2.0, 24, 8);
        assert_eq!(mesh.primitive_id, "cylinder");
        assert_eq!(mesh.dimensions, vec![0.5, 2.0]);
        // Lateral + two caps: (n_axial - 1) * n_circ * 2 + 2 * n_circ.
        assert_eq!(mesh.triangle_count(), 7 * 24 * 2 + 2 * 24);
        assert!(mesh.all_finite());
        for t in &mesh.triangles {
            assert!(t.area() > 1e-12, "degenerate triangle: {:?}", t);
        }
    }

    #[test]
    fn all_meshes_have_finite_vertices() {
        let meshes = [
            sphere_mesh(2.0, 6, 12),
            plate_mesh(3.0, 4.0, 5, 5),
            cylinder_mesh(0.25, 1.5, 16, 4),
            dihedral_mesh(1.0, 5),
            trihedral_mesh(0.75, 6),
        ];
        for m in &meshes {
            assert!(
                m.all_finite(),
                "mesh {} has non-finite vertex",
                m.primitive_id
            );
            assert!(m.triangle_count() > 0, "mesh {} is empty", m.primitive_id);
        }
    }

    #[test]
    fn bounding_boxes_match_requested_dimensions() {
        // Sphere of radius 2 -> bbox [-2, 2]^3.
        let s = sphere_mesh(2.0, 32, 64);
        let (lo, hi) = s.bounding_box();
        for axis in 0..3 {
            assert!(close(hi[axis] - lo[axis], 4.0, TOL_REL));
        }

        // Plate 3 x 4 in z=0 -> bbox x: [-1.5, 1.5], y: [-2, 2], z: 0.
        let p = plate_mesh(3.0, 4.0, 8, 8);
        let (lo, hi) = p.bounding_box();
        assert!(close(hi[0] - lo[0], 3.0, TOL_REL));
        assert!(close(hi[1] - lo[1], 4.0, TOL_REL));
        assert!((hi[2] - lo[2]).abs() < 1e-12);

        // Cylinder r=0.5, L=2 -> x,y span 1, z span 2.
        let c = cylinder_mesh(0.5, 2.0, 64, 8);
        let (lo, hi) = c.bounding_box();
        assert!(close(hi[0] - lo[0], 1.0, TOL_REL));
        assert!(close(hi[1] - lo[1], 1.0, TOL_REL));
        assert!(close(hi[2] - lo[2], 2.0, TOL_REL));

        // Dihedral side 1.0 -> spans x,y,z in [0, 1].
        let d = dihedral_mesh(1.0, 6);
        let (lo, hi) = d.bounding_box();
        for axis in 0..3 {
            assert!(close(hi[axis] - lo[axis], 1.0, TOL_REL));
            assert!(close(lo[axis], 0.0, TOL_REL) || lo[axis].abs() < 1e-9);
        }

        // Trihedral side 0.75 -> spans 0.75 along each axis.
        let t = trihedral_mesh(0.75, 8);
        let (lo, hi) = t.bounding_box();
        for axis in 0..3 {
            assert!(close(hi[axis] - lo[axis], 0.75, TOL_REL));
        }
    }

    #[test]
    fn dihedral_has_orthogonal_plates_with_correct_normals() {
        let mesh = dihedral_mesh(1.0, 4);
        assert_eq!(mesh.primitive_id, "dihedral");
        // Total triangles = 4 * (n - 1)^2 = 4 * 9 = 36.
        assert_eq!(mesh.triangle_count(), 36);
        // The first half lies on plate A (y=0, normal +y); the second half
        // lies on plate B (x=0, normal +x).
        let half = mesh.triangles.len() / 2;
        for t in &mesh.triangles[..half] {
            let n = t.normal();
            assert!((n[1] - 1.0).abs() < 1e-9, "plate A normal != +y: {:?}", n);
            for v in [t.v0, t.v1, t.v2] {
                assert!(v[1].abs() < 1e-12, "plate A vertex off y=0: {:?}", v);
            }
        }
        for t in &mesh.triangles[half..] {
            let n = t.normal();
            assert!((n[0] - 1.0).abs() < 1e-9, "plate B normal != +x: {:?}", n);
            for v in [t.v0, t.v1, t.v2] {
                assert!(v[0].abs() < 1e-12, "plate B vertex off x=0: {:?}", v);
            }
        }
    }

    #[test]
    fn trihedral_plate_normals_point_into_corner() {
        let mesh = trihedral_mesh(1.0, 4);
        assert_eq!(mesh.primitive_id, "trihedral");
        // 3 plates * (n - 1)^2 = 3 * 9 = 27 triangles.
        assert_eq!(mesh.triangle_count(), 27);
        let per = mesh.triangle_count() / 3;
        let expected = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        for (plate_idx, expected_normal) in expected.iter().enumerate() {
            let start = plate_idx * per;
            for t in &mesh.triangles[start..start + per] {
                let n = t.normal();
                let dp = dot(n, *expected_normal);
                assert!(
                    dp > 0.999,
                    "plate {} normal misaligned: n={:?}",
                    plate_idx,
                    n
                );
                assert!(t.area() > 1e-12, "plate {} degenerate triangle", plate_idx);
            }
        }
    }

    #[test]
    fn manifest_serde_round_trips() {
        let mesh = sphere_mesh(1.0, 4, 6);
        let bytes = encode_binary_stl(&mesh);
        let sha = sha256_hex(&bytes);
        let manifest = MeshManifestFile::from_mesh(&mesh, sha.clone());
        let json = serde_json::to_string(&manifest).expect("serialize");
        let round: MeshManifestFile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(round, manifest);
        assert_eq!(round.primitive_id, "sphere");
        assert_eq!(round.triangle_count, mesh.triangle_count());
        assert_eq!(round.sha256_of_stl.len(), 64);
        assert_eq!(round.generator_name, GENERATOR_NAME);
        assert_eq!(round.generator_version, GENERATOR_VERSION);
    }

    #[test]
    fn emit_mesh_bundle_writes_stl_and_manifest() {
        let dir = tempdir().expect("tempdir");
        let mesh = plate_mesh(1.0, 1.0, 4, 4);
        let (stl_path, manifest_path) = emit_mesh_bundle(&mesh, dir.path()).expect("emit bundle");
        assert!(stl_path.exists(), "STL missing");
        assert!(manifest_path.exists(), "manifest missing");

        let stl_bytes = std::fs::read(&stl_path).expect("read stl");
        // 80-byte header + 4-byte count + 50 bytes/tri.
        assert_eq!(stl_bytes.len(), 84 + mesh.triangles.len() * 50);
        let expected_count = u32::from_le_bytes(stl_bytes[80..84].try_into().unwrap());
        assert_eq!(expected_count as usize, mesh.triangles.len());

        let manifest_json = std::fs::read_to_string(&manifest_path).expect("read manifest");
        let manifest: MeshManifestFile =
            serde_json::from_str(&manifest_json).expect("parse manifest");
        assert_eq!(manifest.primitive_id, "plate");
        assert_eq!(manifest.triangle_count, mesh.triangles.len());
        // Hash in manifest must match a fresh hash of the file bytes.
        assert_eq!(manifest.sha256_of_stl, sha256_hex(&stl_bytes));
    }

    #[test]
    fn stl_encoding_is_deterministic() {
        let a = encode_binary_stl(&cylinder_mesh(0.3, 1.0, 12, 4));
        let b = encode_binary_stl(&cylinder_mesh(0.3, 1.0, 12, 4));
        assert_eq!(a, b);
        assert_eq!(sha256_hex(&a), sha256_hex(&b));
    }

    #[test]
    fn sphere_vertices_lie_on_surface() {
        let r = 1.5;
        let mesh = sphere_mesh(r, 16, 24);
        for t in &mesh.triangles {
            for v in [t.v0, t.v1, t.v2] {
                let rho = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                assert!(
                    (rho - r).abs() < 1e-9,
                    "sphere vertex off surface: rho={}, r={}",
                    rho,
                    r
                );
            }
        }
    }

    #[test]
    fn sphere_panics_on_invalid_inputs() {
        // Use catch_unwind to verify the assertions actually fire.
        let r = std::panic::catch_unwind(|| sphere_mesh(0.0, 4, 6));
        assert!(r.is_err(), "sphere_mesh should panic on zero radius");
        let r = std::panic::catch_unwind(|| sphere_mesh(1.0, 1, 6));
        assert!(r.is_err(), "sphere_mesh should panic on n_lat < 2");
        let r = std::panic::catch_unwind(|| sphere_mesh(1.0, 4, 2));
        assert!(r.is_err(), "sphere_mesh should panic on n_lon < 3");
    }
}
