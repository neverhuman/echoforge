//! Private math helpers for mesh QA gate functions.
//!
//! All functions operate on `[f64; 3]` vectors and use only `std` arithmetic.

use crate::mesh::MeshTriangle;

pub(super) fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(super) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(super) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(super) fn vector_is_zero(v: [f64; 3]) -> bool {
    v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0
}

pub(super) fn edge_len(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub3(a, b);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

pub(super) fn tri_centroid(t: &MeshTriangle) -> [f64; 3] {
    [
        (t.v0[0] + t.v1[0] + t.v2[0]) / 3.0,
        (t.v0[1] + t.v1[1] + t.v2[1]) / 3.0,
        (t.v0[2] + t.v1[2] + t.v2[2]) / 3.0,
    ]
}

// ---------------------------------------------------------------------------
// Edge map: deterministic edge -> [triangle indices] adjacency.
// ---------------------------------------------------------------------------

/// Quantisation grid (metres) for matching vertices into shared edges.
/// One nanometre stays well below any realistic mesh tolerance and above
/// f64 ULP noise from the generators in `mesh.rs`.
const EDGE_QUANTUM_M: f64 = 1e-9;

type EdgeKey = ((i64, i64, i64), (i64, i64, i64));

fn quantise(v: [f64; 3]) -> (i64, i64, i64) {
    fn q(x: f64) -> i64 {
        if !x.is_finite() {
            // NaN/Inf vertices get a distinctive sentinel bucket so they
            // never accidentally match a finite vertex.
            i64::MIN
        } else {
            (x / EDGE_QUANTUM_M).round() as i64
        }
    }
    (q(v[0]), q(v[1]), q(v[2]))
}

fn edge_key(a: [f64; 3], b: [f64; 3]) -> EdgeKey {
    let qa = quantise(a);
    let qb = quantise(b);
    if qa <= qb { (qa, qb) } else { (qb, qa) }
}

pub(super) fn build_edge_map(
    mesh: &crate::mesh::ParametricMesh,
) -> std::collections::BTreeMap<EdgeKey, Vec<usize>> {
    let mut map: std::collections::BTreeMap<EdgeKey, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (idx, t) in mesh.triangles.iter().enumerate() {
        for (a, b) in [(t.v0, t.v1), (t.v1, t.v2), (t.v2, t.v0)] {
            map.entry(edge_key(a, b)).or_default().push(idx);
        }
    }
    map
}
