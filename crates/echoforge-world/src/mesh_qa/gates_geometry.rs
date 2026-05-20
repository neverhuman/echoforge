//! QA gates covering per-vertex and per-triangle geometry properties.
//!
//! Gates: finite vertices, degenerate triangles, consistent winding,
//! bounded aspect ratio.

use super::math::{dot3, edge_len, sub3, tri_centroid};
use super::{QaGateResult, QaStatus};
use crate::mesh::ParametricMesh;

/// Gate 1 — every vertex coordinate is finite (no NaN, no +/-Inf).
///
/// FAIL if any triangle carries a non-finite coordinate; `value` reports the
/// count of offending triangles.
pub fn gate_finite_vertices(mesh: &ParametricMesh) -> QaGateResult {
    let mut bad = 0usize;
    for t in &mesh.triangles {
        if !t.all_finite() {
            bad += 1;
        }
    }
    if bad == 0 {
        QaGateResult {
            name: "gate_finite_vertices".to_string(),
            status: QaStatus::Pass,
            message: format!("all {} triangle vertices are finite", mesh.triangles.len()),
            value: 0.0,
        }
    } else {
        QaGateResult {
            name: "gate_finite_vertices".to_string(),
            status: QaStatus::Fail,
            message: format!("{} triangle(s) carry non-finite coordinates", bad),
            value: bad as f64,
        }
    }
}

/// Minimum triangle area, in m^2, below which a triangle is flagged as
/// degenerate.
pub const DEGENERATE_AREA_THRESHOLD_M2: f64 = 1e-12;

/// Gate 2 — every triangle has area strictly greater than
/// [`DEGENERATE_AREA_THRESHOLD_M2`].
///
/// FAIL if any triangle is degenerate; `value` reports the count.
pub fn gate_no_degenerate_triangles(mesh: &ParametricMesh) -> QaGateResult {
    let mut bad = 0usize;
    let mut worst = f64::INFINITY;
    for t in &mesh.triangles {
        let a = t.area();
        // Use !(a > T) rather than (a <= T) so NaN areas are correctly
        // flagged as degenerate; the NaN-rejecting behaviour is intentional.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let is_degenerate = !(a > DEGENERATE_AREA_THRESHOLD_M2);
        if is_degenerate {
            bad += 1;
            if a < worst {
                worst = a;
            }
        }
    }
    if bad == 0 {
        QaGateResult {
            name: "gate_no_degenerate_triangles".to_string(),
            status: QaStatus::Pass,
            message: format!(
                "all {} triangles have area > {:e} m^2",
                mesh.triangles.len(),
                DEGENERATE_AREA_THRESHOLD_M2
            ),
            value: 0.0,
        }
    } else {
        QaGateResult {
            name: "gate_no_degenerate_triangles".to_string(),
            status: QaStatus::Fail,
            message: format!(
                "{} triangle(s) below {:e} m^2 (worst area {:.3e} m^2)",
                bad, DEGENERATE_AREA_THRESHOLD_M2, worst
            ),
            value: bad as f64,
        }
    }
}

/// Maximum per-triangle aspect ratio (max-edge / min-edge) before the WARN
/// threshold trips.
pub const ASPECT_RATIO_WARN: f64 = 100.0;
/// Mean per-triangle aspect ratio above which the FAIL threshold trips.
pub const ASPECT_RATIO_FAIL_MEAN: f64 = 50.0;

/// Gate 4 — bounded triangle aspect ratio (longest edge / shortest edge).
///
/// WARN if any triangle exceeds [`ASPECT_RATIO_WARN`]; FAIL if the mean
/// aspect ratio over all triangles exceeds [`ASPECT_RATIO_FAIL_MEAN`].
/// `value` reports the mean aspect ratio. Degenerate triangles (any edge of
/// length zero) are excluded from the mean.
pub fn gate_bounded_aspect_ratio(mesh: &ParametricMesh) -> QaGateResult {
    if mesh.triangles.is_empty() {
        return QaGateResult {
            name: "gate_bounded_aspect_ratio".to_string(),
            status: QaStatus::Pass,
            message: "empty mesh".to_string(),
            value: 0.0,
        };
    }
    let mut max_ratio: f64 = 0.0;
    let mut sum_ratio: f64 = 0.0;
    let mut counted = 0usize;
    let mut over_warn = 0usize;
    for t in &mesh.triangles {
        let e0 = edge_len(t.v0, t.v1);
        let e1 = edge_len(t.v1, t.v2);
        let e2 = edge_len(t.v2, t.v0);
        let min = e0.min(e1).min(e2);
        let max = e0.max(e1).max(e2);
        if !min.is_finite() || min <= 0.0 {
            continue;
        }
        let r = max / min;
        sum_ratio += r;
        counted += 1;
        if r > max_ratio {
            max_ratio = r;
        }
        if r > ASPECT_RATIO_WARN {
            over_warn += 1;
        }
    }
    if counted == 0 {
        return QaGateResult {
            name: "gate_bounded_aspect_ratio".to_string(),
            status: QaStatus::Pass,
            message: "no measurable edges to inspect".to_string(),
            value: 0.0,
        };
    }
    let mean = sum_ratio / counted as f64;
    let (status, message) = if mean > ASPECT_RATIO_FAIL_MEAN {
        (
            QaStatus::Fail,
            format!(
                "mean aspect ratio {:.2} exceeds {:.0} (max {:.2}, {} over warn-threshold)",
                mean, ASPECT_RATIO_FAIL_MEAN, max_ratio, over_warn
            ),
        )
    } else if over_warn > 0 {
        (
            QaStatus::Warn,
            format!(
                "{} triangle(s) exceed aspect ratio {:.0} (max {:.2}, mean {:.2})",
                over_warn, ASPECT_RATIO_WARN, max_ratio, mean
            ),
        )
    } else {
        (
            QaStatus::Pass,
            format!(
                "all {} triangles within aspect ratio {:.0} (max {:.2}, mean {:.2})",
                counted, ASPECT_RATIO_WARN, max_ratio, mean
            ),
        )
    };
    QaGateResult {
        name: "gate_bounded_aspect_ratio".to_string(),
        status,
        message,
        value: mean,
    }
}

/// Maximum allowed fraction of "inward-pointing" triangles before
/// [`gate_consistent_winding`] flips from WARN to FAIL.
pub const WINDING_FAIL_FRACTION: f64 = 0.05;

/// Gate 3 — majority of triangle normals agree with the outward direction
/// from the bounding-sphere centre.
///
/// We compute the bounding-sphere centroid (mean of the AABB corners), then
/// for every triangle whose normal is non-zero, we check the sign of
/// `dot(normal, vertex_centroid - sphere_centre)`. The gate PASSes if zero
/// triangles are inward, WARNs if less than [`WINDING_FAIL_FRACTION`] are,
/// and FAILs above that threshold. `value` reports the fraction of inward
/// triangles.
pub fn gate_consistent_winding(mesh: &ParametricMesh) -> QaGateResult {
    if mesh.triangles.is_empty() {
        return QaGateResult {
            name: "gate_consistent_winding".to_string(),
            status: QaStatus::Pass,
            message: "empty mesh has no winding to check".to_string(),
            value: 0.0,
        };
    }

    let (lo, hi) = mesh.bounding_box();
    let centre = [
        0.5 * (lo[0] + hi[0]),
        0.5 * (lo[1] + hi[1]),
        0.5 * (lo[2] + hi[2]),
    ];

    let mut counted = 0usize;
    let mut inward = 0usize;
    for t in &mesh.triangles {
        let n = t.normal();
        if n[0] == 0.0 && n[1] == 0.0 && n[2] == 0.0 {
            continue;
        }
        let c = tri_centroid(t);
        let r = sub3(c, centre);
        let d = dot3(n, r);
        if d < 0.0 {
            inward += 1;
        }
        counted += 1;
    }

    if counted == 0 {
        return QaGateResult {
            name: "gate_consistent_winding".to_string(),
            status: QaStatus::Pass,
            message: "no non-degenerate triangles to inspect".to_string(),
            value: 0.0,
        };
    }

    let frac = inward as f64 / counted as f64;
    let (status, message) = if inward == 0 {
        (
            QaStatus::Pass,
            format!("all {} triangles wind CCW from outside", counted),
        )
    } else if frac < WINDING_FAIL_FRACTION {
        (
            QaStatus::Warn,
            format!(
                "{}/{} triangles ({:.2}%) appear inward-wound",
                inward,
                counted,
                100.0 * frac
            ),
        )
    } else {
        (
            QaStatus::Fail,
            format!(
                "{}/{} triangles ({:.2}%) inward-wound exceeds {:.0}% budget",
                inward,
                counted,
                100.0 * frac,
                100.0 * WINDING_FAIL_FRACTION
            ),
        )
    };
    QaGateResult {
        name: "gate_consistent_winding".to_string(),
        status,
        message,
        value: frac,
    }
}
