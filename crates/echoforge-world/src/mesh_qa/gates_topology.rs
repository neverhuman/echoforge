//! QA gates covering topological and frequency-domain properties.
//!
//! Gates: dihedral angle, watertight, bounded volume vs bbox,
//! maximum electrical size.

use super::math::{build_edge_map, cross3, dot3, vector_is_zero};
use super::{QaGateResult, QaStatus};
use crate::mesh::ParametricMesh;

/// Minimum dihedral angle (degrees) below which the WARN threshold trips.
pub const DIHEDRAL_MIN_DEG: f64 = 10.0;
/// Maximum dihedral angle (degrees) above which the WARN threshold trips.
pub const DIHEDRAL_MAX_DEG: f64 = 170.0;

/// Gate 5 — dihedral angle between adjacent triangles is within
/// `[DIHEDRAL_MIN_DEG, DIHEDRAL_MAX_DEG]`.
///
/// WARN if any shared-edge pair is outside the band; `value` reports the
/// count of out-of-band pairs. Adjacency is determined by edge sharing using
/// integer-quantised vertex coordinates (rounded to 1e-9 m). Coplanar seams
/// (180-deg dihedral) are intentionally excluded from the warn condition.
pub fn gate_bounded_dihedral_angle(mesh: &ParametricMesh) -> QaGateResult {
    if mesh.triangles.is_empty() {
        return QaGateResult {
            name: "gate_bounded_dihedral_angle".to_string(),
            status: QaStatus::Pass,
            message: "empty mesh".to_string(),
            value: 0.0,
        };
    }
    let edge_map = build_edge_map(mesh);
    let mut shared_pairs = 0usize;
    let mut violations = 0usize;
    let mut min_seen: f64 = f64::INFINITY;
    let mut max_seen: f64 = f64::NEG_INFINITY;
    for (_edge, tris) in edge_map.iter() {
        if tris.len() < 2 {
            continue;
        }
        for i in 0..tris.len() {
            for j in (i + 1)..tris.len() {
                let a = &mesh.triangles[tris[i]];
                let b = &mesh.triangles[tris[j]];
                let na = a.normal();
                let nb = b.normal();
                if vector_is_zero(na) || vector_is_zero(nb) {
                    continue;
                }
                let d = dot3(na, nb).clamp(-1.0, 1.0);
                let deg = d.acos().to_degrees();
                shared_pairs += 1;
                if deg < min_seen {
                    min_seen = deg;
                }
                if deg > max_seen {
                    max_seen = deg;
                }
                // Coplanar (flat) seams and back-to-back degenerate pairs are
                // expected; only the band (0, 10) U (170, 180) triggers a warn.
                if deg < DIHEDRAL_MIN_DEG || (deg > DIHEDRAL_MAX_DEG && deg < 180.0 - 1e-6) {
                    violations += 1;
                }
            }
        }
    }
    if shared_pairs == 0 {
        return QaGateResult {
            name: "gate_bounded_dihedral_angle".to_string(),
            status: QaStatus::Pass,
            message: "no shared edges to inspect".to_string(),
            value: 0.0,
        };
    }
    let (status, message) = if violations == 0 {
        (
            QaStatus::Pass,
            format!(
                "all {} adjacent-triangle pairs within [{:.0}, {:.0}] deg (range {:.2}..{:.2})",
                shared_pairs, DIHEDRAL_MIN_DEG, DIHEDRAL_MAX_DEG, min_seen, max_seen
            ),
        )
    } else {
        (
            QaStatus::Warn,
            format!(
                "{} of {} pairs outside [{:.0}, {:.0}] deg (range {:.2}..{:.2})",
                violations, shared_pairs, DIHEDRAL_MIN_DEG, DIHEDRAL_MAX_DEG, min_seen, max_seen
            ),
        )
    };
    QaGateResult {
        name: "gate_bounded_dihedral_angle".to_string(),
        status,
        message,
        value: violations as f64,
    }
}

/// Gate 6 — watertight (each edge shared by exactly two triangles).
///
/// PASSes if no open edges remain; WARNs if any edge is shared by fewer than
/// two triangles (open boundary) or by more than two (non-manifold seam).
/// `value` reports the count of open / non-manifold edges. Open primitives
/// (`plate`, `dihedral`, `trihedral`) are still inspected — WARN is the
/// expected outcome and downstream tooling should allow it.
pub fn gate_watertight(mesh: &ParametricMesh) -> QaGateResult {
    if mesh.triangles.is_empty() {
        return QaGateResult {
            name: "gate_watertight".to_string(),
            status: QaStatus::Warn,
            message: "empty mesh cannot be watertight".to_string(),
            value: 0.0,
        };
    }
    let edge_map = build_edge_map(mesh);
    let mut open = 0usize;
    let mut nonmanifold = 0usize;
    for (_edge, tris) in edge_map.iter() {
        match tris.len() {
            0 | 1 => open += 1,
            2 => {}
            _ => nonmanifold += 1,
        }
    }
    let bad = open + nonmanifold;
    let primitive = mesh.primitive_id.as_str();
    let expected_open = matches!(primitive, "plate" | "dihedral" | "trihedral");
    if bad == 0 {
        QaGateResult {
            name: "gate_watertight".to_string(),
            status: QaStatus::Pass,
            message: format!("all {} edges shared by exactly 2 triangles", edge_map.len()),
            value: 0.0,
        }
    } else {
        let suffix = if expected_open {
            " (expected for open primitive)"
        } else {
            ""
        };
        QaGateResult {
            name: "gate_watertight".to_string(),
            status: QaStatus::Warn,
            message: format!(
                "{} open edge(s), {} non-manifold edge(s){}",
                open, nonmanifold, suffix
            ),
            value: bad as f64,
        }
    }
}

/// Lower bound on enclosed volume as a fraction of the bounding-box volume.
pub const VOLUME_LO_FRAC: f64 = 0.05;
/// Upper bound on enclosed volume as a fraction of the bounding-box volume.
/// Real solids have V/V_bbox <= 1.0; a small margin avoids spurious failures
/// from numerical noise.
pub const VOLUME_HI_FRAC: f64 = 1.01;

/// Gate 7 — enclosed volume (signed, via the divergence theorem on triangle
/// normals) is within a sane fraction of the bounding-box volume.
///
/// Open primitives (`plate`, `dihedral`, `trihedral`) are skipped because
/// their signed volume is not physically meaningful.
/// `value` reports the ratio `V / V_bbox`.
pub fn gate_bounded_volume_vs_box(mesh: &ParametricMesh) -> QaGateResult {
    let primitive = mesh.primitive_id.as_str();
    if matches!(primitive, "plate" | "dihedral" | "trihedral") {
        return QaGateResult {
            name: "gate_bounded_volume_vs_box".to_string(),
            status: QaStatus::Skipped,
            message: format!(
                "primitive '{}' is open; volume gate not applicable",
                primitive
            ),
            value: 0.0,
        };
    }
    if mesh.triangles.is_empty() {
        return QaGateResult {
            name: "gate_bounded_volume_vs_box".to_string(),
            status: QaStatus::Skipped,
            message: "empty mesh".to_string(),
            value: 0.0,
        };
    }
    let mut volume = 0.0_f64;
    for t in &mesh.triangles {
        let cr = cross3(t.v1, t.v2);
        volume += dot3(t.v0, cr);
    }
    volume /= 6.0;
    let (lo, hi) = mesh.bounding_box();
    let bbox_vol = (hi[0] - lo[0]).abs() * (hi[1] - lo[1]).abs() * (hi[2] - lo[2]).abs();
    if !bbox_vol.is_finite() || bbox_vol <= 0.0 {
        return QaGateResult {
            name: "gate_bounded_volume_vs_box".to_string(),
            status: QaStatus::Skipped,
            message: "bounding box has zero volume; ratio undefined".to_string(),
            value: 0.0,
        };
    }
    let ratio = volume / bbox_vol;
    let (status, message) = if !ratio.is_finite() {
        (
            QaStatus::Warn,
            format!(
                "non-finite volume ratio (V={:.3e}, V_bbox={:.3e})",
                volume, bbox_vol
            ),
        )
    } else if (VOLUME_LO_FRAC..=VOLUME_HI_FRAC).contains(&ratio) {
        (
            QaStatus::Pass,
            format!(
                "volume {:.3e} m^3 within {:.0}%..{:.0}% of bbox volume {:.3e} m^3 (ratio {:.3})",
                volume,
                100.0 * VOLUME_LO_FRAC,
                100.0 * VOLUME_HI_FRAC,
                bbox_vol,
                ratio
            ),
        )
    } else {
        (
            QaStatus::Warn,
            format!(
                "volume ratio {:.3} outside [{:.2}, {:.2}] (V={:.3e}, V_bbox={:.3e})",
                ratio, VOLUME_LO_FRAC, VOLUME_HI_FRAC, volume, bbox_vol
            ),
        )
    };
    QaGateResult {
        name: "gate_bounded_volume_vs_box".to_string(),
        status,
        message,
        value: ratio,
    }
}

/// Speed of light in vacuum, in metres per second.
pub const SPEED_OF_LIGHT_M_PER_S: f64 = 299_792_458.0;
/// Maximum electrical size (max-object-extent / wavelength) above which
/// [`gate_max_electrical_size`] warns.
pub const ELECTRICAL_SIZE_WARN_LAMBDA: f64 = 1000.0;

/// Gate 8 — frequency-driven electrical-size sanity check.
///
/// If `freq_hz` is `Some`, computes the largest bounding-box extent divided
/// by the wavelength `c / f`. WARNs when the ratio exceeds
/// [`ELECTRICAL_SIZE_WARN_LAMBDA`]. `value` reports the electrical-size ratio.
pub fn gate_max_electrical_size(mesh: &ParametricMesh, freq_hz: Option<f64>) -> QaGateResult {
    let Some(f) = freq_hz else {
        return QaGateResult {
            name: "gate_max_electrical_size".to_string(),
            status: QaStatus::Skipped,
            message: "no frequency provided; electrical-size gate not applicable".to_string(),
            value: 0.0,
        };
    };
    // Use !(f > 0.0) rather than (f <= 0.0) so NaN frequencies are also
    // rejected; the NaN-rejecting behaviour is intentional.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let f_invalid = !(f > 0.0) || !f.is_finite();
    if f_invalid {
        return QaGateResult {
            name: "gate_max_electrical_size".to_string(),
            status: QaStatus::Skipped,
            message: format!("non-positive frequency {} Hz; gate not applicable", f),
            value: 0.0,
        };
    }
    let (lo, hi) = mesh.bounding_box();
    let extent = (hi[0] - lo[0])
        .max(hi[1] - lo[1])
        .max(hi[2] - lo[2])
        .max(0.0);
    let wavelength = SPEED_OF_LIGHT_M_PER_S / f;
    if wavelength <= 0.0 || !wavelength.is_finite() {
        return QaGateResult {
            name: "gate_max_electrical_size".to_string(),
            status: QaStatus::Skipped,
            message: "non-finite wavelength".to_string(),
            value: 0.0,
        };
    }
    let ratio = extent / wavelength;
    let (status, message) = if ratio > ELECTRICAL_SIZE_WARN_LAMBDA {
        (
            QaStatus::Warn,
            format!(
                "object spans {:.1} wavelengths at {:.3e} Hz (> {:.0}); solver may struggle",
                ratio, f, ELECTRICAL_SIZE_WARN_LAMBDA
            ),
        )
    } else {
        (
            QaStatus::Pass,
            format!(
                "object spans {:.2} wavelengths at {:.3e} Hz (max extent {:.3} m, lambda {:.3e} m)",
                ratio, f, extent, wavelength
            ),
        )
    };
    QaGateResult {
        name: "gate_max_electrical_size".to_string(),
        status,
        message,
        value: ratio,
    }
}
