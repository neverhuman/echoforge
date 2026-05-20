//! Mesh quality-assurance gates for [`ParametricMesh`] surfaces.
//!
//! Each individual check is a "gate" that returns a [`QaGateResult`] with one
//! of three statuses: `pass`, `warn`, or `fail`. The aggregated
//! [`run_qa`] entry point bundles every gate into a single [`QaReport`] whose
//! `overall_status` is the worst gate outcome.

use serde::{Deserialize, Serialize};

use crate::mesh::{MeshTriangle, ParametricMesh};

mod math;
mod gates_geometry;
mod gates_topology;

pub use gates_geometry::{
    ASPECT_RATIO_FAIL_MEAN, ASPECT_RATIO_WARN, DEGENERATE_AREA_THRESHOLD_M2,
    WINDING_FAIL_FRACTION,
    gate_bounded_aspect_ratio, gate_consistent_winding, gate_finite_vertices,
    gate_no_degenerate_triangles,
};
pub use gates_topology::{
    DIHEDRAL_MAX_DEG, DIHEDRAL_MIN_DEG, ELECTRICAL_SIZE_WARN_LAMBDA, SPEED_OF_LIGHT_M_PER_S,
    VOLUME_HI_FRAC, VOLUME_LO_FRAC,
    gate_bounded_dihedral_angle, gate_bounded_volume_vs_box, gate_max_electrical_size,
    gate_watertight,
};

#[cfg(test)]
mod tests;

/// Outcome label for a single QA gate or for an aggregate [`QaReport`].
///
/// Ordered from best to worst: `Pass < Warn < Fail`. The aggregate status of a
/// report is the worst label among its gates (with `Skipped` treated as
/// neutral and folded into `Pass`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QaStatus {
    /// Gate passed unconditionally.
    Pass,
    /// Gate found a non-blocking issue worth surfacing in reports.
    Warn,
    /// Gate found a blocking issue: the mesh should not be sent to a solver.
    Fail,
    /// Gate did not apply to this primitive (e.g. volume check on an open
    /// surface). Treated as neutral when computing the report's overall status.
    Skipped,
}

impl QaStatus {
    fn rank(self) -> u8 {
        match self {
            QaStatus::Pass => 0,
            QaStatus::Skipped => 0,
            QaStatus::Warn => 1,
            QaStatus::Fail => 2,
        }
    }

    /// Combine two statuses, returning the worse of the two.
    pub fn worse_of(self, other: QaStatus) -> QaStatus {
        if other.rank() > self.rank() {
            other
        } else {
            self
        }
    }
}

/// Per-gate result emitted by every QA check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QaGateResult {
    /// Stable identifier of the gate (matches the `gate_*` function name).
    pub name: String,
    /// Outcome label (`pass | warn | fail | skipped`).
    pub status: QaStatus,
    /// Short human-readable rationale (always non-empty).
    pub message: String,
    /// Quantitative scalar associated with the gate (e.g. a count or a ratio).
    pub value: f64,
}

/// Aggregated report bundling every gate that was run for a mesh.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QaReport {
    /// `primitive_id` of the mesh that was inspected.
    pub mesh_primitive_id: String,
    /// Worst status across the bundled gates (see [`QaStatus::worse_of`]).
    pub overall_status: QaStatus,
    /// One [`QaGateResult`] per executed gate, in execution order.
    pub gates: Vec<QaGateResult>,
    /// Number of triangles in the mesh.
    pub triangle_count: usize,
    /// Lower corner of the mesh bounding box.
    pub bbox_min: [f64; 3],
    /// Upper corner of the mesh bounding box.
    pub bbox_max: [f64; 3],
}

// ---------------------------------------------------------------------------
// Aggregation entry point
// ---------------------------------------------------------------------------

/// Runs the full QA gate suite against `mesh` and returns a single report.
///
/// `freq_hz`, if `Some`, is forwarded to [`gate_max_electrical_size`]; when
/// `None`, that gate is skipped.
pub fn run_qa(mesh: &ParametricMesh, freq_hz: Option<f64>) -> QaReport {
    let (bbox_min, bbox_max) = mesh.bounding_box();
    let primitive_id = mesh.primitive_id.clone();

    let gates = vec![
        gate_finite_vertices(mesh),
        gate_no_degenerate_triangles(mesh),
        gate_consistent_winding(mesh),
        gate_bounded_aspect_ratio(mesh),
        gate_bounded_dihedral_angle(mesh),
        gate_watertight(mesh),
        gate_bounded_volume_vs_box(mesh),
        gate_max_electrical_size(mesh, freq_hz),
    ];

    let mut overall = QaStatus::Pass;
    for gate in &gates {
        overall = overall.worse_of(gate.status);
    }

    QaReport {
        mesh_primitive_id: primitive_id,
        overall_status: overall,
        gates,
        triangle_count: mesh.triangle_count(),
        bbox_min,
        bbox_max,
    }
}

// ---------------------------------------------------------------------------
// Mutation helpers (test-only in usage; exposed for integration tests in
// sibling crates that need to inject failure modes without re-implementing
// the mutation logic).
// ---------------------------------------------------------------------------

/// Replace the `triangle_index`-th triangle's `v0` with a NaN-loaded vertex.
/// Returns `true` on success.
pub fn inject_nan_vertex(mesh: &mut ParametricMesh, triangle_index: usize) -> bool {
    let Some(t) = mesh.triangles.get_mut(triangle_index) else {
        return false;
    };
    t.v0 = [f64::NAN, 0.0, 0.0];
    true
}

/// Replace the `triangle_index`-th triangle with three colinear points,
/// guaranteeing a zero-area (degenerate) triangle. Returns `true` on success.
pub fn inject_degenerate_triangle(mesh: &mut ParametricMesh, triangle_index: usize) -> bool {
    let Some(t) = mesh.triangles.get_mut(triangle_index) else {
        return false;
    };
    *t = MeshTriangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
    true
}
