use super::generators::*;
use super::*;
use tempfile::tempdir;

const TOL_REL: f64 = 0.01;

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
    let s = sphere_mesh(2.0, 32, 64);
    let (lo, hi) = s.bounding_box();
    for axis in 0..3 {
        assert!(close(hi[axis] - lo[axis], 4.0, TOL_REL));
    }

    let p = plate_mesh(3.0, 4.0, 8, 8);
    let (lo, hi) = p.bounding_box();
    assert!(close(hi[0] - lo[0], 3.0, TOL_REL));
    assert!(close(hi[1] - lo[1], 4.0, TOL_REL));
    assert!((hi[2] - lo[2]).abs() < 1e-12);

    let c = cylinder_mesh(0.5, 2.0, 64, 8);
    let (lo, hi) = c.bounding_box();
    assert!(close(hi[0] - lo[0], 1.0, TOL_REL));
    assert!(close(hi[1] - lo[1], 1.0, TOL_REL));
    assert!(close(hi[2] - lo[2], 2.0, TOL_REL));

    let d = dihedral_mesh(1.0, 6);
    let (lo, hi) = d.bounding_box();
    for axis in 0..3 {
        assert!(close(hi[axis] - lo[axis], 1.0, TOL_REL));
        assert!(close(lo[axis], 0.0, TOL_REL) || lo[axis].abs() < 1e-9);
    }

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
    assert_eq!(mesh.triangle_count(), 36);
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
    assert_eq!(stl_bytes.len(), 84 + mesh.triangles.len() * 50);
    let expected_count = u32::from_le_bytes(stl_bytes[80..84].try_into().unwrap());
    assert_eq!(expected_count as usize, mesh.triangles.len());

    let manifest_json = std::fs::read_to_string(&manifest_path).expect("read manifest");
    let manifest: MeshManifestFile = serde_json::from_str(&manifest_json).expect("parse manifest");
    assert_eq!(manifest.triangle_count, mesh.triangles.len());
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
    let r = std::panic::catch_unwind(|| sphere_mesh(0.0, 4, 6));
    assert!(r.is_err(), "sphere_mesh should panic on zero radius");
    let r = std::panic::catch_unwind(|| sphere_mesh(1.0, 1, 6));
    assert!(r.is_err(), "sphere_mesh should panic on n_lat < 2");
    let r = std::panic::catch_unwind(|| sphere_mesh(1.0, 4, 2));
    assert!(r.is_err(), "sphere_mesh should panic on n_lon < 3");
}
