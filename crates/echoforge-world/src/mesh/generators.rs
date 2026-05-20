use super::{MeshTriangle, ParametricMesh};

/// Triangulated PEC sphere of radius `radius_m`, sampled on `n_lat` latitude
/// bands by `n_lon` longitude slices.
///
/// The dimensions slot order is `[radius_m]`.
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

/// Closed circular cylinder centred on the z-axis with axis-aligned end caps.
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

    let bottom_centre = [0.0, 0.0, z0];
    for j in 0..n_circ {
        let j1 = (j + 1) % n_circ;
        let vj = lat(0, j);
        let vj1 = lat(0, j1);
        triangles.push(MeshTriangle::new(bottom_centre, vj1, vj));
    }

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

/// Dihedral corner reflector: two orthogonal square plates of side `side_m`
/// sharing one edge along the z-axis.
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

    let plate_a = |i: usize, j: usize| -> [f64; 3] { [(i as f64) * step, 0.0, (j as f64) * step] };
    for i in 0..(n - 1) {
        for j in 0..(n - 1) {
            let v00 = plate_a(i, j);
            let v10 = plate_a(i + 1, j);
            let v11 = plate_a(i + 1, j + 1);
            let v01 = plate_a(i, j + 1);
            triangles.push(MeshTriangle::new(v00, v11, v10));
            triangles.push(MeshTriangle::new(v00, v01, v11));
        }
    }

    let plate_b = |i: usize, j: usize| -> [f64; 3] { [0.0, (i as f64) * step, (j as f64) * step] };
    for i in 0..(n - 1) {
        for j in 0..(n - 1) {
            let v00 = plate_b(i, j);
            let v10 = plate_b(i + 1, j);
            let v11 = plate_b(i + 1, j + 1);
            let v01 = plate_b(i, j + 1);
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

/// Trihedral corner reflector: three orthogonal right-triangle plates of leg
/// `side_m`, meeting at the origin.
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

    let xy = |i: usize, j: usize| -> [f64; 3] { [(i as f64) * step, (j as f64) * step, 0.0] };
    emit_plate(&mut triangles, &xy, false);

    let yz = |i: usize, j: usize| -> [f64; 3] { [0.0, (i as f64) * step, (j as f64) * step] };
    emit_plate(&mut triangles, &yz, false);

    let zx = |i: usize, j: usize| -> [f64; 3] { [(j as f64) * step, 0.0, (i as f64) * step] };
    emit_plate(&mut triangles, &zx, false);

    ParametricMesh {
        primitive_id: "trihedral".to_string(),
        dimensions: vec![side_m],
        triangles,
    }
}
