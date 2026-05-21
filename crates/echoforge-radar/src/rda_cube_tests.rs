use super::*;

fn isotropic_manifold() -> PhasedArrayManifold {
    // Single element acts as an isotropic antenna under the ULA model:
    // the steering vector collapses to [1+0j] and beamforming reduces
    // to the channel sample itself.
    PhasedArrayManifold::new(1, 0.015, 10_000_000_000.0, 0.0, 0.0)
}

fn ula_manifold(n: usize) -> PhasedArrayManifold {
    PhasedArrayManifold::new(n, 0.015, 10_000_000_000.0, 0.0, 0.0)
}

fn synth_iq(n_samples: usize, frequency_bin: f64) -> Vec<ComplexSample> {
    // A single Doppler-band tone at the requested fractional bin, with
    // a fast-time impulse at sample 0 so the Doppler FFT energy
    // concentrates predictably.
    let two_pi = 2.0 * std::f64::consts::PI as f32;
    (0..n_samples)
        .map(|i| {
            let phase = two_pi * frequency_bin as f32 * (i as f32);
            ComplexSample::new(phase.cos(), phase.sin())
        })
        .collect()
}

fn channels_from_plane_wave(
    n_channels: usize,
    n_samples: usize,
    spacing_m: f64,
    freq_hz: f64,
    az_deg: f64,
) -> Vec<Vec<ComplexSample>> {
    let lambda = 299_792_458.0_f64 / freq_hz;
    let k = 2.0 * std::f64::consts::PI / lambda;
    let sin_theta = az_deg.to_radians().sin();
    (0..n_channels)
        .map(|ch| {
            let pos = (ch as f64) - (n_channels as f64 - 1.0) * 0.5;
            let phase = (k * spacing_m * pos * sin_theta) as f32;
            let phasor = ComplexSample::new(phase.cos(), phase.sin());
            vec![phasor; n_samples]
        })
        .collect()
}

#[test]
fn empty_channel_iq_yields_zero_filled_cube() {
    let grid = AngleGrid::azimuth_only(vec![-30.0, 0.0, 30.0]);
    let cube = build_rda_cube(
        &[],
        &isotropic_manifold(),
        &grid,
        (8, 4),
        1e-3,
        10_000_000_000.0,
    );
    assert_eq!(cube.range_bins, 8);
    assert_eq!(cube.doppler_bins, 4);
    assert_eq!(cube.angle_count, 3);
    assert_eq!(cube.cube.len(), 3);
    for slice in &cube.cube {
        assert_eq!(slice.len(), 4);
        for row in slice {
            assert_eq!(row.len(), 8);
            for &v in row {
                assert_eq!(v, 0.0);
            }
        }
    }
}

#[test]
fn single_channel_isotropic_matches_one_angle_rd_slice() {
    // With a single channel and a one-angle grid, the cube reduces to
    // a single slice that should agree with an inline Hann-windowed
    // Doppler DFT computed the same way.
    let n_range = 4;
    let n_doppler = 8;
    let total = n_range * n_doppler;
    let iq = synth_iq(total, 0.0); // DC tone
    let cube = build_rda_cube(
        std::slice::from_ref(&iq),
        &isotropic_manifold(),
        &AngleGrid::azimuth_only(vec![0.0]),
        (n_range, n_doppler),
        1e-3,
        10_000_000_000.0,
    );
    assert_eq!(cube.angle_count, 1);
    assert_eq!(cube.cube.len(), 1);
    let slice = &cube.cube[0];

    // Reference: build the same reshape-and-Doppler-FFT path inline.
    let hann = hann_window(n_doppler);
    for r in 0..n_range {
        let mut col: Vec<Complex<f64>> = (0..n_doppler)
            .map(|d| {
                let i = d * n_range + r;
                Complex::new(iq[i].re as f64 * hann[d], iq[i].im as f64 * hann[d])
            })
            .collect();
        dft_in_place(&mut col);
        for (d, doppler_slice) in slice.iter().enumerate().take(n_doppler) {
            let expected = col[d].norm_sqr() as f32;
            let actual = doppler_slice[r];
            assert!(
                (actual - expected).abs() < 1e-3,
                "(d={d},r={r}) cube={actual} ref={expected}",
            );
        }
    }
}

#[test]
fn steered_ula_peaks_at_target_azimuth() {
    // Eight-element ULA, target at +20° azimuth (plane wave across the
    // aperture). Scan over a 41-point azimuth grid; the angle bin
    // closest to +20° should win.
    let n_channels = 8;
    let n_range = 4;
    let n_doppler = 8;
    let total = n_range * n_doppler;
    let spacing = 0.015;
    let freq = 10_000_000_000.0;
    let target = 20.0;

    let channels = channels_from_plane_wave(n_channels, total, spacing, freq, target);
    let manifold = ula_manifold(n_channels);
    let grid_az: Vec<f64> = (-40..=40).step_by(2).map(|i| i as f64).collect();
    let target_bin_idx = grid_az
        .iter()
        .enumerate()
        .min_by(|a, b| {
            (a.1 - target)
                .abs()
                .partial_cmp(&(b.1 - target).abs())
                .unwrap()
        })
        .unwrap()
        .0;

    let grid = AngleGrid::azimuth_only(grid_az);
    let cube = build_rda_cube(
        &channels,
        &manifold,
        &grid,
        (n_range, n_doppler),
        1e-3,
        freq,
    );
    let peak = rda_peak(&cube);
    // Allow ±1 bin tolerance because the steering grid quantisation
    // can place adjacent bins almost equally close.
    let delta = (peak.angle_idx as isize - target_bin_idx as isize).abs();
    assert!(
        delta <= 1,
        "peak angle_idx={} (az={}) target_idx={} target_az={}",
        peak.angle_idx,
        peak.azimuth_deg,
        target_bin_idx,
        target,
    );
}

#[test]
fn extract_angle_slice_out_of_range_returns_none() {
    let grid = AngleGrid::azimuth_only(vec![-10.0, 0.0, 10.0]);
    let cube = RangeDopplerAngle::zeros(4, 2, grid);
    assert!(rda_extract_angle_slice(&cube, 0).is_some());
    assert!(rda_extract_angle_slice(&cube, 2).is_some());
    assert!(rda_extract_angle_slice(&cube, 3).is_none());
    assert!(rda_extract_angle_slice(&cube, 99).is_none());
}

#[test]
fn extract_angle_slice_shape_matches_cube() {
    let grid = AngleGrid::azimuth_only(vec![0.0]);
    let cube = RangeDopplerAngle::zeros(5, 3, grid);
    let slice = rda_extract_angle_slice(&cube, 0).expect("angle 0 present");
    assert_eq!(slice.len(), 3);
    for row in slice {
        assert_eq!(row.len(), 5);
    }
}

#[test]
fn rda_to_rd_sum_preserves_dims_and_is_non_negative() {
    let n_channels = 4;
    let n_range = 6;
    let n_doppler = 4;
    let total = n_range * n_doppler;
    let channels = channels_from_plane_wave(n_channels, total, 0.015, 10_000_000_000.0, 10.0);
    let grid = AngleGrid::azimuth_only(vec![-10.0, 0.0, 10.0, 20.0]);
    let cube = build_rda_cube(
        &channels,
        &ula_manifold(n_channels),
        &grid,
        (n_range, n_doppler),
        1e-3,
        10_000_000_000.0,
    );
    let rd = rda_to_rd_sum(&cube);
    assert_eq!(rd.len(), n_doppler);
    for row in &rd {
        assert_eq!(row.len(), n_range);
        for &v in row {
            assert!(v >= 0.0, "RD sum produced negative cell {v}");
        }
    }
}

#[test]
fn rda_to_rd_sum_equals_sum_of_slices_cellwise() {
    // Sanity check on the reduction: the sum across angles should
    // equal the cell-wise accumulation of individual slices.
    let n_channels = 2;
    let n_range = 3;
    let n_doppler = 4;
    let total = n_range * n_doppler;
    let channels = channels_from_plane_wave(n_channels, total, 0.015, 10_000_000_000.0, 5.0);
    let grid = AngleGrid::azimuth_only(vec![-5.0, 0.0, 5.0]);
    let cube = build_rda_cube(
        &channels,
        &ula_manifold(n_channels),
        &grid,
        (n_range, n_doppler),
        1e-3,
        10_000_000_000.0,
    );
    let rd = rda_to_rd_sum(&cube);
    for d in 0..n_doppler {
        for r in 0..n_range {
            let expected: f32 = cube.cube.iter().map(|slice| slice[d][r]).sum();
            let actual = rd[d][r];
            assert!(
                (actual - expected).abs() < 1e-5,
                "(d={d},r={r}) rd={actual} expected={expected}",
            );
        }
    }
}

#[test]
fn build_is_deterministic_byte_identical() {
    // Two independent invocations on the same input must yield exactly
    // the same cube, with no floating-point reordering or RNG noise.
    let n_channels = 4;
    let n_range = 5;
    let n_doppler = 4;
    let total = n_range * n_doppler;
    let channels = channels_from_plane_wave(n_channels, total, 0.015, 10_000_000_000.0, 12.0);
    let grid = AngleGrid::azimuth_only(vec![-10.0, 0.0, 10.0]);
    let manifold = ula_manifold(n_channels);
    let cube_a = build_rda_cube(
        &channels,
        &manifold,
        &grid,
        (n_range, n_doppler),
        1e-3,
        10_000_000_000.0,
    );
    let cube_b = build_rda_cube(
        &channels,
        &manifold,
        &grid,
        (n_range, n_doppler),
        1e-3,
        10_000_000_000.0,
    );
    assert_eq!(cube_a, cube_b, "build_rda_cube is not deterministic");
}

#[test]
fn rda_peak_handles_degenerate_cube_gracefully() {
    let grid = AngleGrid::azimuth_only(Vec::new());
    let cube = build_rda_cube(
        &[],
        &isotropic_manifold(),
        &grid,
        (4, 2),
        1e-3,
        10_000_000_000.0,
    );
    assert_eq!(cube.angle_count, 0);
    let peak = rda_peak(&cube);
    assert_eq!(peak.magnitude, 0.0);
    assert_eq!(peak.angle_idx, 0);
}

#[test]
fn angle_grid_at_indexes_correctly() {
    let grid = AngleGrid::new(vec![-10.0, 0.0, 10.0], vec![-5.0, 5.0]);
    assert_eq!(grid.len(), 6);
    assert_eq!(grid.at(0), Some((-10.0, -5.0)));
    assert_eq!(grid.at(2), Some((10.0, -5.0)));
    assert_eq!(grid.at(3), Some((-10.0, 5.0)));
    assert_eq!(grid.at(5), Some((10.0, 5.0)));
    assert_eq!(grid.at(6), None);
}

#[test]
fn zero_range_or_doppler_yields_empty_inner_dims() {
    let grid = AngleGrid::azimuth_only(vec![0.0]);
    let cube = build_rda_cube(
        &[vec![ComplexSample::new(1.0, 0.0); 4]],
        &isotropic_manifold(),
        &grid,
        (0, 4),
        1e-3,
        10_000_000_000.0,
    );
    assert_eq!(cube.range_bins, 0);
    assert_eq!(cube.doppler_bins, 4);
    for slice in &cube.cube {
        for row in slice {
            assert!(row.is_empty());
        }
    }
}

#[test]
fn cube_dimensions_match_request_for_uneven_grid() {
    let grid = AngleGrid::new(vec![-15.0, 0.0, 15.0, 30.0], vec![-5.0, 0.0, 5.0]);
    let cube = build_rda_cube(
        &[vec![ComplexSample::new(1.0, 0.0); 16]],
        &isotropic_manifold(),
        &grid,
        (4, 4),
        1e-3,
        10_000_000_000.0,
    );
    assert_eq!(cube.angle_count, 12);
    assert_eq!(cube.cube.len(), 12);
    for slice in &cube.cube {
        assert_eq!(slice.len(), 4);
        for row in slice {
            assert_eq!(row.len(), 4);
        }
    }
}
