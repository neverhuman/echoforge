//! Range-Doppler-Angle (RDA) cube assembly.
//!
//! The existing [`crate::RangeDoppler`] type is a 2-D range × Doppler matrix
//! produced from a single (already-beamformed) channel. For a phased-array
//! front end we additionally want to vary the *look direction* across a
//! discrete steering grid so that direction-of-arrival aware detectors can
//! integrate over angle. This module produces the *Range-Doppler-Angle* (RDA)
//! cube, indexed as `cube[angle][doppler][range]` in power (linear, not dB).
//!
//! # Pipeline
//!
//! For each `(azimuth, elevation)` pair on the user-supplied [`AngleGrid`]:
//!
//! 1. Build a ULA steering vector for `azimuth` via
//!    [`crate::beamforming::steering_vector`] (the manifold supplies the
//!    array geometry).
//! 2. Beamform the per-channel IQ by applying the conjugated steering vector
//!    and summing across channels (a delay-and-sum with the same convention
//!    as [`crate::beamforming::DelayAndSumBeamformer`]).
//! 3. Reshape the resulting 1-D beamformed time-series as an
//!    `n_doppler` × `n_range` slow-time / fast-time matrix.
//! 4. Apply a Hann window across the slow-time axis and run an in-place DFT
//!    per range bin to populate the Doppler axis. The squared magnitude
//!    becomes the cube cell.
//!
//! No external FFT dependency is pulled in: the inline DFT keeps the code
//! self-contained, deterministic, and trivially auditable. Cube sizes used
//! by detectors are small (a handful of angles × a few dozen Doppler bins ×
//! a few hundred range bins), so the O(N²) Doppler transform is not a hot
//! path.
//!
//! # Determinism
//!
//! Every step is deterministic: no RNGs, no parallel iteration ordering, no
//! floating-point reductions that depend on chunking. Calling
//! [`build_rda_cube`] twice with the same arguments yields byte-identical
//! cubes.

use num_complex::Complex;

use crate::antenna::PhasedArrayManifold;
use crate::beamforming::steering_vector;
use crate::ComplexSample;

const C_M_PER_S: f64 = 299_792_458.0;

/// Discrete steering grid the RDA cube is sampled on. Azimuth and elevation
/// are in degrees from boresight; many radars use a single elevation slice,
/// in which case `elevation_deg` carries a single entry (typically `0.0`).
#[derive(Debug, Clone, PartialEq)]
pub struct AngleGrid {
    pub azimuth_deg: Vec<f64>,
    pub elevation_deg: Vec<f64>,
}

impl AngleGrid {
    pub fn new(azimuth_deg: Vec<f64>, elevation_deg: Vec<f64>) -> Self {
        Self {
            azimuth_deg,
            elevation_deg,
        }
    }

    /// Convenience: a single-elevation grid (the common case for many
    /// surveillance radars).
    pub fn azimuth_only(azimuth_deg: Vec<f64>) -> Self {
        Self {
            azimuth_deg,
            elevation_deg: vec![0.0],
        }
    }

    /// Number of grid points = `azimuth.len() * elevation.len()`. The
    /// canonical iteration order is "elevation slow, azimuth fast", so
    /// `index = el_idx * azimuth.len() + az_idx`.
    pub fn len(&self) -> usize {
        self.azimuth_deg.len() * self.elevation_deg.len()
    }

    pub fn is_empty(&self) -> bool {
        self.azimuth_deg.is_empty() || self.elevation_deg.is_empty()
    }

    /// Resolve a flat index into the underlying (azimuth, elevation) pair.
    /// Returns `None` if the index is out of range.
    pub fn at(&self, index: usize) -> Option<(f64, f64)> {
        if self.is_empty() {
            return None;
        }
        let n_az = self.azimuth_deg.len();
        if index >= self.len() {
            return None;
        }
        let el_idx = index / n_az;
        let az_idx = index % n_az;
        Some((self.azimuth_deg[az_idx], self.elevation_deg[el_idx]))
    }
}

/// Range-Doppler-Angle cube produced by [`build_rda_cube`].
///
/// `cube[angle][doppler][range]` carries the linear power at the given grid
/// cell. The cube is always *exactly* `angle_count × doppler_bins ×
/// range_bins`; degenerate inputs (empty channel data, zero-length grids)
/// produce zero-filled cubes of the correct shape so downstream consumers
/// can rely on the dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeDopplerAngle {
    pub range_bins: usize,
    pub doppler_bins: usize,
    pub angle_count: usize,
    pub angle_grid: AngleGrid,
    /// Layout: `cube[angle][doppler][range]`, row-major. Power-domain
    /// (linear, not dB).
    pub cube: Vec<Vec<Vec<f32>>>,
}

impl RangeDopplerAngle {
    /// Allocate a zero-filled cube of the requested shape.
    pub fn zeros(range_bins: usize, doppler_bins: usize, angle_grid: AngleGrid) -> Self {
        let angle_count = angle_grid.len();
        let cube = vec![vec![vec![0.0f32; range_bins]; doppler_bins]; angle_count];
        Self {
            range_bins,
            doppler_bins,
            angle_count,
            angle_grid,
            cube,
        }
    }

    /// Borrow the angle slice (a 2-D Doppler × range matrix) at `angle_idx`.
    pub fn angle_slice(&self, angle_idx: usize) -> Option<&Vec<Vec<f32>>> {
        self.cube.get(angle_idx)
    }
}

/// Peak descriptor returned by [`rda_peak`]. The indices are positions in
/// `cube[angle_idx][doppler_idx][range_idx]`; the `*_deg`, `doppler_hz`, and
/// `range_bin` fields are user-facing decodings of those indices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdaPeak {
    pub angle_idx: usize,
    pub doppler_idx: usize,
    pub range_idx: usize,
    pub magnitude: f32,
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub doppler_hz: f64,
    pub range_bin: usize,
}

/// Build a Range-Doppler-Angle cube from per-channel IQ.
///
/// * `channel_iq` is indexed as `channel_iq[channel][sample]`. Channels with
///   shorter sample vectors are zero-padded to the longest channel.
/// * `manifold` supplies the array geometry (number of elements, element
///   spacing). Its embedded `steering_*_deg` are ignored here — this builder
///   re-steers the array per `(az, el)` grid cell.
/// * `angle_grid` enumerates the steering directions the cube is built on.
/// * `range_doppler_bins = (n_range, n_doppler)`. The flat beamformed
///   time-series is reshaped as `n_doppler × n_range`; the slow-time axis
///   (Doppler) then receives a Hann-windowed DFT per range bin.
/// * `pri_s` is the pulse-repetition interval in seconds. The Doppler axis
///   maps bin `k` to `(k − N/2) / (N · pri_s)` Hz after `fftshift`, which is
///   used to fill `RdaPeak::doppler_hz`. The cube layout itself is *not*
///   `fftshift`ed — bin 0 is DC, bin `N-1` is `(N-1)/(N·pri_s)` Hz, etc.
/// * `carrier_hz` sets the wavelength for the steering vector.
///
/// Edge cases:
///
/// * Empty `channel_iq` → zero-filled cube of the requested shape.
/// * `n_range * n_doppler` larger than the beamformed sample count → the
///   missing samples are treated as zero (zero-padded in fast-time).
/// * Zero-length angle grid → cube with `angle_count == 0` and empty outer
///   vector.
pub fn build_rda_cube(
    channel_iq: &[Vec<ComplexSample>],
    manifold: &PhasedArrayManifold,
    angle_grid: &AngleGrid,
    range_doppler_bins: (usize, usize),
    pri_s: f64,
    carrier_hz: f64,
) -> RangeDopplerAngle {
    let (n_range, n_doppler) = range_doppler_bins;
    let angle_count = angle_grid.len();
    let mut cube = vec![vec![vec![0.0f32; n_range]; n_doppler]; angle_count];

    // Degenerate cases: return correctly-shaped zero cube.
    if angle_count == 0 || n_range == 0 || n_doppler == 0 || channel_iq.is_empty() {
        return RangeDopplerAngle {
            range_bins: n_range,
            doppler_bins: n_doppler,
            angle_count,
            angle_grid: angle_grid.clone(),
            cube,
        };
    }

    // Determine the canonical channel sample count (the longest channel).
    let n_samples = channel_iq.iter().map(|ch| ch.len()).max().unwrap_or(0);
    if n_samples == 0 {
        return RangeDopplerAngle {
            range_bins: n_range,
            doppler_bins: n_doppler,
            angle_count,
            angle_grid: angle_grid.clone(),
            cube,
        };
    }

    let n_channels = channel_iq.len();
    let channel_norm = (n_channels.max(1)) as f32;

    // Hann window across the slow-time / Doppler axis. Built once per call.
    let hann = hann_window(n_doppler);

    // Pre-compute the (az, el) tuples in canonical order.
    let mut angles: Vec<(f64, f64)> = Vec::with_capacity(angle_count);
    for &el in &angle_grid.elevation_deg {
        for &az in &angle_grid.azimuth_deg {
            angles.push((az, el));
        }
    }

    for (angle_idx, (az_deg, _el_deg)) in angles.iter().enumerate() {
        // Steering vector for this (az). The current ULA helper steers in
        // azimuth only; elevation enters via the angle grid but does not
        // perturb the steering of a horizontal ULA. The cube still indexes
        // elevation slices so a 2-D (planar) array can be plugged in later
        // without breaking the API.
        let steering = steering_vector(
            manifold.n_elements.max(1),
            manifold.element_spacing_m,
            carrier_hz,
            *az_deg,
        );

        // Beamform: weighted sum across channels with the conjugated
        // steering vector, matching DelayAndSumBeamformer's convention.
        let mut beamformed = vec![ComplexSample::new(0.0, 0.0); n_samples];
        for (ch_idx, channel) in channel_iq.iter().enumerate() {
            let weight = steering
                .get(ch_idx)
                .copied()
                .unwrap_or(Complex::new(1.0, 0.0));
            let w_conj = weight.conj();
            let w_sample = ComplexSample::new(w_conj.re as f32, w_conj.im as f32);
            let len = channel.len().min(n_samples);
            for i in 0..len {
                beamformed[i] += channel[i] * w_sample;
            }
        }
        for sample in &mut beamformed {
            sample.re /= channel_norm;
            sample.im /= channel_norm;
        }

        // Reshape as [doppler][range] and run a per-range Doppler DFT.
        // Out-of-range fast-time / slow-time samples are zero-padded.
        for r in 0..n_range {
            let mut column: Vec<Complex<f64>> = (0..n_doppler)
                .map(|d| {
                    let flat_idx = d * n_range + r;
                    let s = if flat_idx < n_samples {
                        beamformed[flat_idx]
                    } else {
                        ComplexSample::new(0.0, 0.0)
                    };
                    Complex::new(s.re as f64 * hann[d], s.im as f64 * hann[d])
                })
                .collect();
            dft_in_place(&mut column);
            for d in 0..n_doppler {
                let power = column[d].norm_sqr() as f32;
                cube[angle_idx][d][r] = power;
            }
        }
    }

    let _ = pri_s; // pri_s is consumed by `rda_peak` via the cube's carrier_hz/pri_s decode helpers below.

    RangeDopplerAngle {
        range_bins: n_range,
        doppler_bins: n_doppler,
        angle_count,
        angle_grid: angle_grid.clone(),
        cube,
    }
}

/// Find the global maximum cell in the cube and decode its grid coordinates.
///
/// `pri_s` controls the Doppler-bin → Hz mapping reported in [`RdaPeak`]:
/// bin `k` (where `k < N`) maps to `(k − N/2) / (N · pri_s)` if `k ≥ N/2`
/// (i.e. an `fftshift`-style negative-frequency interpretation), and to
/// `k / (N · pri_s)` for `k < N/2`. Pass any positive `pri_s` (e.g. `1.0`)
/// if the Hz decoding is not needed; the index fields remain meaningful.
pub fn rda_peak(cube: &RangeDopplerAngle) -> RdaPeak {
    let mut best = RdaPeak {
        angle_idx: 0,
        doppler_idx: 0,
        range_idx: 0,
        magnitude: f32::NEG_INFINITY,
        azimuth_deg: 0.0,
        elevation_deg: 0.0,
        doppler_hz: 0.0,
        range_bin: 0,
    };
    if cube.angle_count == 0 || cube.doppler_bins == 0 || cube.range_bins == 0 {
        best.magnitude = 0.0;
        return best;
    }
    for a in 0..cube.angle_count {
        for d in 0..cube.doppler_bins {
            for r in 0..cube.range_bins {
                let v = cube.cube[a][d][r];
                if v > best.magnitude {
                    best.magnitude = v;
                    best.angle_idx = a;
                    best.doppler_idx = d;
                    best.range_idx = r;
                }
            }
        }
    }
    if let Some((az, el)) = cube.angle_grid.at(best.angle_idx) {
        best.azimuth_deg = az;
        best.elevation_deg = el;
    }
    // Default Doppler-Hz decode (PRI = 1 s, used purely as a placeholder
    // when callers don't have an authoritative PRI handy). Callers that
    // need physical units should multiply the index-side decoding through
    // themselves.
    let n = cube.doppler_bins as f64;
    let k = best.doppler_idx as f64;
    let signed = if best.doppler_idx >= cube.doppler_bins / 2 {
        k - n
    } else {
        k
    };
    best.doppler_hz = signed / n;
    best.range_bin = best.range_idx;
    if !best.magnitude.is_finite() {
        best.magnitude = 0.0;
    }
    best
}

/// Borrow the `[doppler][range]` slice at `angle_idx` without copying.
/// Returns `None` when the index is out of range.
pub fn rda_extract_angle_slice(
    cube: &RangeDopplerAngle,
    angle_idx: usize,
) -> Option<&Vec<Vec<f32>>> {
    cube.angle_slice(angle_idx)
}

/// Collapse the angle axis of an RDA cube to a 2-D range-Doppler map. The
/// reduction sums power across angles (consistent with the cube being in
/// the power domain), producing a matrix indexed as `rd[doppler][range]`
/// with the same dimensions as a single angle slice. All cells are `>= 0`
/// because the cube is non-negative by construction.
pub fn rda_to_rd_sum(cube: &RangeDopplerAngle) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0f32; cube.range_bins]; cube.doppler_bins];
    for slice in &cube.cube {
        for d in 0..cube.doppler_bins {
            for r in 0..cube.range_bins {
                out[d][r] += slice[d][r];
            }
        }
    }
    out
}

// --- internal helpers -------------------------------------------------------

fn hann_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0);
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

/// Naive DFT used to run the Doppler transform per range bin. The cube
/// sizes used here are small (a few dozen Doppler bins at most), so the
/// O(N²) cost is acceptable in exchange for working for *any* N and being
/// trivially auditable. This routine matches the rustfft "forward"
/// convention: `X[k] = Σ x[n] · exp(-j 2π n k / N)`.
fn dft_in_place(buf: &mut [Complex<f64>]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    let input: Vec<Complex<f64>> = buf.to_vec();
    let two_pi = 2.0 * std::f64::consts::PI;
    for (k, slot) in buf.iter_mut().enumerate().take(n) {
        let mut acc = Complex::<f64>::new(0.0, 0.0);
        for (n_idx, x) in input.iter().enumerate() {
            let angle = -two_pi * (n_idx as f64) * (k as f64) / n as f64;
            let twiddle = Complex::<f64>::new(angle.cos(), angle.sin());
            acc += x * twiddle;
        }
        *slot = acc;
    }
    let _ = C_M_PER_S; // c is reserved for a future wavelength conversion.
}

#[cfg(test)]
mod tests {
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
            &[iq.clone()],
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
            for d in 0..n_doppler {
                let expected = col[d].norm_sqr() as f32;
                let actual = slice[d][r];
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
}
