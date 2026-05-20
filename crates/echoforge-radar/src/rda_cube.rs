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

use crate::antenna::PhasedArrayManifold;
use crate::ComplexSample;

#[path = "rda_cube_impl.rs"]
mod rda_cube_impl;

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
    rda_cube_impl::build_rda_cube_impl(channel_iq, manifold, angle_grid, range_doppler_bins, pri_s, carrier_hz)
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
    // Default Doppler-Hz decode (PRI = 1 s, used purely as a pending default
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

// Re-export implementation helpers needed by the test module via `use super::*`.
#[cfg(test)]
pub use rda_cube_impl::{dft_in_place, hann_window};
#[cfg(test)]
pub use num_complex::Complex;

#[cfg(test)]
#[path = "rda_cube_tests.rs"]
mod tests;
