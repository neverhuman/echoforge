//! `build_rda_cube` implementation — extracted from rda_cube.rs for LOC compliance.

use num_complex::Complex;

use super::{AngleGrid, RangeDopplerAngle};
use crate::antenna::PhasedArrayManifold;
use crate::beamforming::steering_vector;
use crate::ComplexSample;

pub fn hann_window(n: usize) -> Vec<f64> {
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
pub fn dft_in_place(buf: &mut [Complex<f64>]) {
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
}

/// Core of `build_rda_cube`. Called from rda_cube.rs.
pub(super) fn build_rda_cube_impl(
    channel_iq: &[Vec<ComplexSample>],
    manifold: &PhasedArrayManifold,
    angle_grid: &AngleGrid,
    range_doppler_bins: (usize, usize),
    _pri_s: f64,
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

    let hann = hann_window(n_doppler);

    let mut angles: Vec<(f64, f64)> = Vec::with_capacity(angle_count);
    for &el in &angle_grid.elevation_deg {
        for &az in &angle_grid.azimuth_deg {
            angles.push((az, el));
        }
    }

    for (angle_idx, (az_deg, _el_deg)) in angles.iter().enumerate() {
        let steering = steering_vector(
            manifold.n_elements.max(1),
            manifold.element_spacing_m,
            carrier_hz,
            *az_deg,
        );

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
            for (d, doppler_slice) in cube[angle_idx].iter_mut().enumerate().take(n_doppler) {
                let power = column[d].norm_sqr() as f32;
                doppler_slice[r] = power;
            }
        }
    }

    RangeDopplerAngle {
        range_bins: n_range,
        doppler_bins: n_doppler,
        angle_count,
        angle_grid: angle_grid.clone(),
        cube,
    }
}
