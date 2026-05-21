use std::f32::consts::PI;

use crate::ComplexSample;

/// Complex slow-time DFT over per-pulse compressed-IQ range profiles,
/// producing a `(range, Doppler)` complex grid. Phase is preserved
/// end-to-end so downstream MTI / MTD / micro-Doppler processing can use
/// coherent arithmetic instead of working from a magnitude image.
///
/// Inputs:
///   - `compressed_pulses` — per-pulse complex range profiles (the
///     output of [`crate::pulse_compression::pulse_compress_windowed`]).
///     Each inner vector must have the same length; shorter rows are
///     zero-padded along the range axis.
///   - `n_doppler` — number of slow-time samples (`n_pulses`). The DFT
///     uses N = `compressed_pulses.len()` slow-time samples and emits
///     `n_doppler` Doppler bins (typically equal to N). Passing
///     `n_doppler == 0` returns an empty grid.
///
/// Output: `Vec<Vec<ComplexSample>>` indexed `[range_bin][doppler_bin]`
/// with length `range_len` along axis 0 and `n_doppler` along axis 1.
///
/// Convention (Skolnik, *Introduction to Radar Systems*, 3rd ed., §3.5):
///   `X[k] = Σₙ x[n] · exp(-j · 2π · k · n / N)`
///
/// Implementation: naive O(N²) DFT per range bin. The grid is small
/// (pulses ~ 32 for the takeoff fixture) so an FFT dependency is not
/// justified here; callers that need an FFT can wrap this signature.
pub fn slow_time_complex_dft(
    compressed_pulses: &[Vec<ComplexSample>],
    n_doppler: usize,
) -> Vec<Vec<ComplexSample>> {
    let n_pulses = compressed_pulses.len();
    if n_pulses == 0 || n_doppler == 0 {
        return Vec::new();
    }

    let range_len = compressed_pulses
        .iter()
        .map(|profile| profile.len())
        .max()
        .unwrap_or(0);
    if range_len == 0 {
        return Vec::new();
    }

    let mut output = vec![vec![ComplexSample::new(0.0, 0.0); n_doppler]; range_len];

    let n_pulses_f = n_pulses as f32;
    for (range, output_row) in output.iter_mut().enumerate().take(range_len) {
        for (doppler, output_cell) in output_row.iter_mut().enumerate().take(n_doppler) {
            let mut acc = ComplexSample::new(0.0, 0.0);
            for (pulse, profile) in compressed_pulses.iter().enumerate() {
                let sample = profile
                    .get(range)
                    .copied()
                    .unwrap_or(ComplexSample::new(0.0, 0.0));
                let angle = -2.0 * PI * (doppler as f32) * (pulse as f32) / n_pulses_f;
                let phasor = ComplexSample::new(angle.cos(), angle.sin());
                acc += sample * phasor;
            }
            *output_cell = acc;
        }
    }
    output
}

/// Magnitude image of the slow-time DFT over per-pulse range-profile
/// magnitudes. This is the prior `range_doppler_proxy` signature: the
/// caller has already stripped phase via [`magnitude`] before the
/// slow-time transform, so the result is a magnitude-of-magnitudes
/// proxy and cannot be used for coherent Doppler / MTI / micro-Doppler
/// reasoning. New code must consume [`slow_time_complex_dft`] (and call
/// `.norm()` on each cell if it only wants the magnitude grid).
///
/// Output is laid out `[doppler_bin][range_bin]` and normalised by
/// `1/N_pulses`, matching the pre-Lane-B implementation so byte-stable
/// fixtures continue to reproduce.
///
/// Implementation is now a thin wrapper that lifts each magnitude into
/// a `ComplexSample` with zero imaginary part, dispatches to
/// [`slow_time_complex_dft`], then collapses to per-bin magnitude with
/// the prior 1/N normalisation. The lift is mathematically lossless
/// (the DFT of a real sequence reproduces the magnitudes of the real
/// DFT), so the wrapper is byte-stable with the inlined implementation
/// modulo floating-point summation order — which we hold constant by
/// keeping the same inner loop.
pub fn slow_time_dft_magnitude(profiles: &[Vec<f32>], range_len: usize) -> Vec<Vec<f32>> {
    let pulses = profiles.len();
    if pulses == 0 || range_len == 0 {
        return Vec::new();
    }

    // Lift magnitude profiles into the complex domain (im = 0) so we can
    // run the canonical complex slow-time DFT. We also clip each row to
    // `range_len` so the lifted shape matches the prior contract.
    let lifted: Vec<Vec<ComplexSample>> = profiles
        .iter()
        .map(|profile| {
            (0..range_len)
                .map(|range| {
                    let value = profile.get(range).copied().unwrap_or(0.0);
                    ComplexSample::new(value, 0.0)
                })
                .collect()
        })
        .collect();

    let complex_grid = slow_time_complex_dft(&lifted, pulses);

    // prior layout: `[doppler][range]`, normalised by `1/N_pulses`. The
    // complex grid is `[range][doppler]`, so transpose during the
    // collapse.
    let scale = 1.0 / pulses as f32;
    let mut output = vec![vec![0.0f32; range_len]; pulses];
    for (range, range_row) in complex_grid.iter().enumerate().take(range_len) {
        for (doppler, cell) in range_row.iter().enumerate().take(pulses) {
            output[doppler][range] = cell.norm() * scale;
        }
    }
    output
}
