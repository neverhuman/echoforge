use std::f32::consts::PI;

use super::tests_helpers::high_snr_episode;
use super::*;

/// A pure complex tone `x[n] = exp(j 2π f₀ n / N)` injected into a
/// single range bin must produce a DFT peak at Doppler bin `f₀` of
/// magnitude ~N. This is the canonical "delta-in-frequency-of-tone"
/// identity (Skolnik §3.5): the DFT is a coherent integrator that
/// concentrates a complex sinusoid into one bin while spreading
/// noise across all N. If we had thrown away phase before the
/// transform (the Lane B bug), the peak would be at DC (bin 0)
/// instead of bin `f₀`, because a magnitude sequence is real and
/// non-negative.
#[test]
fn slow_time_complex_dft_preserves_phase() {
    const N: usize = 32;
    const F0: usize = 4;
    let range_len: usize = 3;
    let target_range_bin: usize = 1;

    // Build N pulses, each with a single non-zero range bin holding
    // sample exp(j 2π f₀ n / N).
    let mut pulses: Vec<Vec<crate::ComplexSample>> = Vec::with_capacity(N);
    for n in 0..N {
        let mut profile = vec![crate::ComplexSample::new(0.0, 0.0); range_len];
        let phase = 2.0 * PI * (F0 as f32) * (n as f32) / N as f32;
        profile[target_range_bin] = crate::ComplexSample::new(phase.cos(), phase.sin());
        pulses.push(profile);
    }

    let grid = slow_time_complex_dft(&pulses, N);
    assert_eq!(grid.len(), range_len);
    assert_eq!(grid[target_range_bin].len(), N);

    // Find the peak Doppler bin at the populated range.
    let (peak_bin, peak_mag) = grid[target_range_bin]
        .iter()
        .enumerate()
        .map(|(k, c)| (k, c.norm()))
        .fold((0usize, 0.0f32), |(best_k, best_m), (k, m)| {
            if m > best_m {
                (k, m)
            } else {
                (best_k, best_m)
            }
        });

    assert_eq!(
        peak_bin, F0,
        "complex DFT peak landed at bin {peak_bin}, expected {F0}; \
         phase preservation likely broken"
    );
    assert!(
        (peak_mag - N as f32).abs() < 1e-3,
        "peak magnitude = {peak_mag}, expected ~{N}; coherent integration scaling broken",
    );

    // All other Doppler bins at this range should be near zero
    // (within DFT numerical precision).
    for (k, c) in grid[target_range_bin].iter().enumerate() {
        if k == F0 {
            continue;
        }
        assert!(
            c.norm() < 1e-3,
            "leak into bin {k}: |X[{k}]| = {} (expected ~0)",
            c.norm()
        );
    }

    // Empty range bins should hold all zeros.
    for (range, row) in grid.iter().enumerate().take(range_len) {
        if range == target_range_bin {
            continue;
        }
        for cell in row {
            assert!(cell.norm() < 1e-6, "spurious energy in empty range bin");
        }
    }
}

/// `slow_time_dft_magnitude` is now a wrapper that lifts real
/// magnitudes into the complex domain (im=0), runs
/// `slow_time_complex_dft`, then collapses to magnitude. This test
/// gates that the refactor is byte-stable per bin: feeding the
/// magnitudes-as-complex into the complex transform and taking
/// `.norm() / N_pulses` must reproduce the prior-output exactly
/// (modulo float summation order, which we hold constant).
#[test]
fn slow_time_complex_dft_magnitude_matches_prior() {
    const N: usize = 8;
    const RANGE_LEN: usize = 5;

    // Construct a deterministic real magnitude grid.
    let profiles: Vec<Vec<f32>> = (0..N)
        .map(|n| {
            (0..RANGE_LEN)
                .map(|r| 0.1 + (n as f32) * 0.07 + (r as f32) * 0.13)
                .collect()
        })
        .collect();

    // prior wrapper output (now backed by slow_time_complex_dft).
    let prior = slow_time_dft_magnitude(&profiles, RANGE_LEN);

    // Reference: directly compute the same DFT in real-magnitudes
    // form (the pre-Lane-B inlined implementation, kept here as the
    // ground truth).
    let mut reference = vec![vec![0.0f32; RANGE_LEN]; N];
    for (doppler, reference_row) in reference.iter_mut().enumerate().take(N) {
        for (range, reference_cell) in reference_row.iter_mut().enumerate().take(RANGE_LEN) {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (pulse, profile) in profiles.iter().enumerate() {
                let angle = -2.0 * PI * (doppler as f32) * (pulse as f32) / N as f32;
                let value = profile[range];
                re += value * angle.cos();
                im += value * angle.sin();
            }
            *reference_cell = (re * re + im * im).sqrt() / N as f32;
        }
    }

    assert_eq!(prior.len(), reference.len());
    for (doppler, (l_row, r_row)) in prior.iter().zip(reference.iter()).enumerate() {
        assert_eq!(l_row.len(), r_row.len());
        for (range, (l, r)) in l_row.iter().zip(r_row.iter()).enumerate() {
            assert!(
                (l - r).abs() < 1e-5,
                "doppler={doppler} range={range}: wrapper={l} reference={r}"
            );
        }
    }
}

/// Empty inputs must round-trip to empty output with no panic. Both
/// the all-zero-pulses path and the zero-Doppler-bins path are
/// exercised.
#[test]
fn slow_time_complex_dft_handles_empty() {
    let empty_pulses: Vec<Vec<crate::ComplexSample>> = Vec::new();
    let grid = slow_time_complex_dft(&empty_pulses, 16);
    assert!(grid.is_empty(), "empty pulse input must yield empty grid");

    let pulses: Vec<Vec<crate::ComplexSample>> =
        vec![vec![crate::ComplexSample::new(1.0, 0.0); 4]; 8];
    let grid_zero_doppler = slow_time_complex_dft(&pulses, 0);
    assert!(
        grid_zero_doppler.is_empty(),
        "n_doppler == 0 must yield empty grid"
    );

    // All-empty rows should also yield empty grid (no range bins).
    let zero_range: Vec<Vec<crate::ComplexSample>> = vec![Vec::new(); 8];
    let grid_zero_range = slow_time_complex_dft(&zero_range, 8);
    assert!(
        grid_zero_range.is_empty(),
        "all-empty pulse rows must yield empty grid"
    );

    // prior wrapper must also tolerate empty inputs.
    let mag = slow_time_dft_magnitude(&Vec::<Vec<f32>>::new(), 16);
    assert!(mag.is_empty());
    let mag_zero_range = slow_time_dft_magnitude(&vec![vec![1.0f32; 4]; 8], 0);
    assert!(mag_zero_range.is_empty());
}

/// `synthesize_takeoff_episode` must populate `range_doppler_complex`
/// with shape (compressed_len, n_pulses), where compressed_len is
/// the pulse-compression output length and n_pulses is the pulse
/// count. The grid must not be uniformly zero — a high-SNR scenario
/// has to produce coherent energy somewhere in the (range, Doppler)
/// plane.
#[test]
fn range_doppler_complex_populated() {
    let episode = high_snr_episode(16, 7);

    // Expected shape from the synthesis loop:
    //   reference len = sample_count = pulse_width_s * sample_rate_hz
    //   compressed_len = 2 * sample_count - 1
    let waveform = episode.config.waveform();
    let sample_count = waveform.samples().len();
    let compressed_len = sample_count.saturating_mul(2).saturating_sub(1);

    assert_eq!(
        episode.range_doppler_complex.len(),
        compressed_len,
        "outer dim must equal compressed_len (range bins)",
    );
    for row in &episode.range_doppler_complex {
        assert_eq!(
            row.len(),
            episode.config.pulse_count,
            "inner dim must equal pulse_count (Doppler bins)",
        );
        for cell in row {
            assert!(
                cell.re.is_finite() && cell.im.is_finite(),
                "non-finite complex cell at {cell}",
            );
        }
    }

    let max_mag = episode
        .range_doppler_complex
        .iter()
        .flat_map(|row| row.iter().map(|c| c.norm()))
        .fold(0.0f32, f32::max);
    assert!(
        max_mag > 0.0,
        "range_doppler_complex is all zeros — slow-time DFT is dead",
    );
}
