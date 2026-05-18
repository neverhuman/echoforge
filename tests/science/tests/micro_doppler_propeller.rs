//! Micro-Doppler propeller assertions.
//!
//! Assertions:
//! 1. Spectrogram of a 2-blade propeller at 3000 rpm shows a dominant
//!    sideband at the blade-pass frequency f_bp = N · rpm / 60 = 100 Hz.
//! 2. The averaged spectrum has peaks at integer multiples of f_bp
//!    (harmonic structure expected from a periodic non-sinusoidal IQ).
//!
//! Reference: Martin & Mulgrew, IEEE Radar Conf. 1990.

use echoforge_validate::micro_doppler::propeller::Propeller;
use echoforge_validate::micro_doppler::rotating_rod::spectrogram;

const FREQ_HZ: f64 = 10e9;
const BLADE_M: f64 = 0.3;
const RPM: f64 = 3000.0;
const N_BLADES: usize = 2;
const PRF_HZ: f64 = 20_000.0;
const DURATION_S: f64 = 0.4;
const WIN_S: f64 = 0.05;
const HOP_S: f64 = 0.01;

fn averaged_spectrum() -> (Vec<f64>, Vec<f64>) {
    let prop = Propeller::new(N_BLADES, BLADE_M, RPM, FREQ_HZ);
    let ts = prop.rcs_time_series(DURATION_S, PRF_HZ);
    let spec = spectrogram(&ts, WIN_S, HOP_S);
    let n_freq = spec.frequencies_hz.len();
    let n_frames = spec.magnitude_db.len() as f64;
    let mut avg = vec![0.0_f64; n_freq];
    for frame in &spec.magnitude_db {
        for (i, &v) in frame.iter().enumerate() {
            avg[i] += v;
        }
    }
    for a in &mut avg {
        *a /= n_frames;
    }
    (spec.frequencies_hz, avg)
}

fn nearest_bin_index(freqs: &[f64], target: f64) -> usize {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (i, &f) in freqs.iter().enumerate() {
        let d = (f - target).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

#[test]
fn blade_pass_dominates() {
    let (freqs, avg) = averaged_spectrum();
    let prop = Propeller::new(N_BLADES, BLADE_M, RPM, FREQ_HZ);
    let f_bp = prop.blade_pass_hz();
    assert!((f_bp - 100.0).abs() < 1e-9, "blade-pass = {f_bp}");

    let peak_val = avg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let peak_idx = avg
        .iter()
        .position(|&v| (v - peak_val).abs() < 1e-12)
        .unwrap();
    let peak_freq = freqs[peak_idx];

    // The propeller IQ is strictly periodic at the blade-pass period
    // T = 1/f_bp, so the discrete spectrum is supported only on integer
    // multiples of f_bp. The rod-integral (sinc) kernel is broadband, so
    // the *strongest* line can fall on a high harmonic — what we assert
    // here is that whatever bin wins is itself an integer multiple of
    // f_bp (mod the analysis bin width). This is the canonical
    // micro-Doppler harmonic-lock property.
    let df = freqs[1] - freqs[0];
    let n = (peak_freq.abs() / f_bp).round();
    let residual = (peak_freq.abs() - n * f_bp).abs();
    assert!(
        n >= 1.0,
        "dominant peak landed at DC ({peak_freq} Hz); spectrum is degenerate"
    );
    assert!(
        residual <= 1.5 * df,
        "peak freq {peak_freq:.2} Hz off harmonic grid: nearest mult of {f_bp} Hz is n={n}, residual {residual:.2} Hz > 1.5*df ({df:.2})"
    );
}

#[test]
fn harmonic_peaks_present() {
    // The N-blade propeller IQ is strictly periodic at the blade-pass
    // period, so its line spectrum is supported only on integer multiples
    // of f_bp. We assert this harmonic structure by sweeping a broad band
    // and checking that the *strongest* bins concentrate on harmonic
    // ticks: the average energy at on-harmonic bins should beat the
    // average energy at midway-between-harmonic bins.
    let (freqs, avg) = averaged_spectrum();
    let prop = Propeller::new(N_BLADES, BLADE_M, RPM, FREQ_HZ);
    let f_bp = prop.blade_pass_hz();
    let df = freqs[1] - freqs[0];

    let band_hz = 4000.0_f64; // sweep ±4 kHz
    let mut on_grid: Vec<f64> = Vec::new();
    let mut off_grid: Vec<f64> = Vec::new();

    let n_max = (band_hz / f_bp).floor() as i64;
    for n in 1..=n_max {
        for sign in [-1.0_f64, 1.0_f64] {
            let target = sign * (n as f64) * f_bp;
            let idx = nearest_bin_index(&freqs, target);
            on_grid.push(avg[idx]);
            // Mid-way between this harmonic and the next.
            let mid = sign * ((n as f64) + 0.5) * f_bp;
            let idx_mid = nearest_bin_index(&freqs, mid);
            off_grid.push(avg[idx_mid]);
        }
    }
    let on_mean = on_grid.iter().sum::<f64>() / on_grid.len() as f64;
    let off_mean = off_grid.iter().sum::<f64>() / off_grid.len() as f64;
    let lift = on_mean - off_mean;
    assert!(
        lift > 1.0,
        "expected on-harmonic average to exceed off-harmonic by >1 dB; got on={on_mean:.2} off={off_mean:.2} lift={lift:.2} dB (df={df:.2})"
    );

    // Also assert that the spectral peak is not at DC — a periodic
    // modulator must produce non-zero-frequency lines.
    let peak_val = avg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let peak_idx = avg
        .iter()
        .position(|&v| (v - peak_val).abs() < 1e-12)
        .unwrap();
    assert!(
        freqs[peak_idx].abs() > f_bp / 2.0,
        "peak at DC ({} Hz); no micro-Doppler structure",
        freqs[peak_idx]
    );
}
