//! Rotating thin rod IQ time-series + STFT spectrogram.
//!
//! A thin perfectly-conducting rod of length L rotates in the plane normal
//! to the radar line-of-sight about its midpoint at angular rate omega.
//! The complex (IQ) return at observation time t is the line integral
//!
//! ```text
//!     E(t) = integral_{-L/2}^{+L/2} exp(i * 2k * x * sin(omega t)) dx
//!          = L * sinc(k L sin(omega t))
//! ```
//!
//! where sinc(u) := sin(u)/u (un-normalised) and k = 2 pi / lambda is the
//! wavenumber. The instantaneous Doppler shift is the time derivative of
//! the instantaneous phase, which is zero on average but produces JTFR
//! sidebands whose extent scales linearly with the tip velocity
//! v_tip = ω · L/2 — exactly the property we assert against in the tests.
//!
//! The output IQ here is real-valued (the symmetric integral has no
//! imaginary component about the boresight reference), but we return
//! `Complex<f64>` so the same time-series API supports the propeller case
//! where blades at azimuth offsets contribute complex phases.

use num_complex::Complex;

use crate::units::SPEED_OF_LIGHT;

/// Thin rotating rod parameters.
#[derive(Debug, Clone, Copy)]
pub struct RotatingRod {
    /// Tip-to-tip rod length (m).
    pub length_m: f64,
    /// Angular rotation rate (rad/s).
    pub omega_rad_s: f64,
    /// Carrier / illumination frequency (Hz).
    pub freq_hz: f64,
}

impl RotatingRod {
    pub fn new(length_m: f64, omega_rad_s: f64, freq_hz: f64) -> Self {
        Self {
            length_m,
            omega_rad_s,
            freq_hz,
        }
    }

    /// Wavenumber k = 2π / λ.
    #[inline]
    pub fn wavenumber(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.freq_hz / SPEED_OF_LIGHT
    }

    /// Instantaneous complex return for a single rod at azimuth offset
    /// `phi_offset_rad` (used by [`Propeller`]). The blade orientation at
    /// time `t` is `ω·t + phi_offset_rad`.
    pub fn sample(&self, t: f64, phi_offset_rad: f64) -> Complex<f64> {
        let k = self.wavenumber();
        let theta = self.omega_rad_s * t + phi_offset_rad;
        let u = k * self.length_m * theta.sin();
        let amp = if u.abs() < 1e-12 {
            self.length_m
        } else {
            self.length_m * (u.sin() / u)
        };
        Complex::new(amp, 0.0)
    }
}

/// Generate (t, IQ) pairs for `duration_s` at pulse-repetition `prf_hz`.
pub fn rcs_time_series(
    rod: &RotatingRod,
    duration_s: f64,
    prf_hz: f64,
) -> Vec<(f64, Complex<f64>)> {
    let n = (duration_s * prf_hz).floor() as usize;
    let dt = 1.0 / prf_hz;
    (0..n)
        .map(|i| {
            let t = i as f64 * dt;
            (t, rod.sample(t, 0.0))
        })
        .collect()
}

/// Joint time-frequency representation produced by [`spectrogram`].
#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// FFT bin centres (Hz), symmetric about zero, length = N_fft.
    pub frequencies_hz: Vec<f64>,
    /// Window-centre times (s), length = N_frames.
    pub times_s: Vec<f64>,
    /// Magnitude in dB, indexed as `magnitude_db[frame][freq_bin]`.
    pub magnitude_db: Vec<Vec<f64>>,
}

/// Compute an STFT spectrogram of an IQ time-series. The FFT length is
/// the largest power of two that fits in `window_s · fs` samples; the hop
/// is `hop_s · fs` samples (rounded). A Hann window is applied.
pub fn spectrogram(time_series: &[(f64, Complex<f64>)], window_s: f64, hop_s: f64) -> Spectrogram {
    assert!(time_series.len() >= 2, "need ≥2 samples for STFT");
    let dt = time_series[1].0 - time_series[0].0;
    let fs = 1.0 / dt;

    let raw_win = (window_s * fs).round() as usize;
    let n_fft = prev_pow2(raw_win.max(2));
    let hop = ((hop_s * fs).round() as usize).max(1);

    // Hann window (sum-to-one normalisation isn't needed; dB is relative).
    let hann: Vec<f64> = (0..n_fft)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n_fft as f64 - 1.0)).cos())
        .collect();

    let mut times_s = Vec::new();
    let mut magnitude_db = Vec::new();

    let mut frame_start = 0;
    while frame_start + n_fft <= time_series.len() {
        let mut buf: Vec<Complex<f64>> = (0..n_fft)
            .map(|i| time_series[frame_start + i].1 * hann[i])
            .collect();
        fft_radix2_inplace(&mut buf);

        // Centre frame time = mid-window time.
        let mid = time_series[frame_start + n_fft / 2].0;
        times_s.push(mid);

        // Shift to symmetric ordering (-fs/2 .. +fs/2) and convert to dB.
        let mut shifted = vec![0.0_f64; n_fft];
        for (i, c) in buf.iter().enumerate() {
            let mag = (c.norm_sqr() + 1e-30).sqrt();
            let db = 20.0 * mag.log10();
            // fftshift
            let j = (i + n_fft / 2) % n_fft;
            shifted[j] = db;
        }
        magnitude_db.push(shifted);

        frame_start += hop;
    }

    let frequencies_hz: Vec<f64> = (0..n_fft)
        .map(|i| {
            let signed = i as isize - (n_fft as isize) / 2;
            signed as f64 * fs / n_fft as f64
        })
        .collect();

    Spectrogram {
        frequencies_hz,
        times_s,
        magnitude_db,
    }
}

/// Occupied bandwidth (Hz) at a `drop_db` drop from peak. Computed as the
/// frequency extent of bins whose time-averaged magnitude is within
/// `drop_db` of the global peak.
pub fn sideband_spread_hz(spec: &Spectrogram) -> f64 {
    spread_at_drop(spec, 10.0)
}

/// Same as [`sideband_spread_hz`] but parameterised by drop.
pub fn spread_at_drop(spec: &Spectrogram, drop_db: f64) -> f64 {
    if spec.magnitude_db.is_empty() {
        return 0.0;
    }
    let n_freq = spec.frequencies_hz.len();
    let mut avg = vec![0.0_f64; n_freq];
    for frame in &spec.magnitude_db {
        for (i, &v) in frame.iter().enumerate() {
            avg[i] += v;
        }
    }
    let n_frames = spec.magnitude_db.len() as f64;
    for a in &mut avg {
        *a /= n_frames;
    }
    let peak = avg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let threshold = peak - drop_db;
    let (mut lo, mut hi) = (None::<f64>, None::<f64>);
    for (i, &v) in avg.iter().enumerate() {
        if v >= threshold {
            let f = spec.frequencies_hz[i];
            lo = Some(lo.map_or(f, |x| x.min(f)));
            hi = Some(hi.map_or(f, |x| x.max(f)));
        }
    }
    match (lo, hi) {
        (Some(a), Some(b)) => b - a,
        _ => 0.0,
    }
}

// ---- internal helpers -----------------------------------------------------

fn prev_pow2(n: usize) -> usize {
    if n < 2 {
        return 1;
    }
    let mut p = 1;
    while p * 2 <= n {
        p *= 2;
    }
    p
}

/// In-place radix-2 Cooley-Tukey FFT. `buf.len()` must be a power of two.
/// Iterative formulation with bit-reversal permutation; ~30 LOC and pure
/// safe Rust.
fn fft_radix2_inplace(buf: &mut [Complex<f64>]) {
    let n = buf.len();
    assert!(n.is_power_of_two(), "FFT length must be power of two");

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    // Cooley-Tukey butterflies.
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * std::f64::consts::PI / len as f64;
        let wlen = Complex::new(angle.cos(), angle.sin());
        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = buf[i + k];
                let t = buf[i + k + half] * w;
                buf[i + k] = u + t;
                buf[i + k + half] = u - t;
                w *= wlen;
            }
            i += len;
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_dft_small() {
        let n = 8;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64).cos(), (i as f64).sin()))
            .collect();
        let mut fft_out = input.clone();
        fft_radix2_inplace(&mut fft_out);
        // Naive DFT reference.
        let mut dft_out = vec![Complex::new(0.0, 0.0); n];
        for k in 0..n {
            let mut s = Complex::new(0.0, 0.0);
            for (m, x) in input.iter().enumerate() {
                let ang = -2.0 * std::f64::consts::PI * (k * m) as f64 / n as f64;
                s += *x * Complex::new(ang.cos(), ang.sin());
            }
            dft_out[k] = s;
        }
        for (a, b) in fft_out.iter().zip(dft_out.iter()) {
            assert!((a - b).norm() < 1e-9);
        }
    }

    #[test]
    fn rod_sample_zero_at_alignment() {
        // When the rod is aligned with the line of sight (θ = 0), the
        // line-integral is L (sinc(0) = 1).
        let rod = RotatingRod::new(0.3, 100.0 * std::f64::consts::PI, 10e9);
        let v = rod.sample(0.0, 0.0);
        assert!((v.re - 0.3).abs() < 1e-12);
    }

    #[test]
    fn time_series_length() {
        let rod = RotatingRod::new(0.3, 100.0 * std::f64::consts::PI, 10e9);
        let ts = rcs_time_series(&rod, 0.1, 10_000.0);
        assert_eq!(ts.len(), 1000);
    }
}
