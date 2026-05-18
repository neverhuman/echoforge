//! CSV persistence for micro-Doppler time-series and spectrograms.
//!
//! Kept inside `echoforge-validate` (rather than extending
//! `echoforge-sig::dynamic`) so the micro-Doppler primitive stays
//! self-contained and we avoid churn in the already-shipped sig crate.
//!
//! Format parity with `echoforge-sig::dynamic::write_state_sequence_csv`:
//! a single header row, then one row per record, lossless float
//! representation (`{:.17e}`).

use std::fs;
use std::io;
use std::path::Path;

use num_complex::Complex;

use super::rotating_rod::Spectrogram;

/// Write an IQ time-series as CSV with columns `t_s,real,imag,magnitude`.
pub fn write_time_series_csv(path: &Path, series: &[(f64, Complex<f64>)]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut buf = String::from("t_s,real,imag,magnitude\n");
    for (t, c) in series {
        buf.push_str(&format!(
            "{:.17e},{:.17e},{:.17e},{:.17e}\n",
            t,
            c.re,
            c.im,
            (c.re * c.re + c.im * c.im).sqrt()
        ));
    }
    fs::write(path, buf)
}

/// Write a spectrogram as CSV with columns `t_s,freq_hz,magnitude_db`.
/// One row per (frame, freq-bin) cell. The data is small for the
/// canonical micro-Doppler windows used in tests; if a binary tile format
/// becomes necessary it can replace this without changing call-sites.
pub fn write_spectrogram_csv(path: &Path, spec: &Spectrogram) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut buf = String::from("t_s,freq_hz,magnitude_db\n");
    for (frame_idx, t) in spec.times_s.iter().enumerate() {
        let row = &spec.magnitude_db[frame_idx];
        for (bin_idx, f) in spec.frequencies_hz.iter().enumerate() {
            buf.push_str(&format!("{:.17e},{:.17e},{:.17e}\n", t, f, row[bin_idx]));
        }
    }
    fs::write(path, buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn round_trip_time_series_csv() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ts.csv");
        let series = vec![
            (0.0_f64, Complex::new(1.0, 0.0)),
            (0.001_f64, Complex::new(0.5, -0.5)),
        ];
        write_time_series_csv(&path, &series).unwrap();
        let txt = fs::read_to_string(&path).unwrap();
        assert!(txt.starts_with("t_s,real,imag,magnitude\n"));
        assert_eq!(txt.lines().count(), 3);
    }
}
