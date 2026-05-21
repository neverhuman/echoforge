use crate::{slow_time_complex_dft, ComplexSample};

/// Complex slow-time DFT over a single range bin's slow-time IQ samples.
///
/// `iq` — N complex slow-time samples (one per pulse) for a single range bin.
/// `_pri_s` — pulse repetition interval in seconds (reserved for future
///   bin-frequency labelling; not used in the DFT computation itself).
///
/// Returns the N-bin complex DFT spectrum using the Skolnik convention
/// `X[k] = Σₙ x[n] · exp(-j · 2π · k · n / N)`. The peak bin `k` for a
/// target with radial velocity `v` (m/s), carrier `f_c` (Hz), and PRI `T`
/// (s) satisfies `k ≈ (2·v·f_c / c) · N·T` (wrapped to `[0, N)`).
pub fn slow_time_fft(iq: Vec<ComplexSample>, _pri_s: f64) -> Vec<ComplexSample> {
    let n = iq.len();
    if n == 0 {
        return Vec::new();
    }
    let pulses: Vec<Vec<ComplexSample>> = iq.into_iter().map(|s| vec![s]).collect();
    let mut grid = slow_time_complex_dft(&pulses, n);
    if grid.is_empty() {
        Vec::new()
    } else {
        grid.swap_remove(0)
    }
}
