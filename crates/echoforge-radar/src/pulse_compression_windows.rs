//! Window function implementations (Taylor, Dolph-Chebyshev).
//! Extracted from pulse_compression.rs for LOC compliance.

use std::f64::consts::PI;

/// Taylor weighting per Carrara, Goodman, Majewski 1995 §7.2.4.
pub(super) fn taylor(length: usize, sll_db: f64, nbar: usize) -> Vec<f32> {
    debug_assert!(sll_db < 0.0, "sll_db must be negative");
    debug_assert!(nbar >= 2, "nbar must be at least 2");

    let r = 10f64.powf(sll_db.abs() / 20.0);
    let a = r.ln_acosh() / PI;
    let nbar_f = nbar as f64;
    let sigma_sq = nbar_f * nbar_f / (a * a + (nbar_f - 0.5).powi(2));

    let mut f_coeffs = vec![0.0f64; nbar.saturating_sub(1)];
    for m in 1..nbar {
        let m_f = m as f64;
        let mut num = 1.0f64;
        for n in 1..nbar {
            let n_f = n as f64;
            num *= 1.0 - (m_f * m_f) / (sigma_sq * (a * a + (n_f - 0.5).powi(2)));
        }
        let mut den = 1.0f64;
        for n in 1..nbar {
            if n == m {
                continue;
            }
            let n_f = n as f64;
            den *= 1.0 - (m_f * m_f) / (n_f * n_f);
        }
        let sign = if m % 2 == 1 { 1.0 } else { -1.0 };
        f_coeffs[m - 1] = 0.5 * sign * (num / den);
    }

    let n_f = length as f64;
    let center = (length as f64 - 1.0) / 2.0;
    let mut weights = Vec::with_capacity(length);
    for n in 0..length {
        let np = (n as f64) - center;
        let mut w = 1.0f64;
        for m in 1..nbar {
            w += 2.0 * f_coeffs[m - 1] * (2.0 * PI * (m as f64) * np / n_f).cos();
        }
        weights.push(w as f32);
    }

    let max = weights
        .iter()
        .copied()
        .fold(f32::MIN, f32::max)
        .max(f32::EPSILON);
    for w in weights.iter_mut() {
        *w /= max;
    }
    weights
}

/// Dolph-Chebyshev weights.
pub(super) fn dolph_chebyshev(length: usize, sll_db_abs: f64) -> Vec<f32> {
    debug_assert!(sll_db_abs > 0.0, "sll_db_abs must be positive");
    debug_assert!(length >= 2);

    let n = length;
    let m = n - 1;
    let r = 10f64.powf(sll_db_abs / 20.0);
    let beta = (r.ln_acosh() / (m as f64)).cosh();

    let w_freq: Vec<f64> = (0..n)
        .map(|k| {
            let x = beta * (PI * (k as f64) / (n as f64)).cos();
            chebyshev_t(m, x)
        })
        .collect();

    let n_f = n as f64;
    let center = (n as f64 - 1.0) / 2.0;
    let mut w_time = vec![0.0f64; n];
    for n_idx in 0..n {
        let n_shift = (n_idx as f64) - center;
        let mut acc = 0.0f64;
        for k in 0..n {
            acc += w_freq[k] * (2.0 * PI * (k as f64) * n_shift / n_f).cos();
        }
        w_time[n_idx] = acc;
    }

    let max = w_time
        .iter()
        .copied()
        .fold(f64::MIN, f64::max)
        .max(f64::EPSILON);
    w_time.iter().map(|w| (w / max) as f32).collect()
}

fn chebyshev_t(n: usize, x: f64) -> f64 {
    if x.abs() <= 1.0 {
        let theta = x.acos();
        ((n as f64) * theta).cos()
    } else if x > 1.0 {
        let theta = x.ln_acosh();
        ((n as f64) * theta).cosh()
    } else {
        let theta = (-x).ln_acosh();
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        sign * ((n as f64) * theta).cosh()
    }
}

pub(super) trait LnAcosh {
    fn ln_acosh(self) -> f64;
}

impl LnAcosh for f64 {
    fn ln_acosh(self) -> f64 {
        if self <= 1.0 {
            0.0
        } else if self < 1e8 {
            (self + (self * self - 1.0).sqrt()).ln()
        } else {
            (2.0 * self).ln()
        }
    }
}
