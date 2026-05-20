use crate::detectors::phase_tiered::speed_classifier::PropulsionClass;

/// Heuristic micro-Doppler check: for piston targets look for any bin
/// in the blade-pass band [150, 220] Hz ± 15% whose power exceeds twice
/// the spectrum median (a cheap signal-vs-floor proxy). For jet targets
/// the dossier does not pin a specific blade-pass band, so the function
/// looks for an above-floor line *anywhere outside the body-Doppler
/// region*, which is what jet compressor signatures look like in the
/// public literature.
pub(super) fn check_blade_pass_line(
    spec: &[f32],
    doppler_bin_hz: f64,
    class: PropulsionClass,
) -> bool {
    if spec.is_empty() || doppler_bin_hz <= 0.0 {
        return false;
    }
    let n = spec.len();
    // Compute median as a stable floor estimator (the simple mean is
    // contaminated by any bright body-Doppler peak).
    let mut sorted: Vec<f32> = spec.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[n / 2].max(1e-12);

    let (lo_hz, hi_hz) = match class {
        PropulsionClass::Piston => {
            // 150–220 Hz ± 15% => [127.5, 253.0] Hz.
            (150.0 * 0.85, 220.0 * 1.15)
        }
        PropulsionClass::Jet => {
            // Jet compressor lines typically sit at the high-end of the
            // useful spectrum; the dossier does not bound them tightly,
            // so use the upper half-band of the available spectrum as a
            // proxy for "compressor-like high-frequency content".
            let nyquist_hz = doppler_bin_hz * (n as f64) / 2.0;
            (nyquist_hz * 0.5, nyquist_hz)
        }
        _ => return false,
    };
    let lo_bin = (lo_hz / doppler_bin_hz).max(0.0) as usize;
    let hi_bin = ((hi_hz / doppler_bin_hz) as usize).min(n - 1);
    if lo_bin >= hi_bin {
        return false;
    }
    spec[lo_bin..=hi_bin].iter().any(|&v| v >= 2.0 * median)
}
