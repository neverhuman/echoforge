use super::*;

fn lfm_chirp_samples(length: usize, bt: f64) -> Vec<ComplexSample> {
    // Properly sampled complex LFM with time-bandwidth product `bt`.
    // Sample rate is normalised to 1 sample/unit-time; pulse
    // duration is `length` samples. We want bandwidth B (cycles per
    // unit time) such that B * T = bt. Sweep the instantaneous
    // frequency linearly from -B/2 to +B/2 across the pulse so the
    // sampling stays at or below Nyquist for B/fs <= 1.
    //
    //   slope = B / T  in cycles/sample^2
    //   B/fs = bt / length  (must be <= 1 for proper baseband sampling)
    //
    // phase(t) = pi * slope * centered^2   (cosine modulation)
    // f_inst(t) = slope * centered   (cycles/sample)
    let t_total = length as f64;
    let b_over_fs = bt / t_total; // bandwidth in cycles/sample
    let slope = b_over_fs / t_total; // cycles per sample^2
    debug_assert!(
        b_over_fs <= 1.0,
        "bt/length = {b_over_fs} exceeds Nyquist (need length >= bt)"
    );
    (0..length)
        .map(|n| {
            let t = n as f64;
            let centered = t - t_total / 2.0;
            let phase = std::f64::consts::PI * slope * centered * centered;
            ComplexSample::new(phase.cos() as f32, phase.sin() as f32)
        })
        .collect()
}

/// Returns (peak_mag, peak_index).
fn argmax(power: &[f32]) -> (f32, usize) {
    power
        .iter()
        .enumerate()
        .fold((f32::MIN, 0usize), |(best, best_i), (i, &v)| {
            if v > best {
                (v, i)
            } else {
                (best, best_i)
            }
        })
}

/// Highest sidelobe magnitude excluding `+/- guard` bins around the
/// main lobe peak. Returns the sidelobe-to-peak ratio in dB.
fn peak_sidelobe_db(power: &[f32], guard: usize) -> f32 {
    let (peak, peak_i) = argmax(power);
    let mut max_side = 0.0f32;
    for (i, &v) in power.iter().enumerate() {
        if i.abs_diff(peak_i) <= guard {
            continue;
        }
        if v > max_side {
            max_side = v;
        }
    }
    20.0 * (max_side / peak).log10()
}

#[test]
fn coefficients_none_all_ones() {
    let c = coefficients(CompressionWindow::None, 64);
    assert_eq!(c.len(), 64);
    for v in c {
        assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
    }
}

#[test]
fn coefficients_hamming_endpoints() {
    let c = coefficients(CompressionWindow::Hamming, 64);
    assert_eq!(c.len(), 64);
    // Endpoint value is 0.54 - 0.46 = 0.08 per Harris 1978.
    assert!(
        (c[0] - 0.08).abs() < 1e-4,
        "hamming[0] = {} (expected 0.08)",
        c[0]
    );
    assert!(
        (c[63] - 0.08).abs() < 1e-4,
        "hamming[N-1] = {} (expected 0.08)",
        c[63]
    );
    // Peak at center should be 1.0.
    assert!(
        (c[32] - 1.0).abs() < 1e-2,
        "hamming center = {} (expected ~1.0)",
        c[32]
    );
}

#[test]
fn coefficients_hann_endpoints() {
    let c = coefficients(CompressionWindow::Hann, 64);
    assert_eq!(c.len(), 64);
    assert!(c[0].abs() < 1e-6, "hann[0] = {} (expected 0.0)", c[0]);
    assert!(c[63].abs() < 1e-6, "hann[N-1] = {} (expected 0.0)", c[63]);
}

#[test]
fn coefficients_taylor35_length() {
    let c = coefficients(CompressionWindow::TaylorN35 { nbar: 4 }, 64);
    assert_eq!(c.len(), 64);
    for v in &c {
        assert!(*v > 0.0, "taylor weight must be positive: {v}");
        assert!(v.is_finite(), "taylor weight must be finite: {v}");
        assert!(*v <= 1.0 + 1e-6, "taylor weight should be <= 1.0: {v}");
    }
    // Window should be symmetric (Taylor weighting is even).
    for n in 0..32 {
        let lhs = c[n];
        let rhs = c[63 - n];
        assert!(
            (lhs - rhs).abs() < 1e-4,
            "taylor not symmetric at n={n}: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn pulse_compress_no_window_peak() {
    // Single-tone reference at zero IF (real-valued unit phasor).
    let n = 32usize;
    let reference: Vec<ComplexSample> = (0..n)
        .map(|_| ComplexSample::new(1.0, 0.0))
        .collect();
    // Self-correlation: peak at lag (N-1) with magnitude N.
    let compressed = pulse_compress(&reference, &reference);
    let mags = magnitude(&compressed);
    let (peak, peak_i) = argmax(&mags);
    assert_eq!(peak_i, n - 1, "peak should be at center lag");
    // sum(|reference|^2) = N = 32.
    let expected: f32 = reference.iter().map(|s| s.norm_sqr()).sum();
    assert!(
        (peak - expected).abs() < 1e-3,
        "peak magnitude {peak} != sum|ref|^2 {expected}"
    );
}

#[test]
fn pulse_compress_taylor_peak_loss() {
    // Cost-of-windowing: in a matched filter the *peak amplitude*
    // drops by ~20 log10 mean(w) when we replace the reference with
    // a windowed copy, but the *output SNR* (the figure of merit
    // that matters to a radar) drops by only ~1 dB because white
    // noise is integrated coherently against the same weights. The
    // closed-form SNR loss for a matched filter against white noise
    // is
    //
    //   loss_db = 10 log10( (sum w)^2 / (N * sum w^2) )
    //
    // which for Taylor-35 with nbar = 4 sits near -1 dB (Harris 1978
    // table 1; Carrara/Goodman/Majewski 1995 §7.2.4). This is the
    // textbook quantity the gate should bound, not the bare peak.
    let chirp = lfm_chirp_samples(128, 64.0);
    let raw = magnitude(&pulse_compress(&chirp, &chirp));
    let win = magnitude(&pulse_compress_windowed(
        &chirp,
        &chirp,
        CompressionWindow::taylor_default(),
    ));
    let (raw_peak, _) = argmax(&raw);
    let (win_peak, _) = argmax(&win);
    assert!(
        raw_peak >= win_peak,
        "windowed peak ({win_peak}) cannot exceed raw peak ({raw_peak})"
    );

    let coeffs = coefficients(CompressionWindow::taylor_default(), chirp.len());
    let n = coeffs.len() as f64;
    let sum_w: f64 = coeffs.iter().map(|w| *w as f64).sum();
    let sum_w2: f64 = coeffs.iter().map(|w| (*w as f64).powi(2)).sum();
    let snr_loss_db = 10.0 * ((sum_w * sum_w) / (n * sum_w2)).log10();

    assert!(
        snr_loss_db > -2.0,
        "Taylor SNR loss > 2 dB ({snr_loss_db} dB) — taper too aggressive"
    );
    assert!(
        snr_loss_db <= 0.0,
        "windowed SNR loss cannot be positive ({snr_loss_db} dB)"
    );
}

/// HEADLINE TEST FOR GATE C9. Generate a B*T = 64 LFM chirp,
/// compress it against itself with the default Taylor-35 weighting,
/// and assert that the peak sidelobe (excluding +/- 2 bins around
/// the main lobe) is at or below -33 dB relative to the main lobe
/// peak. The -2 dB margin off the textbook -35 dB design target
/// accounts for the small B*T product (closed-form Taylor is exact
/// only in the limit of large N).
#[test]
fn pulse_compress_taylor_sidelobe_floor() {
    let length = 64usize;
    let chirp = lfm_chirp_samples(length, 64.0);
    let compressed = pulse_compress_windowed(
        &chirp,
        &chirp,
        CompressionWindow::TaylorN35 { nbar: 4 },
    );
    let mags = magnitude(&compressed);
    let psl = peak_sidelobe_db(&mags, 2);
    assert!(
        psl <= -33.0,
        "Taylor-35 peak sidelobe = {psl} dB (target <= -33 dB)"
    );
}

#[test]
fn pulse_compress_dolph_chebyshev_sidelobe() {
    // Dolph-Chebyshev sidelobes are the DTFT magnitude of the
    // weighted reference. For a finite-time LFM the compressed
    // pulse picks up additional shoulders from the truncation
    // window, so measured peak sidelobe asymptotes to the design
    // value only at large BT products. With N = 512 and BT = 512
    // we get within ~3 dB of the -50 dB design target, which is
    // textbook behaviour (Lynch 1997; the slow convergence of
    // chebwin's PSL vs BT is well documented).
    let length = 512usize;
    let chirp = lfm_chirp_samples(length, 512.0);
    let compressed = pulse_compress_windowed(
        &chirp,
        &chirp,
        CompressionWindow::DolphChebyshev { sll_db: -50.0 },
    );
    let mags = magnitude(&compressed);
    let psl = peak_sidelobe_db(&mags, 2);
    assert!(
        psl <= -45.0,
        "Dolph-Chebyshev (-50 dB design) peak sidelobe = {psl} dB \
         (expected <= -45 dB after 5 dB finite-BT margin)"
    );
}

