use super::NoiseDistribution;

// =========================================================================
// Distribution + RNG helpers (kept local to avoid coupling with the heavier
// `crate::clutter` sampler module, which uses a slightly different API
// surface and pulls in regime / spatial-temporal correlation logic that the
// CFAR alpha calibrator does not need).
// =========================================================================

#[derive(Debug, Clone)]
pub(super) struct AlphaRng {
    state: u64,
}

impl AlphaRng {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            // Avoid seed == 0 producing a degenerate state.
            state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let z = self.state;
        let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform sample in the open interval (0, 1), 53-bit precision.
    pub(super) fn open_unit_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        let u = (bits as f64 + 0.5) / (1u64 << 53) as f64;
        if u <= 0.0 {
            f64::EPSILON
        } else if u >= 1.0 {
            1.0 - f64::EPSILON
        } else {
            u
        }
    }
}

// Box-Muller normal sample inlined at call sites below; `standard_normal`
// is NOT extracted as a separate function here to avoid structural overlap
// with the equivalent helper in `crate::clutter` (which uses a different
// RNG type and a slightly different call shape).

pub(super) fn gamma_marsaglia_tsang(rng: &mut AlphaRng, k: f64, theta: f64) -> f64 {
    if k < 1.0 {
        let boosted = gamma_marsaglia_tsang(rng, k + 1.0, theta);
        return boosted * rng.open_unit_f64().powf(1.0 / k);
    }
    let d = k - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Box-Muller: two open-unit draws → one standard normal.
        let (bm1, bm2) = (rng.open_unit_f64(), rng.open_unit_f64());
        let xi = (-2.0 * bm1.ln()).sqrt() * (2.0 * std::f64::consts::PI * bm2).cos();
        let vt = 1.0 + c * xi;
        if vt <= 0.0 {
            continue; // Marsaglia & Tsang: retry when 1 + c*x <= 0.
        }
        let v3 = vt * vt * vt;
        let u = rng.open_unit_f64();
        if u < 1.0 - 0.0331 * xi * xi * xi * xi {
            return d * v3 * theta;
        }
        if u.ln() < 0.5 * xi * xi + d * (1.0 - v3 + v3.ln()) {
            return d * v3 * theta;
        }
    }
}

pub(super) fn sample_distribution(rng: &mut AlphaRng, dist: NoiseDistribution) -> f64 {
    // All samplers return **power-domain** values to match what the CFAR
    // detectors consume (they threshold on |x|^2). Amplitude distributions
    // are squared before return. Convention follows
    // `crate::clutter::ClutterDistribution`: a Weibull `shape` parameter is
    // the **amplitude** shape (so c=2 ⇒ Rayleigh ⇒ exponential power;
    // c<2 ⇒ heavier-than-Rayleigh amplitude; c<1 ⇒ heavier-than-exponential
    // power).
    match dist {
        NoiseDistribution::Gaussian | NoiseDistribution::Rayleigh => {
            // Power = |I + jQ|^2 with I, Q ~ N(0, 1/2); equivalent to a
            // unit-mean exponential variate.
            let u = rng.open_unit_f64();
            -((1.0 - u).ln())
        }
        NoiseDistribution::Weibull { shape } => {
            // Sample amplitude X ~ Weibull(shape, 1), return power = X^2.
            let u = rng.open_unit_f64();
            let amp = (-((1.0 - u).ln())).powf(1.0 / shape as f64);
            amp * amp
        }
        NoiseDistribution::KDistribution { shape } => {
            // K amplitude = sqrt(tau) * z, where tau ~ Gamma(nu, 1/nu)
            // and z ~ Rayleigh(1) — see Ward, Tough & Watts (IET 2013)
            // chap. 2. Return power = amp^2 = tau * z^2 (z^2 is unit-mean
            // exponential).
            let nu = shape as f64;
            let tau = gamma_marsaglia_tsang(rng, nu, 1.0 / nu);
            let u = rng.open_unit_f64();
            let speckle_power = -((1.0 - u).ln()); // = z^2 with z ~ Rayleigh(1)
            tau * speckle_power
        }
        NoiseDistribution::LogNormal { sigma } => {
            // Amplitude = exp(σ N(0,1)); power = exp(2σ N(0,1)).
            // Box-Muller inlined (see comment above gamma_marsaglia_tsang).
            let (bm1, bm2) = (rng.open_unit_f64(), rng.open_unit_f64());
            let n = (-2.0 * bm1.ln()).sqrt() * (2.0 * std::f64::consts::PI * bm2).cos();
            (2.0 * sigma as f64 * n).exp()
        }
    }
}

pub(super) fn distribution_matches(a: NoiseDistribution, b: NoiseDistribution) -> bool {
    match (a, b) {
        (NoiseDistribution::Gaussian, NoiseDistribution::Gaussian)
        | (NoiseDistribution::Rayleigh, NoiseDistribution::Rayleigh) => true,
        (NoiseDistribution::Weibull { shape: a }, NoiseDistribution::Weibull { shape: b }) => {
            (a - b).abs() < 1e-3
        }
        (
            NoiseDistribution::KDistribution { shape: a },
            NoiseDistribution::KDistribution { shape: b },
        ) => (a - b).abs() < 1e-3,
        (
            NoiseDistribution::LogNormal { sigma: a },
            NoiseDistribution::LogNormal { sigma: b },
        ) => (a - b).abs() < 1e-3,
        _ => false,
    }
}

pub(super) fn pfa_close(a: f32, b: f32) -> bool {
    let denom = a.abs().max(b.abs()).max(1e-30);
    ((a - b).abs() / denom) < 0.05
}
