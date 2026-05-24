//! N-blade propeller as a coherent sum of rotating rods.
//!
//! Each blade is modelled by [`RotatingRod`] at an azimuth offset
//! 2π·k/N. The composite IQ time-series is the coherent sum across
//! blades; the spectrogram and sideband-spread accessors delegate to the
//! rotating_rod module so a 1-blade propeller is bit-identical to the
//! standalone rod baseline.

use num_complex::Complex;

use super::rotating_rod::{
    rcs_time_series as rod_time_series, sideband_spread_hz as rod_sideband, spectrogram,
    spread_at_drop, RotatingRod, Spectrogram,
};

#[derive(Debug, Clone, Copy)]
pub struct Propeller {
    pub n_blades: usize,
    pub blade_length_m: f64,
    pub rpm: f64,
    pub freq_hz: f64,
}

impl Propeller {
    pub fn new(n_blades: usize, blade_length_m: f64, rpm: f64, freq_hz: f64) -> Self {
        assert!(n_blades >= 1, "propeller needs ≥1 blade");
        Self {
            n_blades,
            blade_length_m,
            rpm,
            freq_hz,
        }
    }

    /// Angular rate of the hub (rad/s).
    pub fn omega_rad_s(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.rpm / 60.0
    }

    /// Blade-pass frequency: N · rpm / 60 (Hz). The dominant
    /// micro-Doppler sideband sits at this frequency and its harmonics.
    pub fn blade_pass_hz(&self) -> f64 {
        self.n_blades as f64 * self.rpm / 60.0
    }

    /// Build the single-blade rod template.
    fn rod(&self) -> RotatingRod {
        RotatingRod::new(2.0 * self.blade_length_m, self.omega_rad_s(), self.freq_hz)
        // length_m on RotatingRod is tip-to-tip; a propeller blade of
        // physical length `blade_length_m` (hub→tip) sweeps a rod of
        // length 2·blade_length_m when mirrored about the hub.
    }

    /// IQ time-series of the full propeller at `prf_hz` over `duration_s`.
    pub fn rcs_time_series(&self, duration_s: f64, prf_hz: f64) -> Vec<(f64, Complex<f64>)> {
        let rod = self.rod();
        let n = (duration_s * prf_hz).floor() as usize;
        let dt = 1.0 / prf_hz;
        let two_pi_over_n = 2.0 * std::f64::consts::PI / self.n_blades as f64;
        (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                let mut acc = Complex::new(0.0, 0.0);
                for k in 0..self.n_blades {
                    let phi = two_pi_over_n * k as f64;
                    acc += rod.sample(t, phi);
                }
                (t, acc)
            })
            .collect()
    }

    pub fn spectrogram(
        &self,
        time_series: &[(f64, Complex<f64>)],
        window_s: f64,
        hop_s: f64,
    ) -> Spectrogram {
        spectrogram(time_series, window_s, hop_s)
    }

    pub fn sideband_spread_hz(&self, spec: &Spectrogram) -> f64 {
        rod_sideband(spec)
    }

    pub fn spread_at_drop(&self, spec: &Spectrogram, drop_db: f64) -> f64 {
        spread_at_drop(spec, drop_db)
    }
}

/// Convenience: build IQ for a propeller (mirrors the free function used
/// for [`RotatingRod`]).
pub fn rcs_time_series(prop: &Propeller, duration_s: f64, prf_hz: f64) -> Vec<(f64, Complex<f64>)> {
    prop.rcs_time_series(duration_s, prf_hz)
}

/// Convenience: standalone rod time-series wrapper for parity with
/// [`rcs_time_series`].
pub fn rod_rcs_time_series(rod: &RotatingRod, duration_s: f64, prf_hz: f64) -> Vec<Complex<f64>> {
    rod_time_series(rod, duration_s, prf_hz)
        .into_iter()
        .map(|(_, c)| c)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_blade_matches_rod_sample() {
        let prop = Propeller::new(1, 0.15, 3000.0, 10e9);
        let rod = prop.rod();
        let t = 0.001;
        let a = prop
            .rcs_time_series(0.002, 10_000.0)
            .into_iter()
            .find(|(tt, _)| (tt - t).abs() < 1e-9)
            .map(|(_, c)| c)
            .expect("sample present");
        let b = rod.sample(t, 0.0);
        assert!((a - b).norm() < 1e-12);
    }

    #[test]
    fn blade_pass_formula() {
        let p = Propeller::new(2, 0.3, 3000.0, 10e9);
        assert!((p.blade_pass_hz() - 100.0).abs() < 1e-9);
    }
}
