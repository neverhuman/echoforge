//! Antenna manifold models for the EchoForge radar chain.
//!
//! The chain historically treated antennas as isotropic. This module adds a
//! small zoo of antenna-pattern models so downstream synthesis can apply
//! direction-dependent gain when projecting target returns onto a receiver:
//!
//! * [`IsotropicAntenna`] — uniform 0 dB everywhere (the prior assumption).
//! * [`CosinePatternAntenna`] — cosine-shaped main lobe with a configurable
//!   half-power beamwidth.
//! * [`TableLookupAntenna`] — bilinear interpolation across an azimuth/
//!   elevation sample grid (useful for measured patterns).
//! * [`PhasedArrayManifold`] — analytic uniform-linear-array pattern with a
//!   cosine taper applied per element, including main-beam steering.
//!
//! All patterns expose gain in dB through the [`AntennaPattern`] trait so the
//! synthesis layer can ask any model the same question.
//!
//! These models are deterministic, bounded, and have no side effects.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

/// Common interface implemented by every antenna pattern model in the crate.
pub trait AntennaPattern {
    /// Return the antenna gain in dB toward the supplied direction.
    ///
    /// `azimuth_deg` and `elevation_deg` are both measured from boresight,
    /// with positive elevation pointing up. Implementations must return a
    /// finite value for every input.
    fn gain_db(&self, azimuth_deg: f64, elevation_deg: f64) -> f64;
}

/// Uniform 0 dB pattern — the original implicit assumption of the chain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct IsotropicAntenna;

impl IsotropicAntenna {
    pub fn new() -> Self {
        Self
    }
}

impl AntennaPattern for IsotropicAntenna {
    fn gain_db(&self, _azimuth_deg: f64, _elevation_deg: f64) -> f64 {
        0.0
    }
}

/// Cosine-shaped main-lobe pattern with adjustable beamwidth.
///
/// The pattern peaks at `peak_gain_db` on boresight and falls off as a
/// cosine raised to a power chosen so that the half-power (−3 dB) points
/// occur at ±`beamwidth_deg / 2`. The pattern is forced to a very low
/// value beyond the front hemisphere (|az| or |el| ≥ 90°).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CosinePatternAntenna {
    pub peak_gain_db: f64,
    pub beamwidth_deg: f64,
}

impl CosinePatternAntenna {
    pub fn new(peak_gain_db: f64, beamwidth_deg: f64) -> Self {
        Self {
            peak_gain_db,
            beamwidth_deg,
        }
    }

    /// Exponent applied to `cos(angle)` so the power response in a single
    /// principal plane drops by 3 dB at `beamwidth_deg / 2`. With the
    /// pattern treated as a power-domain factor (10·log10), the constraint
    /// `cos^n(x_half) = 0.5` gives `n = ln(0.5)/ln(cos(x_half))`.
    fn cosine_exponent(&self) -> f64 {
        let half = (self.beamwidth_deg.abs() * 0.5).clamp(0.01, 89.99);
        let half_rad = half.to_radians();
        let denom = half_rad.cos().ln();
        if denom.abs() < 1e-9 {
            // beamwidth → 180° (almost flat): use exponent 1.
            return 1.0;
        }
        let n = 0.5f64.ln() / denom;
        n.clamp(0.5, 2_000.0)
    }
}

impl AntennaPattern for CosinePatternAntenna {
    fn gain_db(&self, azimuth_deg: f64, elevation_deg: f64) -> f64 {
        if azimuth_deg.abs() >= 90.0 || elevation_deg.abs() >= 90.0 {
            // Outside the front hemisphere we report a deep null instead
            // of −∞ to keep downstream arithmetic well-behaved.
            return self.peak_gain_db - 60.0;
        }
        let n = self.cosine_exponent();
        let az_rad = azimuth_deg.to_radians();
        let el_rad = elevation_deg.to_radians();
        let lobe = az_rad.cos().abs().powf(n) * el_rad.cos().abs().powf(n);
        let lobe = lobe.max(1e-6);
        self.peak_gain_db + 10.0 * lobe.log10()
    }
}

/// Bilinear lookup of antenna gain across an azimuth × elevation sample
/// grid. The samples are flat `(az_deg, el_deg, gain_db)` triples; the
/// implementation reconstructs the unique azimuth and elevation axes
/// internally, so the caller does not have to provide them in row-major
/// order — only that every (az, el) pair from the cartesian product of
/// the discovered axes appears exactly once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableLookupAntenna {
    pub samples: Vec<(f64, f64, f64)>,
}

impl TableLookupAntenna {
    pub fn new(samples: Vec<(f64, f64, f64)>) -> Self {
        Self { samples }
    }

    fn axes(&self) -> (Vec<f64>, Vec<f64>) {
        let mut az: Vec<f64> = self.samples.iter().map(|s| s.0).collect();
        let mut el: Vec<f64> = self.samples.iter().map(|s| s.1).collect();
        az.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        az.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        el.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        el.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        (az, el)
    }

    fn gain_at(&self, az: f64, el: f64) -> Option<f64> {
        self.samples
            .iter()
            .find(|(a, e, _)| (*a - az).abs() < 1e-6 && (*e - el).abs() < 1e-6)
            .map(|(_, _, g)| *g)
    }
}

impl AntennaPattern for TableLookupAntenna {
    fn gain_db(&self, azimuth_deg: f64, elevation_deg: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let (az_axis, el_axis) = self.axes();
        if az_axis.is_empty() || el_axis.is_empty() {
            return 0.0;
        }

        let az = azimuth_deg.clamp(az_axis[0], *az_axis.last().unwrap());
        let el = elevation_deg.clamp(el_axis[0], *el_axis.last().unwrap());

        let (a_lo, a_hi, t_az) = bracket(&az_axis, az);
        let (e_lo, e_hi, t_el) = bracket(&el_axis, el);

        let g_ll = self.gain_at(a_lo, e_lo).unwrap_or(0.0);
        let g_lh = self.gain_at(a_lo, e_hi).unwrap_or(g_ll);
        let g_hl = self.gain_at(a_hi, e_lo).unwrap_or(g_ll);
        let g_hh = self.gain_at(a_hi, e_hi).unwrap_or(g_ll);

        let g_lo = g_ll + (g_hl - g_ll) * t_az;
        let g_hi = g_lh + (g_hh - g_lh) * t_az;
        g_lo + (g_hi - g_lo) * t_el
    }
}

fn bracket(axis: &[f64], value: f64) -> (f64, f64, f64) {
    if axis.len() == 1 {
        return (axis[0], axis[0], 0.0);
    }
    for window in axis.windows(2) {
        let lo = window[0];
        let hi = window[1];
        if value <= hi {
            let span = (hi - lo).max(1e-12);
            let t = ((value - lo) / span).clamp(0.0, 1.0);
            return (lo, hi, t);
        }
    }
    let last = *axis.last().unwrap();
    (last, last, 0.0)
}

/// Analytic uniform-linear-array (ULA) manifold with a cosine taper applied
/// per element. The pattern is computed from the array factor of `n_elements`
/// equally-spaced elements, with phase progression chosen so the main beam
/// points at `(steering_az_deg, steering_el_deg)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhasedArrayManifold {
    pub n_elements: usize,
    pub element_spacing_m: f64,
    pub frequency_hz: f64,
    pub steering_az_deg: f64,
    pub steering_el_deg: f64,
}

impl PhasedArrayManifold {
    pub fn new(
        n_elements: usize,
        element_spacing_m: f64,
        frequency_hz: f64,
        steering_az_deg: f64,
        steering_el_deg: f64,
    ) -> Self {
        Self {
            n_elements,
            element_spacing_m,
            frequency_hz,
            steering_az_deg,
            steering_el_deg,
        }
    }

    fn wavelength_m(&self) -> f64 {
        let c = 299_792_458.0_f64;
        c / self.frequency_hz.max(1e-3)
    }
}

impl AntennaPattern for PhasedArrayManifold {
    fn gain_db(&self, azimuth_deg: f64, elevation_deg: f64) -> f64 {
        let n = self.n_elements.max(1);
        if n == 1 {
            return 0.0;
        }
        let lambda = self.wavelength_m();
        let k = 2.0 * PI / lambda;
        let d = self.element_spacing_m;

        // ULA along the azimuth axis: the path difference between adjacent
        // elements depends on sin(azimuth) and cos(elevation). Steering is
        // implemented as a phase progression that nulls the path difference
        // at the steering direction.
        let look = azimuth_deg.to_radians().sin() * elevation_deg.to_radians().cos();
        let steer =
            self.steering_az_deg.to_radians().sin() * self.steering_el_deg.to_radians().cos();
        let delta_phi = k * d * (look - steer);

        // Cosine taper across the aperture, normalized so the on-beam
        // response stays close to unity for any element count.
        let mut taper_sum = 0.0_f64;
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for i in 0..n {
            let pos = (i as f64) - (n as f64 - 1.0) * 0.5;
            // Half-cosine window so edges are tapered while the centre
            // weighs near unity.
            let taper = (PI * pos / (n as f64)).cos().max(0.0);
            taper_sum += taper;
            let phase = pos * delta_phi;
            re += taper * phase.cos();
            im += taper * phase.sin();
        }
        let taper_sum = taper_sum.max(1e-12);
        let magnitude = (re * re + im * im).sqrt() / taper_sum;
        20.0 * magnitude.max(1e-6).log10()
    }
}


#[cfg(test)]
#[path = "antenna_tests.rs"]
mod tests;
