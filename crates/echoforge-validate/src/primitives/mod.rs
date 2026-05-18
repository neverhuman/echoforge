//! Canonical analytic scatterer ground-truth primitives.
//!
//! Each primitive returns a [`Truth`] structure that records the analytic value
//! and the regime label (e.g. "rayleigh", "resonance", "optical", "broadside",
//! "main_lobe", "tip_on", "skipped") so the comparator can pick the right
//! tolerance band.

use crate::tolerance::ToleranceBand;
use crate::units::{Frame, Polarization};
use serde::{Deserialize, Serialize};

pub mod cone;
pub mod cylinder;
pub mod dihedral;
pub mod flat_plate;
pub mod sphere;
pub mod trihedral;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Pass,
    Warn,
    Fail,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Truth {
    pub value: f64,
    pub regime: String,
    pub status: Status,
}

impl Truth {
    pub fn ok(value: f64, regime: &str) -> Self {
        Self {
            value,
            regime: regime.to_string(),
            status: Status::Pass,
        }
    }
    pub fn skipped(reason: &str) -> Self {
        Self {
            value: f64::NAN,
            regime: reason.to_string(),
            status: Status::Skipped,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conditions {
    pub frequency_hz: f64,
    /// Azimuth/elevation angles in radians (0 = broadside/tip-on per primitive convention).
    pub theta_rad: f64,
    pub phi_rad: f64,
    pub tx_polarization: Polarization,
    pub rx_polarization: Polarization,
    pub frame: Frame,
}

impl Conditions {
    pub fn broadside(frequency_hz: f64) -> Self {
        Self {
            frequency_hz,
            theta_rad: 0.0,
            phi_rad: 0.0,
            tx_polarization: Polarization::V,
            rx_polarization: Polarization::V,
            frame: Frame::Body,
        }
    }

    pub fn wavelength_m(&self) -> f64 {
        crate::units::SPEED_OF_LIGHT / self.frequency_hz
    }
}

pub trait CanonicalTruth {
    fn sigma_m2(&self, conditions: &Conditions) -> Truth;
    fn validity_mask(&self, conditions: &Conditions) -> bool;
    fn tolerance(&self, conditions: &Conditions) -> ToleranceBand;
}
