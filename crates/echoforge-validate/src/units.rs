//! Unit and frame wrappers for canonical scatterer validation.
//!
//! Newtypes intentionally do not implement arithmetic; conversion functions are
//! the single allowed bridge between units to keep the analytic code explicit
//! about what it accepts.

use serde::{Deserialize, Serialize};

pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

macro_rules! scalar_newtype {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
        pub struct $name(pub f64);

        impl $name {
            #[inline]
            pub fn value(self) -> f64 {
                self.0
            }
        }
    };
}

scalar_newtype!(Hz);
scalar_newtype!(Meters);
scalar_newtype!(Radians);
scalar_newtype!(Degrees);
scalar_newtype!(SigmaM2);
scalar_newtype!(DbSm);

#[inline]
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * std::f64::consts::PI / 180.0
}

#[inline]
pub fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / std::f64::consts::PI
}

/// Convert RCS in square meters to dBsm.
///
/// Values <= 0 are floored to a tiny positive so the log stays finite; callers
/// that care about negative-sigma conditions should pre-check.
#[inline]
pub fn sigma_to_dbsm(sigma_m2: f64) -> f64 {
    let s = sigma_m2.max(1e-30);
    10.0 * s.log10()
}

#[inline]
pub fn dbsm_to_sigma(dbsm: f64) -> f64 {
    10f64.powf(dbsm / 10.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Frame {
    Enu,
    Ned,
    Body,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Polarization {
    H,
    V,
    L,
    R,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deg_rad_roundtrip() {
        for d in [-180.0, -90.0, 0.0, 1.0, 45.0, 90.0, 180.0] {
            assert!((rad_to_deg(deg_to_rad(d)) - d).abs() < 1e-12);
        }
    }

    #[test]
    fn dbsm_roundtrip() {
        for s in [1e-6, 1e-3, 1.0, 100.0, 1e4] {
            let db = sigma_to_dbsm(s);
            let back = dbsm_to_sigma(db);
            assert!((back / s - 1.0).abs() < 1e-12);
        }
    }
}
