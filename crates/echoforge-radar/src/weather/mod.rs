//! Weather and sea-state physics primitives that extend the propagation
//! chain defined in [`crate::propagation`].
//!
//! Wave 14 of the radar-expert credibility sweep adds three new physics
//! modules layered on top of the ITU-R P.676 / P.838 / P.453 primitives
//! that already live in [`crate::propagation`]:
//!
//! - [`cloud_cover`] — ITU-R Recommendation P.840-8 cloud / fog
//!   attenuation in the Rayleigh region (`K_L` lookup × LWC × path).
//! - [`pierson_moskowitz`] — fully developed wind-sea spectrum
//!   `S(ω) = α·g²/ω⁵ · exp(-β·(g/(U·ω))⁴)` from Pierson & Moskowitz
//!   1964, plus `H_s`, `ω_peak`, and Beaufort-force helpers.
//! - [`modtran_eo_ir`] — MODTRAN-style band-integrated atmospheric
//!   transmission for the Vis / NIR / SWIR / MWIR / LWIR EO/IR bands.
//!
//! These primitives are consumed by the named `weather_profile` cards
//! shipped under `object-packs/weather-profiles-v1/` and ultimately by
//! the link-budget / EO-IR sensor archetypes that compose the
//! end-to-end detection chain.
//!
//! # Strict-open posture
//!
//! Every implementation cites the published source for each formula
//! and clamps frequency / wavelength inputs to the documented domain of
//! validity. The module never claims measured-truth fidelity; all values
//! are derived from public ITU-R, WMO, or peer-reviewed references.

pub mod cloud_cover;
pub mod modtran_eo_ir;
pub mod pierson_moskowitz;

pub use cloud_cover::{cloud_loss_db, cloud_specific_attenuation_db_per_km};
pub use modtran_eo_ir::{eo_ir_transmission, EoIrBand};
pub use pierson_moskowitz::{
    beaufort_force_from_u10, peak_frequency_rad_s, pierson_moskowitz_spectrum,
    significant_wave_height_m, GRAVITY_M_PER_S2, PHILLIPS_ALPHA, PM_BETA,
};
