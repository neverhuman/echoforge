//! Multi-modal sensor archetypes (Wave 16 of the radar-expert credibility
//! roadmap).
//!
//! This module ships three first-class non-radar sensor families that
//! plug into the counter-UAS chain alongside the named
//! `radar_platform_card` cards in `object-packs/radar-platforms-v1/`:
//!
//! - [`eo_ir`] — pinhole-camera / focal-plane-array model with
//!   Johnson-criteria detect / recognize / identify ranges, optionally
//!   composed with the MODTRAN band-integrated atmospheric transmission
//!   from [`crate::weather::modtran_eo_ir`].
//! - [`acoustic`] — ISO 9613-2 outdoor sound attenuation with
//!   source-SPL@100 m propagation to detection range; calibrated against
//!   Ukrainian "Sky Fortress" Shahed-class detection envelopes (Defense
//!   One, Reuters reporting).
//! - [`passive_rf`] — passive RF / electronic-support sensor model with
//!   AoA-method discriminator, TDoA cross-range accuracy, free-space
//!   link budget, and an explicit null-detection branch for
//!   pre-programmed RF-silent OWA drones (Shahed-class GNSS+INS).
//!
//! # References
//!
//! - Holst G. C., *Electro-Optical Imaging System Performance*, 6th ed.,
//!   SPIE 2017 — Chapters on NETD, IFOV, Johnson criteria.
//! - Johnson J., *Analysis of Image Forming Systems*, U.S. Army NVL 1958
//!   (republished 1985) — original 1 / 4 / 8 pixel "Johnson criteria"
//!   for detect / recognize / identify.
//! - ISO 9613-2:1996, *Acoustics — Attenuation of sound during
//!   propagation outdoors — Part 2: A general method of calculation*.
//! - Stein S., *Algorithms for ambiguity function processing*, IEEE
//!   Trans. ASSP 1981 — TDoA estimation framework.
//! - Robin Radar Systems, *Counter-UAS Sensor Architecture Survey*,
//!   2023 — passive-RF capability table referenced in
//!   `tips/detectors/tip1.txt` §4.6.
//!
//! # Strict-open posture
//!
//! Every formula is cited; every unit test pins a measured acceptance
//! gate. Vendor names appear only in the card layer
//! (`object-packs/sensors-multi-modal-v1/`) and only against published
//! datasheets. No measured-truth claim is made anywhere in this module.

pub mod acoustic;
pub mod eo_ir;
pub mod passive_rf;

pub use acoustic::{
    detection_range_m as acoustic_detection_range_m,
    iso_9613_2_attenuation_db_per_km,
    spl_at_range,
    AcousticSensorParams,
};
pub use eo_ir::{
    declared_range_for_task,
    eo_ir_snr_with_weather,
    instantaneous_fov_mrad,
    johnson_required_pixels,
    pixels_on_target,
    EoIrBand,
    EoIrSensorParams,
    JohnsonTask,
};
pub use passive_rf::{
    detection_range_m as passive_rf_detection_range_m,
    detects_rf_silent_target,
    tdoa_cross_range_accuracy_m,
    AoaMethod,
    PassiveRfSensorParams,
};
