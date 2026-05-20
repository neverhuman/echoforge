//! ECM / Jamming models (Wave 18 — Skolnik §11 + Schleher EW).
//!
//! Electronic Countermeasures (ECM) cover the family of techniques an
//! adversary uses to degrade a radar's detection / tracking performance.
//! This module collects first-principles models for five classes that
//! Skolnik's *Introduction to Radar Systems* 3rd ed. chap. 11 treats as
//! the canonical taxonomy, supplemented by Schleher's *Electronic Warfare
//! in the Information Age* (Artech 1999) for the active deception sub-modes
//! that aren't given a closed form in Skolnik:
//!
//! - [`barrage`] — wide-band noise jamming (Skolnik §11.6).
//! - [`spot`] — narrow-band in-band noise jamming, with frequency-agility
//!   counter (Skolnik §11.6, §11.10).
//! - [`deception`] — range-gate pull-off (Schleher chap. 3-4).
//! - [`repeater`] — DRFM repeater / ghost-target jammer (Schleher chap. 4).
//! - [`gate_stealing`] — velocity-gate pull-off (Doppler analogue of
//!   range-gate stealing; Schleher chap. 4).
//!
//! Each sub-module is deterministic, allocation-explicit, and dimensionless
//! through unit suffixes on the public types (so an engineer reading a
//! `BarrageJammer { erp_dbw: 50.0, .. }` immediately knows the ERP is
//! 50 dBW, not dBm). The models intentionally stop at the first-order
//! radar-equation form — multipath, atmospheric losses, and antenna
//! sidelobe contributions are left to the caller, who can compose with
//! `link_budget`, `propagation`, and `antenna` as needed.
//!
//! # Posture
//!
//! Strict-open: every model is sourced to a public-domain radar handbook
//! (Skolnik 3rd ed., Schleher EW). No measured-truth claim is implied;
//! the J/S degradation numbers reported in unit tests are *simulator
//! consequences of the model parameters*, not field measurements.

pub mod barrage;
pub mod deception;
pub mod gate_stealing;
pub mod repeater;
pub mod spot;

pub use barrage::{
    jammer_received_power_w, jamming_to_signal_ratio_db, BarrageJammer,
};
pub use deception::{range_offset_at_time, DeceptionJammer};
pub use gate_stealing::{doppler_offset_at_time, VelocityGateStealer};
pub use repeater::{ghost_range_offsets_m, RepeaterJammer};
pub use spot::{spot_jamming_loss_db, SpotJammer};
