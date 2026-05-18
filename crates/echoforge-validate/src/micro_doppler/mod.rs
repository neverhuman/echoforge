//! Micro-Doppler primitives (rotating rigid scatterers).
//!
//! Adds analytic IQ time-series generators and a joint time-frequency
//! representation (JTFR) for canonical rotating geometries, in support of
//! tip3 of EchoForge's design tips (micro-Doppler differentiator).
//!
//! # FFT choice
//!
//! The spectrogram uses an in-crate radix-2 Cooley-Tukey FFT (≈40 LOC,
//! pure Rust) rather than pulling in `rustfft` here. Reasoning:
//!
//! * Keeps `echoforge-validate` dependency-light (consistent with the
//!   strict-open posture; no new workspace deps required).
//! * The analytic-truth crate is verification surface — the FFT is part of
//!   what we are asserting against, so the implementation lives next to
//!   the physics it scores.
//! * `rustfft` is already in the lockfile via `echoforge-radar` for the
//!   production CPU chain. The crates intentionally keep separate FFT
//!   implementations (validate = analytic oracle, radar = production).
//!
//! # Physics
//!
//! A rigid scatterer rotating with angular rate ω modulates the returned
//! IQ phase as e^{i 2 k r_t(t)} where r_t(t) is the projected range to a
//! moving point on the body. For a thin straight rod of length L rotating
//! in the boresight plane about its midpoint, the line-integral of the
//! per-point phase yields the closed-form sinc envelope used here. For an
//! N-blade propeller the rod contributions add coherently with azimuth
//! offsets 2π k / N, producing JTFR sidebands at integer multiples of the
//! blade-pass frequency f_bp = N · rpm / 60.
//!
//! References:
//! * Chen, "The Micro-Doppler Effect in Radar", Artech House (2011),
//!   chapters 4 (rotating targets) and 5 (helicopter/propeller).
//! * Martin & Mulgrew, "Analysis of the theoretical radar return signal
//!   from aircraft propeller blades", IEEE Radar Conf. 1990.

pub mod persist;
pub mod propeller;
pub mod rotating_rod;

pub use persist::{write_spectrogram_csv, write_time_series_csv};
pub use propeller::Propeller;
pub use rotating_rod::{sideband_spread_hz, spectrogram, RotatingRod, Spectrogram};
