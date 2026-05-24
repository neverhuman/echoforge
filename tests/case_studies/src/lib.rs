//! Named-platform case studies — Wave 9 deliverable.
//!
//! Pairs a real named radar platform (loaded via `echoforge-packs`) with a
//! real named adversary platform, computes predicted detection range via
//! the monostatic radar equation, compares to vendor-declared range, and
//! renders an HTML credibility report.
//!
//! See `tests/saab_giraffe_vs_shahed_136.rs` for the first case study.

pub mod link_budget;
pub mod report;
