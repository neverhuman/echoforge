//! Public-proxy reference RCS tables (seeded_public_proxy_v1).
//!
//! These tables are **public-proxy references patterned after published
//! aggregate measurements; not measured equivalents; do not claim
//! platform-specific signature truth.** Each table carries a verbatim
//! citation of the source it was patterned after.

use super::{AspectGrid, Polarization, RcsLookup, SwerlingModel};

/// Build a row-major RCS table from an azimuth pattern and elevation offsets.
fn build_az_el_rcs(az_pattern: &[f64], el_offset: &[f64]) -> Vec<f64> {
    let mut rcs_dbsm = Vec::with_capacity(az_pattern.len() * el_offset.len());
    for az_val in az_pattern {
        for el_off in el_offset {
            rcs_dbsm.push(*az_val + *el_off);
        }
    }
    rcs_dbsm
}

/// 12-az × 3-el table (every 30 deg azimuth; 0/15/30 deg elevation).
/// Broadside (90 / 270 deg) bulges up to ~-10 dBsm; nose / tail dips
/// to ~-25 dBsm; elevation modulates by a few dB.
pub(super) fn small_fixed_wing_uas_x_band_vv() -> RcsLookup {
    let az_pattern: [f64; 12] = [
        -25.0, -22.0, -16.0, -10.0, -16.0, -22.0, -25.0, -22.0, -16.0, -10.0, -16.0, -22.0,
    ];
    let elevations: [f64; 3] = [0.0, 15.0, 30.0];
    let el_offset: [f64; 3] = [0.0, -1.5, -3.0];
    let rcs_dbsm = build_az_el_rcs(&az_pattern, &el_offset);
    RcsLookup {
        target_class: "fixed-wing-uas-small".to_string(),
        frequency_ghz: 10.0,
        polarization: Polarization::Vv,
        aspect_grid: AspectGrid {
            azimuth_deg: (0..12).map(|i| (i as f64) * 30.0).collect(),
            elevation_deg: elevations.to_vec(),
        },
        rcs_dbsm,
        fluctuation: SwerlingModel::Swerling1,
        citation: "MDPI Drones 2023, 7(1):39 (small fixed-wing UAV RCS aggregate distribution)"
            .to_string(),
        citation_url: Some("https://www.mdpi.com/2504-446X/7/1/39".to_string()),
    }
}

/// 12-az × 3-el table for a single large bird. Body-only return is
/// low (~-30 dBsm); broadside wing-flash spikes the cross-section to
/// ~-15 dBsm.
pub(super) fn single_large_bird_x_band_hh() -> RcsLookup {
    let az_pattern: [f64; 12] = [
        -30.0, -28.0, -22.0, -15.0, -22.0, -28.0, -30.0, -28.0, -22.0, -15.0, -22.0, -28.0,
    ];
    let elevations: [f64; 3] = [0.0, 15.0, 30.0];
    let el_offset: [f64; 3] = [0.0, -1.0, -2.5];
    let rcs_dbsm = build_az_el_rcs(&az_pattern, &el_offset);
    RcsLookup {
        target_class: "bird-large-single".to_string(),
        frequency_ghz: 10.0,
        polarization: Polarization::Hh,
        aspect_grid: AspectGrid {
            azimuth_deg: (0..12).map(|i| (i as f64) * 30.0).collect(),
            elevation_deg: elevations.to_vec(),
        },
        rcs_dbsm,
        fluctuation: SwerlingModel::Swerling3,
        citation: "Rahman & Robertson, Nature Sci Rep 8:17396 (2018) — Radar micro-Doppler signatures of drones and birds at K-band and W-band".to_string(),
        citation_url: Some("https://www.nature.com/articles/s41598-018-35880-9".to_string()),
    }
}

/// 12-az × 3-el table for a quadrotor. More omnidirectional than a
/// fixed-wing planform; small ±2 dB variation around -20 dBsm with mild
/// broadside bias. Pulse-to-pulse fluctuation (rotor chopping) → Swerling 2.
pub(super) fn quadrotor_x_band_vv() -> RcsLookup {
    let az_pattern: [f64; 12] = [
        -22.0, -21.5, -20.5, -19.5, -20.5, -21.5, -22.0, -21.5, -20.5, -19.5, -20.5, -21.5,
    ];
    let elevations: [f64; 3] = [0.0, 15.0, 30.0];
    let el_offset: [f64; 3] = [0.0, -0.5, -1.5];
    let rcs_dbsm = build_az_el_rcs(&az_pattern, &el_offset);
    RcsLookup {
        target_class: "quadrotor".to_string(),
        frequency_ghz: 10.0,
        polarization: Polarization::Vv,
        aspect_grid: AspectGrid {
            azimuth_deg: (0..12).map(|i| (i as f64) * 30.0).collect(),
            elevation_deg: elevations.to_vec(),
        },
        rcs_dbsm,
        fluctuation: SwerlingModel::Swerling2,
        citation: "Ezuma et al., arXiv:2102.11954 (UAV RF and RCS statistical recognition)"
            .to_string(),
        citation_url: Some("https://arxiv.org/abs/2102.11954".to_string()),
    }
}
