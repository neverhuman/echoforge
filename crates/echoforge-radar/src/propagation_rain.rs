//! Rain attenuation and refractivity helpers — extracted from propagation.rs
//! for LOC compliance. Declared as a child module of propagation.rs.

use super::RainPolarization;

pub(super) fn rain_kalpha_pair(freq_ghz: f64) -> (f64, f64, f64, f64) {
    const ANCHORS: &[(f64, f64, f64, f64, f64)] = &[
        (1.0, 0.0000259, 0.9691, 0.0000308, 0.8592),
        (3.0, 0.0001543, 1.0329, 0.0001533, 0.9491),
        (10.0, 0.01217, 1.2571, 0.01129, 1.2156),
        (15.0, 0.04481, 1.1233, 0.04164, 1.1044),
        (30.0, 0.2403, 0.9485, 0.2291, 0.9129),
        (40.0, 0.4365, 0.8516, 0.4274, 0.8126),
    ];
    let f = freq_ghz.clamp(ANCHORS[0].0, ANCHORS[ANCHORS.len() - 1].0);
    let log_f = f.log10();
    for win in ANCHORS.windows(2) {
        let (a, b) = (win[0], win[1]);
        if log_f >= a.0.log10() && log_f <= b.0.log10() {
            let span = b.0.log10() - a.0.log10();
            let t = if span > 0.0 {
                (log_f - a.0.log10()) / span
            } else {
                0.0
            };
            let k_h = lerp_log10(a.1, b.1, t);
            let alpha_h = a.2 + t * (b.2 - a.2);
            let k_v = lerp_log10(a.3, b.3, t);
            let alpha_v = a.4 + t * (b.4 - a.4);
            return (k_h, alpha_h, k_v, alpha_v);
        }
    }
    let a = if f <= ANCHORS[0].0 {
        ANCHORS[0]
    } else {
        ANCHORS[ANCHORS.len() - 1]
    };
    (a.1, a.2, a.3, a.4)
}

pub(super) fn lerp_log10(a: f64, b: f64, t: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return a + t * (b - a);
    }
    10f64.powf(a.log10() + t * (b.log10() - a.log10()))
}

/// Atmospheric refractivity `N(h)` (N-units) at altitude `altitude_m`
/// under the exponential reference atmosphere of ITU-R P.453-14 §1:
///
/// ```text
/// N(h) = N_0 · exp(-h / h_0),   N_0 = 315,  h_0 = 7350 m
/// ```
pub(super) fn refractivity_n_units(altitude_m: f64) -> f64 {
    const N0: f64 = 315.0;
    const H0: f64 = 7350.0;
    N0 * (-altitude_m / H0).exp()
}

/// Core of `itu_r_p676_gas_attenuation_db`. Called from propagation.rs.
pub(super) fn gas_attenuation_db(
    freq_ghz: f64,
    range_km: f64,
    temperature_k: f64,
    pressure_kpa: f64,
    water_vapor_g_per_m3: f64,
) -> f64 {
    if range_km <= 0.0 {
        return 0.0;
    }
    let f = freq_ghz.clamp(1.0, 40.0);
    const ANCHORS: &[(f64, f64)] = &[
        (1.0, 0.005),
        (3.0, 0.008),
        (10.0, 0.013),
        (15.0, 0.050),
        (30.0, 0.200),
        (40.0, 0.450),
    ];
    let gamma_ref = super::interpolate_linear(f, ANCHORS);
    let pressure_scale = (pressure_kpa / 101.325).max(0.0);
    let temperature_scale = if temperature_k > 0.0 {
        288.15 / temperature_k
    } else {
        1.0
    };
    let h2o_scale = if f >= 10.0 {
        (water_vapor_g_per_m3 / 7.5).max(0.0)
    } else {
        1.0
    };
    let gamma = gamma_ref * pressure_scale * temperature_scale * h2o_scale;
    gamma * range_km
}

/// Core of `itu_r_p838_rain_attenuation_db`. Called from propagation.rs.
pub(super) fn rain_attenuation_db(
    freq_ghz: f64,
    rain_rate_mm_per_hr: f64,
    range_km: f64,
    polarization: RainPolarization,
) -> f64 {
    if rain_rate_mm_per_hr <= 0.0 || range_km <= 0.0 {
        return 0.0;
    }
    let (k_h, alpha_h, k_v, alpha_v) = rain_kalpha_pair(freq_ghz);
    let (k, alpha) = match polarization {
        RainPolarization::Horizontal => (k_h, alpha_h),
        RainPolarization::Vertical => (k_v, alpha_v),
        RainPolarization::Circular => {
            let denom = k_h + k_v;
            let k_c = 0.5 * denom;
            let alpha_c = if denom > 0.0 {
                (k_h * alpha_h + k_v * alpha_v) / denom
            } else {
                0.0
            };
            (k_c, alpha_c)
        }
    };
    let gamma = k * rain_rate_mm_per_hr.powf(alpha);
    gamma * range_km
}
