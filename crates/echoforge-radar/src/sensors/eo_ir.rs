//! Electro-optical / infrared (EO/IR) sensor archetype.
//!
//! The model is a textbook pinhole + focal-plane-array (FPA) chain:
//!
//! 1. **Geometry.** Instantaneous field of view (IFOV) is set by detector
//!    pitch and lens focal length, `IFOV = pitch / f` (radians). With
//!    pitch expressed in micrometres and focal length in millimetres,
//!    `IFOV_mrad = (pitch_um / f_mm)` — the dimensions cancel down to a
//!    pure milliradian count (Holst 2017, ch. 4).
//! 2. **Pixels on target.** A target of cross-range extent `L` (m) at
//!    slant-range `R` (m) subtends `L / R` radians, which divided by IFOV
//!    gives the linear pixel count `N_pix = L / (R · IFOV)`.
//! 3. **Johnson 50 % task criteria.** Johnson 1958/1985 measured human
//!    observers and reported the line-pair (≡ pixel-cycle) counts needed
//!    on the minimum target dimension for 50 % probability of:
//!    * **Detect** ≥ 1 px on target,
//!    * **Recognize** ≥ 4 px,
//!    * **Identify** ≥ 8 px.
//!    The numbers are reproduced verbatim by `johnson_required_pixels`.
//!    Modern variants (e.g. Holst's "discrimination criteria" or
//!    DRI/N50/Vollmerhausen) replace these with finer values; we ship
//!    the canonical Johnson values because they are the most widely
//!    cited and they preserve the 1 / 4 / 8 ratio that the case
//!    studies in this crate are calibrated against.
//! 4. **Range solve.** With IFOV fixed and Johnson task `T` requiring
//!    `N_T` pixels on the cross-range extent `L`, the maximum task
//!    range follows directly from inverting (2):
//!    `R_T = L / (N_T · IFOV)`.
//! 5. **Atmosphere coupling.** Optionally compose with
//!    [`crate::weather::modtran_eo_ir::eo_ir_transmission`] to obtain
//!    end-to-end SNR including weather. This module ships the helper
//!    [`eo_ir_snr_with_weather`] that maps our band enum to the
//!    weather-module enum.
//!
//! # References
//!
//! - Holst G. C., *Electro-Optical Imaging System Performance*, 6th ed.,
//!   SPIE Press 2017 — §4 (pinhole geometry & IFOV) and §10 (Johnson
//!   criteria / N50 discrimination).
//! - Johnson J., *Analysis of Image Forming Systems*, U.S. Army Night
//!   Vision Laboratory 1958, republished in *Image Intensifier Symposium
//!   Proceedings* 1985 — the original 1 / 4 / 8 pixel-on-target table.
//! - MODTRAN — A. Berk et al., *MODTRAN 5.4 Update*, SSI 2014 — the
//!   atmospheric transmission tables consumed by
//!   [`crate::weather::modtran_eo_ir`].

use crate::weather::modtran_eo_ir::{eo_ir_transmission, EoIrBand as WeatherEoIrBand};

/// Spectral band identifier for an EO/IR sensor. Matches the
/// `band` enum of `schemas/eo_ir_sensor_card.schema.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EoIrBand {
    /// Colour visible (0.4-0.7 μm).
    VisColor,
    /// Mono visible (0.4-0.7 μm).
    VisMono,
    /// Near-IR (0.7-1.0 μm).
    Nir,
    /// Short-wave IR (1.5-2.5 μm).
    Swir,
    /// Mid-wave IR (3-5 μm), cooled detector (cold-shield, NETD 10-20 mK).
    MwirCooled,
    /// Mid-wave IR (3-5 μm), uncooled detector (NETD 30-60 mK).
    MwirUncooled,
    /// Long-wave IR (8-14 μm), cooled detector.
    LwirCooled,
    /// Long-wave IR (8-14 μm), uncooled microbolometer (NETD 30-80 mK).
    LwirUncooled,
}

impl EoIrBand {
    /// Map our band enum to the
    /// [`crate::weather::modtran_eo_ir::EoIrBand`] used by the MODTRAN
    /// transmission helper. The weather module only carries the
    /// five-band MODTRAN grouping (Vis / NIR / SWIR / MWIR / LWIR);
    /// cooled / uncooled detector variants share the same atmospheric
    /// band because the detector type does not change the path
    /// transmission, only the noise floor.
    pub fn weather_band(self) -> WeatherEoIrBand {
        match self {
            EoIrBand::VisColor | EoIrBand::VisMono => WeatherEoIrBand::Vis,
            EoIrBand::Nir => WeatherEoIrBand::Nir,
            EoIrBand::Swir => WeatherEoIrBand::Swir,
            EoIrBand::MwirCooled | EoIrBand::MwirUncooled => WeatherEoIrBand::Mwir,
            EoIrBand::LwirCooled | EoIrBand::LwirUncooled => WeatherEoIrBand::Lwir,
        }
    }
}

/// EO/IR sensor parameters used by the IFOV / Johnson / SNR
/// computations. All fields are positive.
#[derive(Debug, Clone, Copy)]
pub struct EoIrSensorParams {
    /// Spectral band.
    pub band: EoIrBand,
    /// Lens focal length (mm).
    pub focal_length_mm: f64,
    /// Entrance pupil / aperture diameter (mm). With focal length this
    /// pins f-number = `focal_length_mm / aperture_diameter_mm`.
    pub aperture_diameter_mm: f64,
    /// Horizontal field of view (deg).
    pub fov_h_deg: f64,
    /// Vertical field of view (deg).
    pub fov_v_deg: f64,
    /// Horizontal pixel count.
    pub array_h_pixels: u32,
    /// Vertical pixel count.
    pub array_v_pixels: u32,
    /// Detector pitch (μm). Equal in H and V for square pixels.
    pub pitch_um: f64,
    /// Noise-equivalent ΔT in millikelvins (thermal-band sensors only;
    /// caller may pass 0 for visible-band sensors where NETD is
    /// irrelevant).
    pub netd_mk: f64,
    /// Integration / exposure time (ms).
    pub integration_time_ms: f64,
    /// End-to-end optics transmission `[0, 1]`.
    pub optics_transmission: f64,
}

/// Instantaneous field of view (mrad) for a single detector cell.
///
/// `IFOV = pitch / focal_length` (radians). With pitch in μm and focal
/// length in mm, the dimensional cancellation gives `pitch_um /
/// focal_length_mm` mrad directly (Holst 2017 §4.2).
pub fn instantaneous_fov_mrad(params: &EoIrSensorParams) -> f64 {
    if params.focal_length_mm <= 0.0 {
        return 0.0;
    }
    // (pitch [m]) / (f [m]) = rad; convert to mrad by ×1000.
    //   = (pitch_um · 1e-6) / (focal_length_mm · 1e-3) · 1000
    //   = pitch_um / focal_length_mm     [mrad]  ← dimensional cancellation
    params.pitch_um / params.focal_length_mm
}

/// Linear pixel count on a target of cross-range extent
/// `target_cross_range_m` (m) at slant-range `range_m` (m).
///
/// `N_pix = L / (R · IFOV)` with IFOV in radians. Returns 0 for
/// non-positive range. Holst 2017 §10 — this is the "pixels-on-target"
/// metric that drives the Johnson task ladder.
pub fn pixels_on_target(
    params: &EoIrSensorParams,
    target_cross_range_m: f64,
    range_m: f64,
) -> f64 {
    if range_m <= 0.0 || target_cross_range_m <= 0.0 {
        return 0.0;
    }
    let ifov_rad = instantaneous_fov_mrad(params) * 1e-3;
    if ifov_rad <= 0.0 {
        return 0.0;
    }
    target_cross_range_m / (range_m * ifov_rad)
}

/// Johnson 50 %-probability discrimination task: detect, recognize, or
/// identify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JohnsonTask {
    /// 1 pixel on minimum target dimension — Johnson 1958.
    Detect,
    /// 4 pixels on minimum target dimension.
    Recognize,
    /// 8 pixels on minimum target dimension.
    Identify,
}

/// Required pixel count on the minimum target dimension for 50 %
/// probability of the chosen Johnson task. Returns the canonical 1 / 4
/// / 8 values (Johnson 1958, republished 1985).
pub fn johnson_required_pixels(task: JohnsonTask) -> u32 {
    match task {
        JohnsonTask::Detect => 1,
        JohnsonTask::Recognize => 4,
        JohnsonTask::Identify => 8,
    }
}

/// Solve for the maximum slant-range (m) at which the sensor still
/// places `johnson_required_pixels(task)` pixels on a target of
/// cross-range extent `target_cross_range_m`. Invert the pixel-on-target
/// formula:
///
/// ```text
/// N_T  = L / (R_T · IFOV)   ⇒   R_T = L / (N_T · IFOV)
/// ```
///
/// Returns `0.0` if any input is non-positive.
pub fn declared_range_for_task(
    params: &EoIrSensorParams,
    target_cross_range_m: f64,
    task: JohnsonTask,
) -> f64 {
    let n_pix = johnson_required_pixels(task) as f64;
    let ifov_rad = instantaneous_fov_mrad(params) * 1e-3;
    if n_pix <= 0.0 || ifov_rad <= 0.0 || target_cross_range_m <= 0.0 {
        return 0.0;
    }
    target_cross_range_m / (n_pix * ifov_rad)
}

/// End-to-end "effective SNR" proxy that composes the atmospheric
/// transmission with the optics transmission. Returns the dimensionless
/// product `τ_atm · τ_optics` clamped to `[1e-6, 1]`. This is *not* a
/// full radiometric calculation — for that the caller must supply
/// target ΔT, scene background, NETD, and detector quantum efficiency —
/// but it is the dominant range-dependent term that makes the difference
/// between "clear-air 8 km MWIR detection" and "fog 1 km MWIR
/// detection". Callers wanting a full NETD-equivalent contrast SNR can
/// multiply by `(ΔT_target / NETD)` separately.
///
/// Wave 14's [`crate::weather::modtran_eo_ir::eo_ir_transmission`] is
/// the upstream model; we map our richer band enum to its five-band
/// MODTRAN grouping via [`EoIrBand::weather_band`].
pub fn eo_ir_snr_with_weather(
    params: &EoIrSensorParams,
    range_m: f64,
    humidity_relative: f64,
    visibility_km: f64,
    aerosol_optical_depth: f64,
) -> f64 {
    let tau_atm = eo_ir_transmission(
        params.band.weather_band(),
        range_m,
        humidity_relative,
        visibility_km,
        aerosol_optical_depth,
    );
    let tau_optics = params.optics_transmission.clamp(0.0, 1.0);
    (tau_atm * tau_optics).clamp(1e-6, 1.0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn typical_cooled_mwir() -> EoIrSensorParams {
        // Worked-example sensor: 200 mm f/2 lens, 640×512 cooled-MWIR
        // FPA with 15 μm pitch and 18 mK NETD — matches
        // `tests/schemas/eo_ir_sensor_card.sample.json`.
        EoIrSensorParams {
            band: EoIrBand::MwirCooled,
            focal_length_mm: 200.0,
            aperture_diameter_mm: 100.0,
            fov_h_deg: 2.7,
            fov_v_deg: 2.2,
            array_h_pixels: 640,
            array_v_pixels: 512,
            pitch_um: 15.0,
            netd_mk: 18.0,
            integration_time_ms: 2.0,
            optics_transmission: 0.85,
        }
    }

    /// Test 1 — IFOV gate (Holst 2017 §4.2). For a 200 mm focal length
    /// with 15 μm pitch the IFOV is exactly 15 / 200 = 0.075 mrad
    /// (the formula is `pitch_um / focal_length_mm` mrad after
    /// dimensional cancellation).
    #[test]
    fn ifov_200mm_15um_is_075_mrad() {
        let params = typical_cooled_mwir();
        let ifov = instantaneous_fov_mrad(&params);
        let expected = 0.075;
        assert!(
            (ifov - expected).abs() < 0.005,
            "IFOV for 200 mm / 15 μm was {ifov:.4} mrad, expected {expected} ± 0.005"
        );
    }

    /// Test 2 — Johnson detect-range acceptance gate: a 2.5 m cross-
    /// range Shahed-class target (wingspan ≈ 2.5 m for the OWA platform
    /// addressed in `tips/detectors/tip1.txt` §3) detected at 1 px
    /// (Johnson detect) by the 0.075 mrad IFOV sensor must reach
    /// ≈ 33 km in vacuum (`R = L / (1 · IFOV) = 2.5 / 7.5e-5 ≈
    /// 33 333 m`). Acceptance gate: 33 km ± 2 km. The real-world
    /// detection range will be substantially shorter once weather and
    /// NETD-contrast losses are folded in; this gate isolates the
    /// pinhole geometry.
    #[test]
    fn johnson_detect_range_shahed_class_geometry_only() {
        let params = typical_cooled_mwir();
        let range_m =
            declared_range_for_task(&params, 2.5, JohnsonTask::Detect);
        let expected_m = 33_333.0;
        assert!(
            (range_m - expected_m).abs() < 2_000.0,
            "Johnson detect range was {range_m:.0} m, expected {expected_m} m ± 2000"
        );
    }

    /// Test 3 — Recognize range must equal detect range / 4 by the
    /// Johnson 1 / 4 / 8 pixel-count ladder. This is a structural
    /// invariant — the absolute geometry cancels out of the ratio.
    #[test]
    fn recognize_range_is_quarter_of_detect_range() {
        let params = typical_cooled_mwir();
        let detect = declared_range_for_task(&params, 2.5, JohnsonTask::Detect);
        let recognize = declared_range_for_task(&params, 2.5, JohnsonTask::Recognize);
        assert!(
            (recognize - detect / 4.0).abs() < 1.0,
            "recognize range {recognize:.1} m ≠ detect/4 ({:.1} m)",
            detect / 4.0
        );
    }

    /// Test 4 — `johnson_required_pixels` returns the canonical 1 / 4
    /// / 8 values that the test bench is calibrated against (Johnson
    /// 1958, republished 1985).
    #[test]
    fn johnson_required_pixels_canonical_values() {
        assert_eq!(johnson_required_pixels(JohnsonTask::Detect), 1);
        assert_eq!(johnson_required_pixels(JohnsonTask::Recognize), 4);
        assert_eq!(johnson_required_pixels(JohnsonTask::Identify), 8);
    }

    /// Test 5 — Pixels-on-target scales inversely with range (geometry
    /// of pinhole projection): doubling range halves the pixel count.
    #[test]
    fn pixels_on_target_halves_when_range_doubles() {
        let params = typical_cooled_mwir();
        let near = pixels_on_target(&params, 2.5, 5_000.0);
        let far = pixels_on_target(&params, 2.5, 10_000.0);
        assert!(
            (near - 2.0 * far).abs() < 0.01,
            "pixels-on-target should halve when range doubles: near={near:.4}, far={far:.4}"
        );
    }

    /// Test 6 — Weather composition reduces effective SNR vs vacuum
    /// path. Long-range MWIR through 5 km path / 60 % RH / 23 km
    /// visibility / AOD 0.15 must give an effective SNR proxy strictly
    /// less than the optics transmission alone but well above the
    /// `1e-6` clamp. With τ_atm ≈ 0.55 and τ_optics = 0.85 the product
    /// is ≈ 0.47.
    #[test]
    fn snr_with_weather_composes_with_optics() {
        let params = typical_cooled_mwir();
        let snr = eo_ir_snr_with_weather(&params, 5_000.0, 0.60, 23.0, 0.15);
        assert!(
            snr > 1e-3 && snr < params.optics_transmission,
            "expected weather-composed SNR strictly below optics τ ({}) and well above clamp; got {snr:.4}",
            params.optics_transmission
        );
        // Bracket: product should be ~0.45-0.5 with these inputs.
        assert!(
            (0.40..0.55).contains(&snr),
            "weather-composed SNR {snr:.3} outside 0.40-0.55 window for MWIR"
        );
    }
}
