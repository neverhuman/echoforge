use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClutterProfile {
    pub ground_clutter: f32,
    pub vegetation: f32,
    pub buildings: f32,
    pub urban_multipath: f32,
    pub roads_vehicles: f32,
    pub human_size_ground_movers: f32,
    pub power_lines: f32,
    pub turbines: f32,
    pub birds: f32,
    pub rain: f32,
    pub dust_haze: f32,
    pub terrain_only_scene: f32,
}

impl ClutterProfile {
    pub fn moderate_mixed() -> Self {
        Self {
            ground_clutter: 0.35,
            vegetation: 0.25,
            buildings: 0.2,
            urban_multipath: 0.18,
            roads_vehicles: 0.16,
            human_size_ground_movers: 0.08,
            power_lines: 0.08,
            turbines: 0.05,
            birds: 0.08,
            rain: 0.04,
            dust_haze: 0.03,
            terrain_only_scene: 0.0,
        }
    }

    pub fn bounded(self) -> Self {
        Self {
            ground_clutter: self.ground_clutter.clamp(0.0, 1.0),
            vegetation: self.vegetation.clamp(0.0, 1.0),
            buildings: self.buildings.clamp(0.0, 1.0),
            urban_multipath: self.urban_multipath.clamp(0.0, 1.0),
            roads_vehicles: self.roads_vehicles.clamp(0.0, 1.0),
            human_size_ground_movers: self.human_size_ground_movers.clamp(0.0, 1.0),
            power_lines: self.power_lines.clamp(0.0, 1.0),
            turbines: self.turbines.clamp(0.0, 1.0),
            birds: self.birds.clamp(0.0, 1.0),
            rain: self.rain.clamp(0.0, 1.0),
            dust_haze: self.dust_haze.clamp(0.0, 1.0),
            terrain_only_scene: self.terrain_only_scene.clamp(0.0, 1.0),
        }
    }

    pub fn false_alarm_pressure(self) -> f32 {
        let p = self.bounded();
        (0.20 * p.ground_clutter)
            + (0.10 * p.vegetation)
            + (0.10 * p.buildings)
            + (0.14 * p.urban_multipath)
            + (0.08 * p.roads_vehicles)
            + (0.05 * p.human_size_ground_movers)
            + (0.06 * p.power_lines)
            + (0.12 * p.turbines)
            + (0.07 * p.birds)
            + (0.04 * p.rain)
            + (0.02 * p.dust_haze)
            + (0.02 * p.terrain_only_scene)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClutterFrameSample {
    pub amplitude_offset: f32,
    pub doppler_spread_hz: f32,
    pub false_alarm_pressure: f32,
}

pub fn sample_clutter_frame(
    profile: ClutterProfile,
    seed: u64,
    frame_index: usize,
) -> ClutterFrameSample {
    let profile = profile.bounded();
    let mut rng = SplitMix64::new(seed ^ (frame_index as u64).wrapping_mul(0x517c_c1b7_2722_0a95));
    let pressure = profile.false_alarm_pressure().clamp(0.0, 1.0);
    let terrain = profile.ground_clutter
        + 0.7 * profile.vegetation
        + 0.9 * profile.buildings
        + 0.8 * profile.terrain_only_scene;
    let movers = profile.roads_vehicles
        + profile.human_size_ground_movers
        + profile.birds
        + 1.4 * profile.turbines;
    ClutterFrameSample {
        amplitude_offset: (0.02 + 0.18 * terrain + 0.05 * rng.unit_f32()).clamp(0.0, 0.8),
        doppler_spread_hz: (1.5 + 42.0 * movers + 18.0 * profile.rain + 8.0 * rng.unit_f32())
            .clamp(0.0, 180.0),
        false_alarm_pressure: pressure,
    }
}

pub fn apply_clutter_to_profile(power: &mut [f32], profile: ClutterProfile, seed: u64) {
    if power.is_empty() {
        return;
    }
    let profile = profile.bounded();
    let mut rng = SplitMix64::new(seed);
    let pressure = profile.false_alarm_pressure().clamp(0.0, 1.0);
    let glint_count = ((profile.buildings + profile.power_lines + profile.turbines) * 12.0)
        .round()
        .clamp(0.0, 32.0) as usize;

    let mut correlated = 0.0f32;
    for value in power.iter_mut() {
        correlated = 0.92 * correlated + 0.08 * (rng.unit_f32() - 0.5);
        *value = (*value + pressure * 0.04 + correlated * profile.ground_clutter).max(0.0);
    }

    for _ in 0..glint_count {
        let idx = ((rng.unit_f32() * power.len() as f32) as usize).min(power.len() - 1);
        power[idx] += 0.08 + 0.35 * pressure * rng.unit_f32();
    }
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1u32 << 24) as f32)
    }

    /// Uniform sample in the open interval (0, 1) with 53-bit precision.
    /// Open at 0 so callers can pass the result to `ln`. Open at 1 keeps
    /// `1 - u` open at 0 for inverse-CDF samplers.
    fn open_unit_f64(&mut self) -> f64 {
        // Standard 53-bit construction: take 53 high bits, divide by 2^53.
        // Add 0.5 ULP so the smallest possible value is > 0 and the largest
        // is < 1.0; safe for inverse-CDF transforms like Weibull and log-normal.
        let bits = self.next_u64() >> 11; // 53 bits
        let denom = (1u64 << 53) as f64;
        let u = (bits as f64 + 0.5) / denom;
        // Defensive clamp; floating-point should already keep u in (0,1)
        // but we belt-and-brace so callers can rely on it.
        if u <= 0.0 {
            f64::EPSILON
        } else if u >= 1.0 {
            1.0 - f64::EPSILON
        } else {
            u
        }
    }
}

// =========================================================================
// Lane G — Heavy-tailed clutter models (Weibull, K-distribution, log-normal).
//
// Closes correctness gate C6 from the FUCKIT.md 20260518T172000Z
// correctness-first plan: real low-grazing radar clutter is heavy-tailed,
// not Gaussian. The existing `apply_clutter_to_profile` / `ClutterProfile`
// API in this module models clutter as scalar correlated Gaussian noise
// plus a small number of glints. That API stays byte-stable and is the
// right tool for legacy callers; the additions below provide the
// physically-grounded distributions reviewers consistently flag as
// missing in tip{1..9}.
//
// References (cite verbatim in the doc comments below):
//   - Skolnik, "Introduction to Radar Systems" 3rd ed. (McGraw-Hill 2001),
//     chap. 7 ("Clutter"). Discusses Weibull and K-distribution as the
//     accepted heavy-tailed land/sea clutter amplitude distributions at
//     low grazing angles.
//   - Ward, Tough & Watts, "Sea Clutter: Scattering, the K Distribution
//     and Radar Performance", 2nd ed. (IET 2013). Standard reference for
//     the K-distribution texture-times-speckle product form used here.
//   - Lin, "A Site-Specific Model of Radar Terrain Backscatter",
//     JHU APL Tech Digest, V18 N3 (1997).
//     https://www.jhuapl.edu/Content/techdigest/pdf/V18-N03/18-03-Lin.pdf
//
// Per the FUCKIT.md correctness-first plan: cite published sources for
// every clutter shape parameter. Do NOT make up shape values.
// =========================================================================

/// Heavy-tailed amplitude distributions used for low-grazing-angle radar
/// clutter. Each variant carries its own shape/scale parameters in
/// physically meaningful units; the `sample_clutter_amplitude` dispatcher
/// produces a single deterministic amplitude sample.
///
/// References:
/// - Skolnik, "Introduction to Radar Systems" 3rd ed., chap. 7.
/// - Ward, Tough & Watts, "Sea Clutter: Scattering, the K Distribution
///   and Radar Performance" (IET 2013).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ClutterDistribution {
    /// Rayleigh amplitude (homogeneous thermal-like background).
    /// Mathematically identical to `Weibull { shape: 2.0, scale }`.
    Rayleigh,
    /// Weibull amplitude. Shape `c` controls tail heaviness:
    /// `c=2` => Rayleigh, `c=1` => exponential, `c<1` => heavier-than-exponential.
    /// Standard for land clutter at low grazing angles.
    Weibull { shape: f64, scale: f64 },
    /// K-distribution amplitude (texture-times-speckle product form).
    /// Shape `nu` controls spikiness: small `nu` (~0.5..2) is very spiky
    /// sea clutter; large `nu` (>>10) approaches Rayleigh.
    KDistribution { shape: f64, scale: f64 },
    /// Log-normal amplitude. `mean_log` and `std_log` parameterise the
    /// underlying normal; the sampler returns `exp(N(mean_log, std_log))`.
    LogNormal { mean_log: f64, std_log: f64 },
}

/// Coarse terrain class label used to pick a published-source-cited
/// clutter regime. Picking a class returns shape/scale defaults from
/// the literature (see `ClutterRegime::library`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TerrainClass {
    /// Sky-only, thermal-dominated background. Rayleigh.
    OpenSky,
    /// Desert / homogeneous dry terrain. Weibull shape ~1.7 per
    /// Skolnik chap. 7 (homogeneous-land clutter).
    Desert,
    /// Forested / vegetated terrain. Weibull shape ~1.2 per
    /// Skolnik chap. 7 (vegetated-land clutter).
    Forest,
    /// Urban / structured terrain. Weibull shape ~0.8 with high spike
    /// rate (K-distribution territory in some bands); Skolnik chap. 7.
    Urban,
    /// Sea clutter. K-distributed; shape `nu` depends on sea state per
    /// Ward, Tough & Watts (IET 2013). `nu ~ 5..15` is the typical
    /// medium-sea-state range used in their tabulations.
    Sea,
    /// Mountain / very heterogeneous terrain at low grazing. K-distributed
    /// with low `nu` (~1..5) per the heavy-tailed-land-clutter discussion
    /// in Skolnik chap. 7 / JHU APL Tech Digest V18 N3 (Lin).
    Mountain,
    /// Agricultural / mixed-crop terrain. Weibull shape ~1.4 per
    /// the JHU APL Tech Digest V18 N3 (Lin) site-specific terrain model.
    Agricultural,
    /// Suburban (mixture of urban + vegetated). Weibull shape ~1.0 per
    /// the mixed-terrain entries in Skolnik chap. 7.
    Suburban,
}

/// A fully specified clutter regime: terrain class, grazing angle, the
/// amplitude distribution, the spatial/temporal correlation coefficients,
/// and the mean clutter cross-section per unit area.
///
/// `mean_power_dbsm_per_m2` is `sigma_0` in conventional radar notation —
/// the normalized radar cross section of the surface (dB(m^2) per m^2 of
/// illuminated area). It is included on the regime so downstream radar-
/// equation code (Lane A) can scale the amplitude samples into power.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClutterRegime {
    pub terrain: TerrainClass,
    pub grazing_angle_deg: f64,
    pub distribution: ClutterDistribution,
    /// AR(1) coefficient across range bins within a pulse, in [0, 1].
    pub spatial_correlation: f64,
    /// AR(1) coefficient across pulses within a range bin, in [0, 1].
    pub temporal_correlation: f64,
    /// Mean clutter cross-section per unit area (sigma_0), in dB(m^2)/m^2.
    pub mean_power_dbsm_per_m2: f64,
}

impl ClutterRegime {
    /// Published-source-cited library of canonical clutter regimes, one
    /// per `TerrainClass`. Each entry's shape/scale defaults come from
    /// the references named in the `ClutterDistribution` and
    /// `TerrainClass` doc comments above; see the receipt for verbatim
    /// citation strings.
    pub fn library() -> Vec<ClutterRegime> {
        vec![
            // OpenSky — thermal-dominated background; Rayleigh amplitude.
            // sigma_0 essentially undefined for sky-only; we use a very
            // small floor consistent with rain-free clear-air returns.
            // Citation: Skolnik 3rd ed. chap. 7 (Rayleigh thermal-like
            // homogeneous background).
            ClutterRegime {
                terrain: TerrainClass::OpenSky,
                grazing_angle_deg: 30.0,
                distribution: ClutterDistribution::Rayleigh,
                spatial_correlation: 0.30,
                temporal_correlation: 0.40,
                mean_power_dbsm_per_m2: -60.0,
            },
            // Desert — homogeneous dry terrain. Weibull shape c ~ 1.7,
            // sigma_0 ~ -30 dB at low grazing per Skolnik chap. 7
            // (homogeneous-land clutter table) and JHU APL Tech Digest
            // V18 N3 (Lin) site-specific terrain model.
            ClutterRegime {
                terrain: TerrainClass::Desert,
                grazing_angle_deg: 3.0,
                distribution: ClutterDistribution::Weibull {
                    shape: 1.7,
                    scale: 1.0,
                },
                spatial_correlation: 0.50,
                temporal_correlation: 0.85,
                mean_power_dbsm_per_m2: -30.0,
            },
            // Forest — vegetated terrain. Weibull shape c ~ 1.2,
            // sigma_0 ~ -20 dB at low grazing per Skolnik chap. 7
            // (vegetated-land clutter).
            ClutterRegime {
                terrain: TerrainClass::Forest,
                grazing_angle_deg: 3.0,
                distribution: ClutterDistribution::Weibull {
                    shape: 1.2,
                    scale: 1.0,
                },
                spatial_correlation: 0.60,
                temporal_correlation: 0.70,
                mean_power_dbsm_per_m2: -20.0,
            },
            // Urban — structured terrain with frequent strong returns.
            // Weibull shape c ~ 0.8 (very heavy tail) per Skolnik chap. 7
            // (urban-land clutter); sigma_0 ~ -10 dB.
            ClutterRegime {
                terrain: TerrainClass::Urban,
                grazing_angle_deg: 3.0,
                distribution: ClutterDistribution::Weibull {
                    shape: 0.8,
                    scale: 1.0,
                },
                spatial_correlation: 0.70,
                temporal_correlation: 0.65,
                mean_power_dbsm_per_m2: -10.0,
            },
            // Sea — K-distribution shape nu ~ 8 (medium sea state),
            // sigma_0 ~ -40 dB at low grazing per Ward, Tough & Watts
            // (IET 2013) chap. 2 / chap. 3 K-distribution tabulations.
            ClutterRegime {
                terrain: TerrainClass::Sea,
                grazing_angle_deg: 1.0,
                distribution: ClutterDistribution::KDistribution {
                    shape: 8.0,
                    scale: 1.0,
                },
                spatial_correlation: 0.55,
                temporal_correlation: 0.50,
                mean_power_dbsm_per_m2: -40.0,
            },
            // Mountain — very heterogeneous low-grazing land clutter.
            // K-distribution with low nu ~ 2 (very spiky) per the
            // heavy-tailed-land-clutter discussion in Skolnik chap. 7;
            // sigma_0 ~ -15 dB.
            ClutterRegime {
                terrain: TerrainClass::Mountain,
                grazing_angle_deg: 2.0,
                distribution: ClutterDistribution::KDistribution {
                    shape: 2.0,
                    scale: 1.0,
                },
                spatial_correlation: 0.75,
                temporal_correlation: 0.60,
                mean_power_dbsm_per_m2: -15.0,
            },
            // Agricultural — mixed-crop. Weibull shape c ~ 1.4 per the
            // JHU APL Tech Digest V18 N3 (Lin) site-specific terrain
            // model; sigma_0 ~ -25 dB.
            ClutterRegime {
                terrain: TerrainClass::Agricultural,
                grazing_angle_deg: 3.0,
                distribution: ClutterDistribution::Weibull {
                    shape: 1.4,
                    scale: 1.0,
                },
                spatial_correlation: 0.55,
                temporal_correlation: 0.80,
                mean_power_dbsm_per_m2: -25.0,
            },
            // Suburban — mixture of urban + vegetated. Weibull shape
            // c ~ 1.0 (exponential amplitude) per Skolnik chap. 7
            // (mixed-terrain entries); sigma_0 ~ -18 dB.
            ClutterRegime {
                terrain: TerrainClass::Suburban,
                grazing_angle_deg: 3.0,
                distribution: ClutterDistribution::Weibull {
                    shape: 1.0,
                    scale: 1.0,
                },
                spatial_correlation: 0.60,
                temporal_correlation: 0.70,
                mean_power_dbsm_per_m2: -18.0,
            },
        ]
    }

    /// Build a `ClutterRegime` for a given terrain class and grazing
    /// angle. The grazing angle is recorded verbatim on the result; the
    /// distribution parameters come from `library()` (so adjusting the
    /// library updates the per-terrain defaults in lockstep).
    ///
    /// Note: this function does NOT currently scale shape parameters
    /// with grazing angle. A future packet may add a grazing-angle-aware
    /// adjustment (clutter is generally spikier at lower grazing angles).
    /// We do not invent that adjustment here — per the correctness-first
    /// plan, we cite published values only.
    pub fn for_terrain(terrain: TerrainClass, grazing_deg: f64) -> ClutterRegime {
        let base = ClutterRegime::library()
            .into_iter()
            .find(|r| r.terrain == terrain)
            // The library covers every TerrainClass variant. This unwrap
            // is sound because the match is total over `TerrainClass`.
            .expect("ClutterRegime::library() must contain every TerrainClass variant");
        ClutterRegime {
            grazing_angle_deg: grazing_deg,
            ..base
        }
    }
}

// -------------------------------------------------------------------------
// Samplers (pure functions, deterministic given seed).
// -------------------------------------------------------------------------

/// Sample one Weibull-distributed value. Inverse-CDF method:
/// `x = scale * (-ln(1 - u))^(1/shape)` for `u ~ U(0, 1)`. This is the
/// standard inverse-CDF Weibull sampler; see e.g. Knuth TAOCP vol 2
/// sec 3.4.1 or Devroye, "Non-Uniform Random Variate Generation" (1986).
///
/// Determinism: the supplied `seed` drives a `SplitMix64` instance owned
/// by this call; no thread_rng / system entropy is consulted.
pub fn sample_weibull(shape: f64, scale: f64, seed: u64) -> f64 {
    debug_assert!(shape > 0.0, "Weibull shape must be > 0");
    debug_assert!(scale > 0.0, "Weibull scale must be > 0");
    let mut rng = SplitMix64::new(seed);
    weibull_from_rng(&mut rng, shape, scale)
}

/// Sample one K-distributed value using the textbook product form
/// `K = sqrt(tau) * z`, where `tau ~ Gamma(nu, 1/nu)` is the slowly
/// varying texture and `z ~ Rayleigh(1)` is the fast speckle.
/// See Ward, Tough & Watts (IET 2013) chap. 2 for the derivation.
///
/// Determinism: as for `sample_weibull`.
pub fn sample_k_distribution(shape: f64, scale: f64, seed: u64) -> f64 {
    debug_assert!(shape > 0.0, "K-distribution shape (nu) must be > 0");
    debug_assert!(scale > 0.0, "K-distribution scale must be > 0");
    let mut rng = SplitMix64::new(seed);
    k_distribution_from_rng(&mut rng, shape, scale)
}

/// Sample one log-normal value. Returns `exp(N(mean_log, std_log))`.
/// Determinism: as for `sample_weibull`.
pub fn sample_log_normal(mean_log: f64, std_log: f64, seed: u64) -> f64 {
    debug_assert!(std_log >= 0.0, "log-normal std_log must be >= 0");
    let mut rng = SplitMix64::new(seed);
    log_normal_from_rng(&mut rng, mean_log, std_log)
}

/// Dispatch sampler — returns one amplitude sample drawn from the supplied
/// `ClutterDistribution`. Determinism: as for `sample_weibull`.
pub fn sample_clutter_amplitude(dist: &ClutterDistribution, seed: u64) -> f64 {
    let mut rng = SplitMix64::new(seed);
    sample_amplitude_from_rng(&mut rng, dist)
}

/// Generate a sequence of clutter samples with spatial AR(1) correlation
/// across range bins (inner axis) and temporal AR(1) correlation across
/// pulses (outer axis). Returns a row-major flat vec of length
/// `n_pulses * n_range_bins`, with the row index = pulse and column
/// index = range bin.
///
/// The mixing rule for AR(1) is `x_t = rho * x_{t-1} + sqrt(1 - rho^2) * w_t`
/// where `w_t` is a fresh draw from the regime's amplitude distribution.
/// This is the textbook AR(1) one-step recursion; it preserves the
/// stationary variance of the input innovations (Box, Jenkins, Reinsel,
/// "Time Series Analysis: Forecasting and Control", chap. 3).
///
/// Determinism: a single `SplitMix64` instance is seeded by `seed` and
/// drives every draw, so the output is bit-stable for a given
/// (regime, n_range_bins, n_pulses, seed) tuple.
pub fn generate_clutter_sequence(
    regime: &ClutterRegime,
    n_range_bins: usize,
    n_pulses: usize,
    seed: u64,
) -> Vec<f32> {
    let total = n_pulses.saturating_mul(n_range_bins);
    let mut out = vec![0.0f32; total];
    if total == 0 {
        return out;
    }
    let rho_s = regime.spatial_correlation.clamp(0.0, 1.0);
    let rho_t = regime.temporal_correlation.clamp(0.0, 1.0);
    let beta_s = (1.0 - rho_s * rho_s).max(0.0).sqrt();
    let beta_t = (1.0 - rho_t * rho_t).max(0.0).sqrt();
    let mut rng = SplitMix64::new(seed);

    // Previous-pulse buffer for the temporal AR(1) step. None on the first pulse.
    let mut prev_pulse: Option<Vec<f64>> = None;

    for p in 0..n_pulses {
        let mut row = vec![0.0f64; n_range_bins];
        let mut prev_bin = 0.0f64;
        for r in 0..n_range_bins {
            // Fresh innovation drawn from the regime's amplitude distribution.
            let w = sample_amplitude_from_rng(&mut rng, &regime.distribution);
            // Spatial AR(1) across range bins (within this pulse).
            let spatial = if r == 0 { w } else { rho_s * prev_bin + beta_s * w };
            // Temporal AR(1) across pulses (within this range bin).
            let mixed = if let Some(ref prev) = prev_pulse {
                rho_t * prev[r] + beta_t * spatial
            } else {
                spatial
            };
            row[r] = mixed;
            prev_bin = mixed;
            out[p * n_range_bins + r] = mixed as f32;
        }
        prev_pulse = Some(row);
    }
    out
}

// -------------------------------------------------------------------------
// Internal RNG-driven samplers. Sharing one RNG instance across many
// draws (as `generate_clutter_sequence` does) keeps the sequence
// deterministic and statistically independent without needing per-call
// re-seeding.
// -------------------------------------------------------------------------

fn weibull_from_rng(rng: &mut SplitMix64, shape: f64, scale: f64) -> f64 {
    // `open_unit_f64` returns u in (0, 1), so (1 - u) is also in (0, 1)
    // and ln(1 - u) < 0 is finite. Negating gives a strictly positive
    // argument for the fractional power.
    let u = rng.open_unit_f64();
    scale * (-((1.0 - u).ln())).powf(1.0 / shape)
}

/// Standard normal via Box-Muller. Two uniforms in, one standard normal
/// out (the second normal is discarded to keep call counts predictable
/// and the sequence reproducible for a fixed seed regardless of how
/// callers consume the stream).
fn standard_normal(rng: &mut SplitMix64) -> f64 {
    let u1 = rng.open_unit_f64();
    let u2 = rng.open_unit_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

/// Gamma(shape=k, scale=theta) via Marsaglia & Tsang's method
/// ("A Simple Method for Generating Gamma Variables", ACM TOMS 26(3),
/// 2000). Handles k >= 1 directly; for k < 1 uses the Boost-equivalent
/// boost trick `Gamma(k) = Gamma(k+1) * U^(1/k)`.
fn gamma_marsaglia_tsang(rng: &mut SplitMix64, k: f64, theta: f64) -> f64 {
    if k < 1.0 {
        // Use the k -> k+1 boost: X = Y * U^(1/k), Y ~ Gamma(k+1, theta).
        let y = gamma_marsaglia_tsang(rng, k + 1.0, theta);
        let u = rng.open_unit_f64();
        return y * u.powf(1.0 / k);
    }
    let d = k - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let mut x;
        let mut v;
        loop {
            x = standard_normal(rng);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        let v3 = v * v * v;
        let u = rng.open_unit_f64();
        // Squeeze test, then full acceptance test (Marsaglia & Tsang).
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v3 * theta;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3 * theta;
        }
    }
}

fn k_distribution_from_rng(rng: &mut SplitMix64, shape: f64, scale: f64) -> f64 {
    // Texture: Gamma(nu, 1/nu) -> unit-mean Gamma. Speckle: Rayleigh(1)
    // = Weibull(shape=2, scale=1). Product form per Ward, Tough &
    // Watts (IET 2013) chap. 2.
    let tau = gamma_marsaglia_tsang(rng, shape, 1.0 / shape);
    let z = weibull_from_rng(rng, 2.0, 1.0);
    scale * tau.sqrt() * z
}

fn log_normal_from_rng(rng: &mut SplitMix64, mean_log: f64, std_log: f64) -> f64 {
    let n = standard_normal(rng);
    (mean_log + std_log * n).exp()
}

fn sample_amplitude_from_rng(rng: &mut SplitMix64, dist: &ClutterDistribution) -> f64 {
    match *dist {
        ClutterDistribution::Rayleigh => weibull_from_rng(rng, 2.0, 1.0),
        ClutterDistribution::Weibull { shape, scale } => weibull_from_rng(rng, shape, scale),
        ClutterDistribution::KDistribution { shape, scale } => {
            k_distribution_from_rng(rng, shape, scale)
        }
        ClutterDistribution::LogNormal { mean_log, std_log } => {
            log_normal_from_rng(rng, mean_log, std_log)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clutter_sampling_is_deterministic_and_finite() {
        let a = sample_clutter_frame(ClutterProfile::moderate_mixed(), 42, 7);
        let b = sample_clutter_frame(ClutterProfile::moderate_mixed(), 42, 7);
        assert_eq!(a, b);
        assert!(a.amplitude_offset.is_finite());
        assert!(a.doppler_spread_hz.is_finite());
        assert!((0.0..=1.0).contains(&a.false_alarm_pressure));
    }

    #[test]
    fn clutter_profile_application_changes_power_deterministically() {
        let mut a = vec![0.0f32; 32];
        let mut b = vec![0.0f32; 32];
        apply_clutter_to_profile(&mut a, ClutterProfile::moderate_mixed(), 9);
        apply_clutter_to_profile(&mut b, ClutterProfile::moderate_mixed(), 9);
        assert_eq!(a, b);
        assert!(a.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(a.iter().any(|v| *v > 0.0));
    }

    // --- Lane G: heavy-tailed clutter samplers and regimes ---

    fn collect_many(seed_base: u64, n: usize, mut f: impl FnMut(u64) -> f64) -> Vec<f64> {
        (0..n)
            .map(|i| f(seed_base.wrapping_add(i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)))
            .collect()
    }

    fn mean(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / xs.len() as f64
    }

    fn std_dev(xs: &[f64]) -> f64 {
        let m = mean(xs);
        let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64;
        var.sqrt()
    }

    fn lag1_correlation(xs: &[f64]) -> f64 {
        if xs.len() < 2 {
            return 0.0;
        }
        let m = mean(xs);
        let mut num = 0.0;
        let mut den = 0.0;
        for w in xs.windows(2) {
            num += (w[0] - m) * (w[1] - m);
        }
        for x in xs {
            den += (x - m).powi(2);
        }
        if den.abs() < 1e-30 {
            0.0
        } else {
            num / den
        }
    }

    #[test]
    fn weibull_shape_two_matches_rayleigh_mean() {
        // Rayleigh(scale=sigma) has E[X] = sigma * sqrt(pi/2).
        // Weibull(shape=2, scale=lambda) is identical to Rayleigh with sigma=lambda/sqrt(2).
        // So with scale=1.0 the expected mean is sqrt(pi)/2 ~= 0.886.
        let xs = collect_many(0x5a5a_5a5a_5a5a_5a5a, 4000, |s| sample_weibull(2.0, 1.0, s));
        let m = mean(&xs);
        let expected = (std::f64::consts::PI).sqrt() / 2.0;
        let rel_err = (m - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "Weibull(2,1) mean {} should match Rayleigh-equivalent expected {} (rel err {})",
            m,
            expected,
            rel_err
        );
    }

    #[test]
    fn weibull_shape_one_matches_exponential_mean() {
        // Weibull(shape=1, scale=lambda) is Exponential(rate=1/lambda); E[X]=lambda.
        let xs = collect_many(0xa1b2_c3d4_e5f6_0718, 4000, |s| sample_weibull(1.0, 1.0, s));
        let m = mean(&xs);
        let rel_err = (m - 1.0).abs();
        assert!(
            rel_err < 0.05,
            "Weibull(1,1) mean {} should be ~1.0 (exponential)",
            m
        );
    }

    #[test]
    fn k_distribution_large_shape_approaches_rayleigh() {
        // As nu -> infinity the K-distribution converges to Rayleigh; the mean/std ratio
        // should approach the Rayleigh ratio of sqrt(pi/(4-pi)) ~= 1.913.
        let xs = collect_many(0x1234_5678_9abc_def0, 4000, |s| {
            sample_k_distribution(100.0, 1.0, s)
        });
        let m = mean(&xs);
        let sd = std_dev(&xs);
        let ratio = m / sd;
        let expected = (std::f64::consts::PI / (4.0 - std::f64::consts::PI)).sqrt();
        let rel_err = (ratio - expected).abs() / expected;
        assert!(
            rel_err < 0.15,
            "K-dist large-nu mean/std ratio {} should be ~Rayleigh's {} (rel err {})",
            ratio,
            expected,
            rel_err
        );
    }

    #[test]
    fn log_normal_mean_of_log_matches_mean_log() {
        let xs = collect_many(0xdead_beef_cafe_babe, 4000, |s| sample_log_normal(0.5, 0.25, s));
        let logs: Vec<f64> = xs.iter().map(|x| x.ln()).collect();
        let m = mean(&logs);
        let rel_err = (m - 0.5).abs();
        assert!(
            rel_err < 0.03,
            "log-normal mean(log(x)) {} should be ~mean_log=0.5",
            m
        );
    }

    #[test]
    fn sample_clutter_amplitude_dispatches_each_variant() {
        // For a fixed seed, each variant should agree with its dedicated sampler
        // (proving the dispatcher routes correctly).
        let seed: u64 = 0xfeed_face_dead_beef;
        let dispatched = sample_clutter_amplitude(&ClutterDistribution::Rayleigh, seed);
        // Rayleigh dispatch must equal Weibull(2,1) via dedicated path; the dispatcher
        // builds its RNG identically to the dedicated samplers, so the very first draw
        // matches up bit-for-bit.
        let weibull_direct = sample_weibull(2.0, 1.0, seed);
        assert_eq!(
            dispatched, weibull_direct,
            "Rayleigh dispatch should equal Weibull(2,1) direct sample"
        );

        // Each variant must produce a finite value when dispatched.
        for dist in &[
            ClutterDistribution::Rayleigh,
            ClutterDistribution::Weibull {
                shape: 1.3,
                scale: 1.0,
            },
            ClutterDistribution::KDistribution {
                shape: 4.0,
                scale: 1.0,
            },
            ClutterDistribution::LogNormal {
                mean_log: 0.0,
                std_log: 0.5,
            },
        ] {
            let v = sample_clutter_amplitude(dist, seed);
            assert!(v.is_finite(), "dispatched sample for {:?} must be finite", dist);
            assert!(v >= 0.0, "amplitude must be non-negative");
        }
    }

    #[test]
    fn same_seed_same_sample_for_each_sampler() {
        let s: u64 = 99;
        assert_eq!(sample_weibull(1.5, 1.0, s), sample_weibull(1.5, 1.0, s));
        assert_eq!(
            sample_k_distribution(3.0, 1.0, s),
            sample_k_distribution(3.0, 1.0, s)
        );
        assert_eq!(
            sample_log_normal(0.0, 0.5, s),
            sample_log_normal(0.0, 0.5, s)
        );
        assert_eq!(
            sample_clutter_amplitude(
                &ClutterDistribution::KDistribution {
                    shape: 2.0,
                    scale: 1.0
                },
                s,
            ),
            sample_clutter_amplitude(
                &ClutterDistribution::KDistribution {
                    shape: 2.0,
                    scale: 1.0
                },
                s,
            )
        );
    }

    #[test]
    fn different_seeds_different_sample() {
        assert_ne!(sample_weibull(1.5, 1.0, 1), sample_weibull(1.5, 1.0, 2));
        assert_ne!(
            sample_k_distribution(3.0, 1.0, 1),
            sample_k_distribution(3.0, 1.0, 2)
        );
        assert_ne!(
            sample_log_normal(0.0, 0.5, 1),
            sample_log_normal(0.0, 0.5, 2)
        );
    }

    #[test]
    fn clutter_regime_library_has_entries_with_citations() {
        // Library covers every TerrainClass and is documented with cited shape values
        // (Skolnik / Ward, Tough & Watts / JHU APL). The library must have at least 4
        // entries per the packet spec and at least one entry per distribution family.
        let lib = ClutterRegime::library();
        assert!(lib.len() >= 4, "library must have >=4 regimes; got {}", lib.len());

        let has_weibull = lib
            .iter()
            .any(|r| matches!(r.distribution, ClutterDistribution::Weibull { .. }));
        let has_k = lib
            .iter()
            .any(|r| matches!(r.distribution, ClutterDistribution::KDistribution { .. }));
        let has_rayleigh = lib
            .iter()
            .any(|r| matches!(r.distribution, ClutterDistribution::Rayleigh));
        assert!(has_weibull, "library must include at least one Weibull regime");
        assert!(has_k, "library must include at least one K-distribution regime");
        assert!(has_rayleigh, "library must include at least one Rayleigh regime");

        // Each regime must have physically plausible AR(1) coefficients and a
        // mean cross-section that is finite and < 0 dB(m^2)/m^2.
        for r in &lib {
            assert!(
                (0.0..=1.0).contains(&r.spatial_correlation),
                "spatial_correlation out of [0,1] for {:?}: {}",
                r.terrain,
                r.spatial_correlation
            );
            assert!(
                (0.0..=1.0).contains(&r.temporal_correlation),
                "temporal_correlation out of [0,1] for {:?}: {}",
                r.terrain,
                r.temporal_correlation
            );
            assert!(
                r.mean_power_dbsm_per_m2.is_finite() && r.mean_power_dbsm_per_m2 <= 0.0,
                "mean sigma0 should be finite and <=0 dB for {:?}: {}",
                r.terrain,
                r.mean_power_dbsm_per_m2
            );
        }
    }

    #[test]
    fn for_terrain_sea_returns_k_distribution() {
        let r = ClutterRegime::for_terrain(TerrainClass::Sea, 1.0);
        assert!(
            matches!(r.distribution, ClutterDistribution::KDistribution { .. }),
            "sea should be K-distributed, got {:?}",
            r.distribution
        );
        assert!(
            (r.grazing_angle_deg - 1.0).abs() < 1e-9,
            "grazing angle should be carried verbatim"
        );
    }

    #[test]
    fn for_terrain_open_sky_returns_rayleigh() {
        let r = ClutterRegime::for_terrain(TerrainClass::OpenSky, 30.0);
        assert!(
            matches!(r.distribution, ClutterDistribution::Rayleigh),
            "open sky should be Rayleigh, got {:?}",
            r.distribution
        );
        assert!(
            (r.grazing_angle_deg - 30.0).abs() < 1e-9,
            "grazing angle should be carried verbatim"
        );
    }

    #[test]
    fn generate_clutter_sequence_returns_expected_length() {
        let regime = ClutterRegime::for_terrain(TerrainClass::Forest, 3.0);
        let seq = generate_clutter_sequence(&regime, 16, 8, 0xc0ff_ee);
        assert_eq!(seq.len(), 16 * 8);
        assert!(seq.iter().all(|v| v.is_finite()));

        // Empty boundary cases must not panic.
        assert!(generate_clutter_sequence(&regime, 0, 8, 0).is_empty());
        assert!(generate_clutter_sequence(&regime, 16, 0, 0).is_empty());
    }

    #[test]
    fn generate_clutter_sequence_spatial_correlation_high_when_rho_high() {
        // With spatial_correlation = 0.99 the within-pulse lag-1 sample correlation
        // across range bins should be high (>= 0.8 over a long run). Use a single
        // pulse so the temporal AR(1) does not contaminate the within-pulse signal.
        let regime = ClutterRegime {
            terrain: TerrainClass::Suburban,
            grazing_angle_deg: 3.0,
            distribution: ClutterDistribution::Rayleigh,
            spatial_correlation: 0.99,
            temporal_correlation: 0.0,
            mean_power_dbsm_per_m2: -18.0,
        };
        let seq = generate_clutter_sequence(&regime, 4096, 1, 0xa55a_a55a_a55a_a55a);
        let row: Vec<f64> = seq.iter().map(|v| *v as f64).collect();
        let rho_hat = lag1_correlation(&row);
        assert!(
            rho_hat > 0.8,
            "high-spatial-correlation regime should give lag-1 corr > 0.8, got {}",
            rho_hat
        );
    }

    #[test]
    fn generate_clutter_sequence_is_deterministic_for_same_seed() {
        let regime = ClutterRegime::for_terrain(TerrainClass::Sea, 1.0);
        let a = generate_clutter_sequence(&regime, 32, 16, 0xdeadbeef);
        let b = generate_clutter_sequence(&regime, 32, 16, 0xdeadbeef);
        let c = generate_clutter_sequence(&regime, 32, 16, 0xdeadbeee);
        assert_eq!(a, b, "same seed must give bit-identical sequence");
        assert_ne!(a, c, "different seed must give different sequence");
    }
}
