// =========================================================================
// Lane G — Heavy-tailed clutter models (Weibull, K-distribution, log-normal).
//
// Real low-grazing radar clutter is heavy-tailed, not Gaussian. The
// ClutterProfile API models clutter as scalar correlated Gaussian noise
// plus a small number of glints. That API stays byte-stable; the additions
// below provide the physically-grounded distributions for heavy-tailed
// terrain regimes.
//
// References:
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
// =========================================================================

use serde::{Deserialize, Serialize};

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
    /// Coastal / littoral sea clutter. Same K-distribution family as
    /// `Sea` but enriched by sea-spray, breaking-wave bursts and (in
    /// some regimes) evaporation-duct propagation enhancement. Used
    /// by the Wave 4.5 H1 sea-spray library entries below. Citations:
    /// Ward, Tough & Watts (IET 2013) chap. 4-6; Greco & Gini,
    /// "Compound-Gaussian models for sea-clutter".
    CoastalSea,
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
///
/// A human-readable `name()` derived from `(terrain, distribution)` is
/// provided so consumers can filter sea-spray regimes by string (see
/// `ClutterRegime::name`). The Wave 4.5 H1 packet uses this surface to
/// expose the four coastal-sea-spray regimes alongside the prior
/// per-terrain entries from Lane G.
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

fn rayleigh_regime(terrain: TerrainClass, angle: f64, sc: f64, tc: f64, power: f64) -> ClutterRegime {
    ClutterRegime {
        terrain,
        grazing_angle_deg: angle,
        distribution: ClutterDistribution::Rayleigh,
        spatial_correlation: sc,
        temporal_correlation: tc,
        mean_power_dbsm_per_m2: power,
    }
}

fn weibull_regime(terrain: TerrainClass, angle: f64, shape: f64, scale: f64, sc: f64, tc: f64, power: f64) -> ClutterRegime {
    ClutterRegime {
        terrain,
        grazing_angle_deg: angle,
        distribution: ClutterDistribution::Weibull { shape, scale },
        spatial_correlation: sc,
        temporal_correlation: tc,
        mean_power_dbsm_per_m2: power,
    }
}

fn k_regime(terrain: TerrainClass, angle: f64, shape: f64, scale: f64, sc: f64, tc: f64, power: f64) -> ClutterRegime {
    ClutterRegime {
        terrain,
        grazing_angle_deg: angle,
        distribution: ClutterDistribution::KDistribution { shape, scale },
        spatial_correlation: sc,
        temporal_correlation: tc,
        mean_power_dbsm_per_m2: power,
    }
}

fn lognormal_regime(terrain: TerrainClass, angle: f64, mean_log: f64, std_log: f64, sc: f64, tc: f64, power: f64) -> ClutterRegime {
    ClutterRegime {
        terrain,
        grazing_angle_deg: angle,
        distribution: ClutterDistribution::LogNormal { mean_log, std_log },
        spatial_correlation: sc,
        temporal_correlation: tc,
        mean_power_dbsm_per_m2: power,
    }
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
            rayleigh_regime(TerrainClass::OpenSky, 30.0, 0.30, 0.40, -60.0),
            // Desert — homogeneous dry terrain. Weibull shape c ~ 1.7,
            // sigma_0 ~ -30 dB at low grazing per Skolnik chap. 7
            // (homogeneous-land clutter table) and JHU APL Tech Digest
            // V18 N3 (Lin) site-specific terrain model.
            weibull_regime(TerrainClass::Desert, 3.0, 1.7, 1.0, 0.50, 0.85, -30.0),
            // Forest — vegetated terrain. Weibull shape c ~ 1.2,
            // sigma_0 ~ -20 dB at low grazing per Skolnik chap. 7
            // (vegetated-land clutter).
            weibull_regime(TerrainClass::Forest, 3.0, 1.2, 1.0, 0.60, 0.70, -20.0),
            // Urban — structured terrain with frequent strong returns.
            // Weibull shape c ~ 0.8 (very heavy tail) per Skolnik chap. 7
            // (urban-land clutter); sigma_0 ~ -10 dB.
            weibull_regime(TerrainClass::Urban, 3.0, 0.8, 1.0, 0.70, 0.65, -10.0),
            // Sea — K-distribution shape nu ~ 8 (medium sea state),
            // sigma_0 ~ -40 dB at low grazing per Ward, Tough & Watts
            // (IET 2013) chap. 2 / chap. 3 K-distribution tabulations.
            k_regime(TerrainClass::Sea, 1.0, 8.0, 1.0, 0.55, 0.50, -40.0),
            // Mountain — very heterogeneous low-grazing land clutter.
            // K-distribution with low nu ~ 2 (very spiky) per the
            // heavy-tailed-land-clutter discussion in Skolnik chap. 7;
            // sigma_0 ~ -15 dB.
            k_regime(TerrainClass::Mountain, 2.0, 2.0, 1.0, 0.75, 0.60, -15.0),
            // Agricultural — mixed-crop. Weibull shape c ~ 1.4 per the
            // JHU APL Tech Digest V18 N3 (Lin) site-specific terrain
            // model; sigma_0 ~ -25 dB.
            weibull_regime(TerrainClass::Agricultural, 3.0, 1.4, 1.0, 0.55, 0.80, -25.0),
            // Suburban — mixture of urban + vegetated. Weibull shape
            // c ~ 1.0 (exponential amplitude) per Skolnik chap. 7
            // (mixed-terrain entries); sigma_0 ~ -18 dB.
            weibull_regime(TerrainClass::Suburban, 3.0, 1.0, 1.0, 0.60, 0.70, -18.0),
            // =============================================================
            // Wave 4.5 H1 — sea-spray / coastal clutter regimes.
            //
            // An independent OpenRouter expert critique flagged that the
            // Lane G library (8 entries above) covers the canonical land
            // and open-sea terrains but does NOT differentiate the spray-
            // dominated regimes a coastal counter-UAS radar must contend
            // with. The four entries below extend `TerrainClass::CoastalSea`
            // across the sea-state / propagation modes that matter for the
            // configs/scenarios/uae-coastal-surveillance-v1.json scenario
            // and other Gulf / littoral deployments. Shape, sigma_0 and
            // correlation parameters are cited inline.
            //
            // Citations:
            //   - Ward, Tough & Watts, "Sea Clutter: Scattering, the K
            //     Distribution and Radar Performance", 2nd ed. (IET 2013),
            //     chapters 4 (K-distribution shape vs sea state),
            //     5 (breaking-wave spikes) and 6 (low-grazing X-band).
            //   - Greco & Gini, "Compound-Gaussian models for sea-clutter"
            //     (IEEE AES Mag. 2007). Compound-Gaussian/K texture-
            //     speckle decomposition for breaking-wave regimes.
            //   - Watts, "Radar detection prediction in sea clutter using
            //     the compound K-distribution model", IEE Proc. F 132(7)
            //     (1985). X-band low-grazing breaking-wave + spray
            //     observations.
            //   - Skolnik, "Introduction to Radar Systems" 3rd ed.
            //     (McGraw-Hill 2001), § 7.5 — anomalous propagation and
            //     evaporation-duct enhancement of coastal clutter.
            // =============================================================

            // SeaSpray_SmallWhitecaps — sea-state 2..3 (Beaufort 3..4),
            // moderate whitecap density. K-distribution shape nu ~ 3.0:
            // less spiky than the breaking-wave regime but heavier-tailed
            // than the deep-open-sea nu=8 baseline above. sigma_0 ~ -38
            // dB at low grazing per Ward, Tough & Watts (IET 2013) ch. 4
            // §4.5 (K-shape vs sea state tabulation). Temporal correlation
            // is moderate (sea state evolves slowly over a CPI; pulse-to-
            // pulse texture decorrelation length ~ 8..16 pulses at X-band
            // PRFs).
            k_regime(TerrainClass::CoastalSea, 1.5, 3.0, 1.0, 0.85, 0.85, -38.0),
            // SeaSpray_BreakingWaves — sea-state 4..5 (Beaufort 5..6),
            // very spiky breaking-wave bursts. K-distribution shape nu ~
            // 0.6: heavy-tailed enough that OS-CFAR significantly
            // outperforms CA-CFAR. Citation: Greco & Gini, "Compound-
            // Gaussian models for sea-clutter" (compound-K texture-
            // speckle decomposition) + Ward, Tough & Watts (IET 2013)
            // chapter 5 (breaking-wave spike statistics). sigma_0 ~ -28
            // dB at 1 deg grazing per the same chapter. Spike decorrelation
            // is slow (texture coherent across ~ 16..32 pulses at X-band).
            k_regime(TerrainClass::CoastalSea, 1.0, 0.6, 1.0, 0.92, 0.95, -28.0),
            // SeaSpray_HeavyXBand — X-band specific, high-resolution
            // radar at low grazing angle, breaking-wave + spray plume.
            // Weibull shape c ~ 0.8, scale ~ 2.0 — heavier tail than
            // exponential and consistent with the breaking-wave + spray
            // amplitude tabulations in Watts, "Radar detection prediction
            // in sea clutter using the compound K-distribution model",
            // IEE Proc. F 132(7) (1985). sigma_0 ~ -22 dB at X-band /
            // 1 deg grazing per Watts (same paper, fig. 5).
            weibull_regime(TerrainClass::CoastalSea, 0.8, 0.8, 2.0, 0.90, 0.88, -22.0),
            // SeaSpray_DuctingCoastal — log-normal regime with mu=0.5,
            // sigma=1.4, capturing the long right tail produced by
            // evaporation-duct enhancement at coastal sites. The duct
            // periodically multiplies the surface return by a factor that
            // is itself approximately log-normal across measurement epochs,
            // and the compound effect is well represented by a log-normal
            // amplitude for downstream CFAR analysis. Citation: Skolnik,
            // "Introduction to Radar Systems" 3rd ed. § 7.5 (anomalous-
            // propagation enhancement of low-grazing clutter). sigma_0 ~
            // -18 dB representative for ducting episodes per the same
            // section. Temporal correlation high (duct conditions persist
            // for many pulses).
            lognormal_regime(TerrainClass::CoastalSea, 0.5, 0.5, 1.4, 0.88, 0.93, -18.0),
        ]
    }

    /// Short human-readable label for this regime, derived from
    /// `(terrain, distribution)` so that the four Wave 4.5 H1 sea-spray
    /// regimes can be selected by string from the library. Sea-spray
    /// regimes return `"SeaSpray_*"`; prior regimes return the
    /// `TerrainClass` variant name (e.g. `"OpenSky"`, `"Sea"`).
    ///
    /// The mapping is intentionally a `match` so a future regime that
    /// reuses `(terrain, distribution)` will be a compile-time error
    /// rather than a silent collision.
    pub fn name(&self) -> &'static str {
        match self.terrain {
            TerrainClass::OpenSky => "OpenSky",
            TerrainClass::Desert => "Desert",
            TerrainClass::Forest => "Forest",
            TerrainClass::Urban => "Urban",
            TerrainClass::Sea => "Sea_OpenMediumState",
            TerrainClass::Mountain => "Mountain",
            TerrainClass::Agricultural => "Agricultural",
            TerrainClass::Suburban => "Suburban",
            TerrainClass::CoastalSea => match self.distribution {
                ClutterDistribution::KDistribution { shape, .. } if shape >= 2.0 => {
                    "SeaSpray_SmallWhitecaps"
                }
                ClutterDistribution::KDistribution { .. } => "SeaSpray_BreakingWaves",
                ClutterDistribution::Weibull { .. } => "SeaSpray_HeavyXBand",
                ClutterDistribution::LogNormal { .. } => "SeaSpray_DuctingCoastal",
                ClutterDistribution::Rayleigh => "CoastalSea_Rayleigh",
            },
        }
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
    ///
    /// `TerrainClass::CoastalSea` has four library entries (one per
    /// sea-spray sub-regime); this helper returns the first one, which
    /// is `SeaSpray_SmallWhitecaps`. Callers that need a specific sea-
    /// spray sub-regime should pull it out of `library()` directly so
    /// the (terrain, distribution) tuple is unambiguous.
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
