use super::*;

// These helpers are also used by tests_a.rs but are in module scope here.
// We redeclare them locally since each test module is independent.
fn collect_many(seed_base: u64, n: usize, mut f: impl FnMut(u64) -> f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            f(seed_base
                .wrapping_add(i as u64)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15))
        })
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

// ---------------------------------------------------------------------
// Wave 4.5 H1 — sea-spray / coastal-sea regimes.
//
// Independent-expert critique flagged the Lane G library as missing the
// sea-spray spike regime that dominates the false-alarm budget at low
// grazing on coastal sites (see configs/scenarios/uae-coastal-
// surveillance-v1.json). The tests below pin:
//   1. Presence of at least 4 sea-spray library entries.
//   2. Breaking-wave (nu=0.6) kurtosis is heavy-tailed (>= 6.0).
//   3. Small-whitecap (nu=3) kurtosis lies in the moderate band
//      (3.5..7.0; the K-distribution kurtosis depends on both texture
//      and speckle and is above the Rayleigh baseline of ~3.245).
//   4. Every sea-spray regime maps to TerrainClass::Sea or
//      TerrainClass::CoastalSea.
//
// Citations: Ward, Tough & Watts (IET 2013) chapters 4-6; Greco & Gini
// "Compound-Gaussian models for sea-clutter"; Watts IEE Proc. F 1985.
// ---------------------------------------------------------------------

/// Sample kurtosis (fourth standardised moment).
fn kurtosis(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let m = mean(xs);
    let m2 = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    let m4 = xs.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
    if m2 > 0.0 {
        m4 / (m2 * m2)
    } else {
        0.0
    }
}

#[test]
fn library_includes_sea_spray_regimes() {
    // Wave 4.5 H1 — at least four named sea-spray regimes must ship in
    // the library so coastal scenarios can pick them by terrain or by
    // name. We accept either `SeaSpray_*` (the canonical Wave 4.5 H1
    // names) or any regime whose name starts with `Spray`.
    let lib = ClutterRegime::library();
    let count = lib
        .iter()
        .filter(|r| r.name().contains("SeaSpray") || r.name().contains("Spray"))
        .count();
    assert!(
        count >= 4,
        "expected >=4 sea-spray regimes; found {} (names: {:?})",
        count,
        lib.iter().map(|r| r.name()).collect::<Vec<_>>(),
    );
}

#[test]
fn sea_spray_breaking_waves_kurtosis_high() {
    // Find the SeaSpray_BreakingWaves entry (K-distribution nu = 0.6).
    // 10 000 IID draws should give an empirical kurtosis >= 6.0 — the
    // K-distribution with very small shape parameter is strongly
    // heavy-tailed (Ward, Tough & Watts (IET 2013) chapter 5; OS-CFAR
    // outperforms CA-CFAR in this regime precisely because the tail
    // is so much heavier than the Rayleigh / Gaussian baseline ~3).
    let lib = ClutterRegime::library();
    let breaking = lib
        .iter()
        .find(|r| r.name() == "SeaSpray_BreakingWaves")
        .expect("SeaSpray_BreakingWaves must be in library");
    let dist = breaking.distribution;
    let xs = collect_many(0xb16b_00b5_dead_beef, 10_000, |s| {
        sample_clutter_amplitude(&dist, s)
    });
    let k = kurtosis(&xs);
    assert!(
        k >= 6.0,
        "SeaSpray_BreakingWaves (K nu=0.6) sample kurtosis {} should be >= 6.0",
        k
    );
}

#[test]
fn sea_spray_small_whitecaps_kurtosis_moderate() {
    // SeaSpray_SmallWhitecaps is K-distribution with nu=3.0. For the
    // K-amplitude with shape nu, the closed-form fourth moment vs
    // second moment ratio is finite for nu>1 and yields an excess
    // kurtosis that decays as ~ 6/nu (Ward, Tough & Watts (IET 2013)
    // chapter 4). With nu=3 the empirical kurtosis sits in the
    // moderate band roughly 3.5..7.0 — measurably above the Rayleigh
    // baseline (~3.245) but well below the breaking-wave regime.
    let lib = ClutterRegime::library();
    let small = lib
        .iter()
        .find(|r| r.name() == "SeaSpray_SmallWhitecaps")
        .expect("SeaSpray_SmallWhitecaps must be in library");
    let dist = small.distribution;
    let xs = collect_many(0x00c0_ffee_dead_beef, 10_000, |s| {
        sample_clutter_amplitude(&dist, s)
    });
    let k = kurtosis(&xs);
    assert!(
        (3.5..=7.0).contains(&k),
        "SeaSpray_SmallWhitecaps (K nu=3) sample kurtosis {} should be in 3.5..7.0",
        k
    );
}

#[test]
fn sea_spray_regimes_have_terrain_association() {
    // Every sea-spray regime must declare TerrainClass::Sea or the
    // new TerrainClass::CoastalSea — silently filing them under e.g.
    // Forest would defeat the point of the taxonomy.
    let lib = ClutterRegime::library();
    let sprays: Vec<&ClutterRegime> = lib
        .iter()
        .filter(|r| r.name().contains("SeaSpray"))
        .collect();
    assert!(
        !sprays.is_empty(),
        "library must contain sea-spray regimes by name"
    );
    for r in sprays {
        assert!(
            matches!(r.terrain, TerrainClass::Sea | TerrainClass::CoastalSea),
            "sea-spray regime {} must associate with Sea or CoastalSea, got {:?}",
            r.name(),
            r.terrain
        );
    }
}

#[test]
fn sea_spray_breaking_waves_more_spiky_than_open_sea() {
    // Sanity gate — breaking-wave kurtosis must exceed the open-sea
    // (nu ~ 8) baseline. Without this, a future tweak to the K nu
    // parameter on either side could silently equalise the two.
    let lib = ClutterRegime::library();
    let open = lib
        .iter()
        .find(|r| r.name() == "Sea_OpenMediumState")
        .expect("Sea_OpenMediumState must be in library");
    let breaking = lib
        .iter()
        .find(|r| r.name() == "SeaSpray_BreakingWaves")
        .expect("SeaSpray_BreakingWaves must be in library");

    let open_xs = collect_many(0x1111_2222_3333_4444, 8_000, |s| {
        sample_clutter_amplitude(&open.distribution, s)
    });
    let break_xs = collect_many(0x5555_6666_7777_8888, 8_000, |s| {
        sample_clutter_amplitude(&breaking.distribution, s)
    });
    let k_open = kurtosis(&open_xs);
    let k_break = kurtosis(&break_xs);
    assert!(
        k_break > k_open,
        "breaking-wave kurtosis ({}) must exceed open-sea kurtosis ({})",
        k_break,
        k_open,
    );
}
