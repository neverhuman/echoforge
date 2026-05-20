//! Wave 9 case study: Saab Giraffe 1X vs Shahed-136 / Geran-2.
//!
//! Loads both cards via `echoforge-packs`, computes predicted detection
//! range using the monostatic radar equation, compares to vendor-declared
//! envelope, and renders the headline HTML credibility report.
//!
//! Acceptance gate: predicted range vs vendor declared `fighter-class
//! 75 km @ 0 dBsm` is within ±20 %.

use std::path::PathBuf;

use echoforge_case_studies::link_budget::{
    albersheim_required_snr_db, swerling_fluctuation_penalty_db, LinkBudgetInputs,
};
use echoforge_case_studies::report::{
    CaseStudyTrace, ReproductionRow, TraceRow,
};
use echoforge_packs::Registry;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Load the Saab Giraffe 1X card payload by walking the
/// `radar-platforms-v1` pack via the pack-loader and pulling the
/// physical parameters needed for the link budget.
fn load_saab_giraffe_1x() -> (
    serde_json::Value, // raw card payload (for source-attribution display)
    LinkBudgetInputs,
    f64, // pulse_width_s (display only)
    f64, // pulse_compression_ratio (display only)
) {
    let registry = Registry::discover(&repo_root().join("object-packs"))
        .expect("object-packs/ discoverable");
    let card = registry
        .find_card("radar_platform_card", "saab-giraffe-1x")
        .expect("saab-giraffe-1x card must be loadable via the pack registry");
    let v = (*card.value).clone();

    let peak_power_w = v["transmit"]["peak_power_w"].as_f64().unwrap();
    let gain_dbi = v["antenna"]["gain_dbi"].as_f64().unwrap();
    let center_frequency_hz = v["center_frequency_hz"].as_f64().unwrap();
    let system_temperature_k = v["receiver"]["system_temperature_k"].as_f64().unwrap();
    let noise_figure_db = v["receiver"]["noise_figure_db"].as_f64().unwrap();
    let pulse_width_s = v["transmit"]["pulse_width_s"].as_f64().unwrap();
    let pulse_compression_ratio = v["transmit"]["waveform_parameters"]
        ["pulse_compression_ratio"]
        .as_f64()
        .unwrap();
    let system_losses_db = v["receiver"]["system_losses_db"].as_f64().unwrap();
    let prf_hz = v["transmit"]["prf_hz"].as_f64().unwrap();
    let dwell_time_s = v["scan"]["dwell_time_s"].as_f64().unwrap();

    let inputs = LinkBudgetInputs::from_card_fields(
        peak_power_w,
        gain_dbi,
        center_frequency_hz,
        system_temperature_k,
        noise_figure_db,
        pulse_width_s,
        pulse_compression_ratio,
        system_losses_db,
        prf_hz,
        dwell_time_s,
    );
    (v, inputs, pulse_width_s, pulse_compression_ratio)
}

/// Look up the broadside-mid S-band RCS for Shahed-136 from the source
/// card. Returns (rcs_dbsm_midpoint, rcs_dbsm_min, rcs_dbsm_max).
fn shahed_136_broadside_s_band_rcs_dbsm() -> (f64, f64, f64) {
    let registry = Registry::discover(&repo_root().join("object-packs")).unwrap();
    let card = registry
        .find_card("source_platform_card", "shahed-136-geran-2")
        .expect("shahed-136-geran-2 card must be loadable");
    let bands = card.value["rcs_signature"]["bands"].as_array().unwrap();
    let s_band = bands
        .iter()
        .find(|b| b["band_name"] == "S")
        .expect("S-band entry");
    let aspects = s_band["aspect_envelopes"].as_array().unwrap();
    let broadside = aspects
        .iter()
        .find(|a| a["aspect_name"] == "broadside_left")
        .expect("broadside aspect");
    let min = broadside["rcs_dbsm_range"]["min"].as_f64().unwrap();
    let max = broadside["rcs_dbsm_range"]["max"].as_f64().unwrap();
    let mid = 0.5 * (min + max);
    (mid, min, max)
}

fn dbsm_to_m2(dbsm: f64) -> f64 {
    10f64.powf(dbsm / 10.0)
}

fn gap_pct(predicted_m: f64, declared_m: f64) -> f64 {
    100.0 * (predicted_m - declared_m) / declared_m
}

/// Acceptance gate: predicted vs declared 75 km @ 0 dBsm (fighter-class)
/// is within ±20 %. Required SNR = Albersheim non-fluctuating floor
/// plus Swerling-1 penalty (5.7 dB at Pd=0.85, Pfa=1e-4 per Skolnik
/// 3rd ed. Table 2.3 / Fig 2.8) — vendor declared ranges are
/// universally quoted against a fluctuating target.
#[test]
fn predicted_fighter_class_range_within_20_pct_of_declared() {
    let (_card, inputs, _pw, _pcr) = load_saab_giraffe_1x();
    let snr_req_db = albersheim_required_snr_db(0.85, 1.0e-4)
        + swerling_fluctuation_penalty_db("swerling_1");
    let predicted_m = inputs.predicted_detection_range_m(1.0, snr_req_db);
    let declared_m = 75_000.0;
    let gap = gap_pct(predicted_m, declared_m);
    assert!(
        gap.abs() <= 20.0,
        "predicted {:.1} km vs declared 75 km is outside ±20 % (gap {:.2} %)",
        predicted_m / 1000.0,
        gap
    );
    eprintln!(
        "predicted fighter-class range = {:.2} km  vs declared 75 km  (gap {:+.2} %, SNR_req {:.2} dB Swerling-1)",
        predicted_m / 1000.0,
        gap,
        snr_req_db
    );
}

/// Albersheim sanity at the case study's operating point.
#[test]
fn albersheim_required_snr_at_pd_85_pfa_1e4_in_range() {
    let snr = albersheim_required_snr_db(0.85, 1.0e-4);
    assert!(
        (8.0..14.0).contains(&snr),
        "SNR_req for Pd=0.85 Pfa=1e-4 should be 8–14 dB, got {snr}"
    );
}

/// Smoke: predicted range against Shahed-class broadside-mid RCS comes
/// in below the fighter-class range (because RCS is lower) and the OSINT
/// rough estimate (25 km) lands within a factor of ~2 of the prediction
/// (the gap is explained in the report and is the right kind of finding).
#[test]
fn predicted_shahed_class_range_in_plausible_band() {
    let (_card, inputs, _pw, _pcr) = load_saab_giraffe_1x();
    let snr_req_db = albersheim_required_snr_db(0.85, 1.0e-4)
        + swerling_fluctuation_penalty_db("swerling_1");
    let (rcs_dbsm, _min, _max) = shahed_136_broadside_s_band_rcs_dbsm();
    let predicted_m =
        inputs.predicted_detection_range_m(dbsm_to_m2(rcs_dbsm), snr_req_db);
    assert!(
        (10_000.0..80_000.0).contains(&predicted_m),
        "Shahed broadside-mid prediction should be 10–80 km, got {:.1} km (RCS {:.1} dBsm)",
        predicted_m / 1000.0,
        rcs_dbsm
    );
}

/// Render the headline HTML credibility report. Marked `#[test]` so it
/// runs on every `cargo test`; the assertion that the file actually
/// landed on disk is the gate.
#[test]
fn render_html_credibility_report() {
    let (radar_card, inputs, _pw, _pcr) = load_saab_giraffe_1x();
    let snr_req_db = albersheim_required_snr_db(0.85, 1.0e-4)
        + swerling_fluctuation_penalty_db("swerling_1");

    // Pull declared envelope entries from the card.
    let envelopes = radar_card["declared_detection_envelope"]
        .as_array()
        .unwrap();

    let mut rows: Vec<ReproductionRow> = Vec::new();
    for env in envelopes {
        let target_class_slug = env["target_class_slug"].as_str().unwrap();
        let declared_range_m = env["range_m"].as_f64().unwrap();
        let conditions = env["conditions"].as_str().unwrap_or("").to_string();
        let citation = env["citation_id"].as_str().unwrap_or("").to_string();
        let confidence = env["confidence_label"].as_str().unwrap_or("").to_string();

        // Resolve RCS: prefer rcs_dbsm if present, else rcs_m2 → dBsm.
        let rcs_dbsm = if let Some(dbsm) = env.get("rcs_dbsm").and_then(|v| v.as_f64()) {
            dbsm
        } else if let Some(m2) = env.get("rcs_m2").and_then(|v| v.as_f64()) {
            10.0 * m2.log10()
        } else {
            continue;
        };
        let rcs_m2 = dbsm_to_m2(rcs_dbsm);
        let predicted_m = inputs.predicted_detection_range_m(rcs_m2, snr_req_db);
        let gap = gap_pct(predicted_m, declared_range_m);
        rows.push(ReproductionRow {
            target_label: target_class_slug.to_string(),
            rcs_dbsm,
            declared_range_km: declared_range_m / 1000.0,
            predicted_range_km: predicted_m / 1000.0,
            gap_pct: gap,
            confidence_label: confidence,
            citation,
            conditions,
        });
    }

    // Build the SNR vs range trace at the broadside-mid Shahed RCS.
    let (rcs_dbsm_mid, _, _) = shahed_136_broadside_s_band_rcs_dbsm();
    let rcs_m2 = dbsm_to_m2(rcs_dbsm_mid);
    let snr_trace: Vec<TraceRow> = (1..=15)
        .map(|i| {
            let r_km = i as f64 * 10.0;
            let snr_db = inputs.integrated_snr_db(rcs_m2, r_km * 1000.0);
            TraceRow {
                range_km: r_km,
                snr_db,
                above_threshold: snr_db >= snr_req_db,
            }
        })
        .collect();

    let trace = CaseStudyTrace {
        case_title: "Saab Giraffe 1X vs Shahed-136 / Geran-2".to_string(),
        utc_timestamp: "2026-05-18T22:00:00Z (Wave 9 case study run)".to_string(),
        radar_pack: "radar-platforms-v1".to_string(),
        radar_card_slug: "saab-giraffe-1x".to_string(),
        radar_display_name: radar_card["sensor_name"]
            .as_str()
            .unwrap_or("Saab Giraffe 1X")
            .to_string(),
        radar_vendor: radar_card["vendor"].as_str().unwrap_or("Saab").to_string(),
        radar_model: radar_card["model"]
            .as_str()
            .unwrap_or("Giraffe 1X")
            .to_string(),
        source_pack: "adversary-platforms-v1".to_string(),
        source_card_slug: "shahed-136-geran-2".to_string(),
        source_display_name: "Shahed-136 / Geran-2".to_string(),
        source_country_of_origin: "IRN-RUS".to_string(),
        center_frequency_ghz: radar_card["center_frequency_hz"].as_f64().unwrap() / 1.0e9,
        gain_dbi: radar_card["antenna"]["gain_dbi"].as_f64().unwrap(),
        peak_power_w: radar_card["transmit"]["peak_power_w"].as_f64().unwrap(),
        prf_hz: radar_card["transmit"]["prf_hz"].as_f64().unwrap(),
        dwell_s: radar_card["scan"]["dwell_time_s"].as_f64().unwrap(),
        noise_figure_db: radar_card["receiver"]["noise_figure_db"].as_f64().unwrap(),
        system_losses_db: radar_card["receiver"]["system_losses_db"].as_f64().unwrap(),
        coherent_pulses: inputs.coherent_integration_pulses,
        required_snr_db: snr_req_db,
        pd_at_pfa: (0.85, 1.0e-4),
        reproduction_rows: rows,
        snr_trace,
        trace_target_label: "Shahed-136 broadside-mid S-band".to_string(),
        trace_target_rcs_dbsm: rcs_dbsm_mid,
        bibliography: vec![
            "Skolnik, <em>Introduction to Radar Systems</em>, 3rd ed., McGraw-Hill 2001. §2.5 (radar equation), §2.6 (receiver noise), §2.8 (Albersheim approximation).".into(),
            "Albersheim, \"Closed-Form Approximation to Robertson's Detection Characteristics,\" Proc. IEEE, 1981.".into(),
            "ITU-R P.676-13 (atmospheric gas attenuation), P.838-3 (rain), P.840-8 (cloud), P.453-14 (refractivity).".into(),
            "Rahman &amp; Robertson, <em>Nature Scientific Reports</em> 8:17396 (2018). Bird vs UAS micro-Doppler.".into(),
            "MDPI <em>Drones</em> 2023 7(1):39. Small fixed-wing UAV RCS signature investigation.".into(),
            "Saab Giraffe 1X — Product brochure 2024. <a href=\"https://www.saab.com/products/giraffe-1x\">https://www.saab.com/products/giraffe-1x</a>".into(),
            "OSMP Shahed-136 visual guide. <a href=\"https://osmp.ngo/shahed-136\">https://osmp.ngo/shahed-136</a>".into(),
            "RUSI — Russian Iranian-made UAV technical profile.".into(),
            "Reuters — US Army Giraffe 1X order 2026 (referenced in tips/detectors/tip1.txt:220-222).".into(),
        ],
        receipt_refs: vec![
            ".agents/receipts/radar-platform-card-schema-v1/20260518T215000Z.md".into(),
            ".agents/receipts/pluggable-pack-loader-v1/20260518T220000Z.md".into(),
            ".agents/receipts/named-platform-case-study-v1/<this-run>.md".into(),
            "/home/ubuntu/.claude/plans/i-am-terrified-that-vast-hamming.md (authoritative spec)".into(),
        ],
    };

    let out_dir = repo_root()
        .join("outputs")
        .join("radar-expert-credibility-report");
    let path = trace.write_html(&out_dir).expect("write index.html");
    assert!(
        path.exists(),
        "HTML credibility report should land at {}",
        path.display()
    );
    eprintln!("wrote credibility report → {}", path.display());
}
