use std::fs;

use echoforge_radar::{
    ca_cfar_1d, ca_cfar_scale, magnitude, pulse_compress, CfarParams, CpuBackend, LfmChirp,
    RadarChain,
};
use echoforge_sig::{pending_analytic_report, EchoSigArtifactBundle};

fn peak_index(samples: &[f32]) -> usize {
    samples
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

#[test]
fn echo_sig_bundle_round_trip_preserves_manifest_and_cards() {
    let workdir = tempfile::tempdir().expect("tempdir");
    let bundle = EchoSigArtifactBundle::pending("ef:artifact:demo:0001:0");
    let object_card = "kind: object\nname: demo-object\n".to_string();
    let material_card = "kind: material\nname: demo-material\n".to_string();
    let solver_card = "kind: solver\nname: demo-solver\n".to_string();
    let mut bundle = bundle;
    bundle.object_card = Some(echoforge_sig::BundleCard {
        kind: "object_card".to_string(),
        raw_yaml: object_card.clone(),
    });
    bundle.material_card = Some(echoforge_sig::BundleCard {
        kind: "material_card".to_string(),
        raw_yaml: material_card.clone(),
    });
    bundle.solver_card = Some(echoforge_sig::BundleCard {
        kind: "solver_card".to_string(),
        raw_yaml: solver_card.clone(),
    });

    bundle.write_to_dir(workdir.path()).expect("write");
    let round_trip = EchoSigArtifactBundle::read_from_dir(workdir.path()).expect("read");

    assert_eq!(round_trip.manifest.artifact_id, "ef:artifact:demo:0001:0");
    assert_eq!(round_trip.manifest.axes.len(), 11);
    assert_eq!(round_trip.provenance.seed, 0);
    assert_eq!(round_trip.license.expression, "Apache-2.0");
    assert_eq!(
        round_trip.object_card.as_ref().unwrap().raw_yaml,
        object_card
    );
    assert_eq!(
        round_trip.material_card.as_ref().unwrap().raw_yaml,
        material_card
    );
    assert_eq!(
        round_trip.solver_card.as_ref().unwrap().raw_yaml,
        solver_card
    );

    let manifest_path = workdir.path().join("manifest.json");
    assert!(manifest_path.exists());
    assert!(fs::metadata(manifest_path).expect("metadata").is_file());
}

#[test]
fn pending_analytic_report_covers_expected_primitives() {
    let report = pending_analytic_report();
    assert_eq!(report.cases.len(), 6);
    assert!(!report.passed);
    assert!(report.summary.contains("pending"));
}

#[test]
fn lfm_pulse_compression_peaks_at_self_match_lag() {
    let chirp = LfmChirp {
        sample_rate_hz: 20_000.0,
        pulse_width_s: 0.002,
        bandwidth_hz: 2_000.0,
        carrier_hz: 0.0,
        initial_phase_rad: 0.0,
    };
    let reference = chirp.samples();
    let compressed = pulse_compress(&reference, &reference);
    let magnitudes = magnitude(&compressed);
    let expected_peak = reference.len() - 1;
    assert_eq!(peak_index(&magnitudes), expected_peak);
}

#[test]
fn ca_cfar_flags_an_isolated_target_cell() {
    let mut power = vec![1.0f32; 64];
    power[31] = 100.0;
    let params = CfarParams::new(8, 2, 1e-3);
    let decisions = ca_cfar_1d(&power, params);
    let target = decisions
        .iter()
        .find(|decision| decision.index == 31)
        .unwrap();

    assert!(target.evaluated);
    assert!(target.detected);
    assert!(target.threshold > 0.0);
    assert!(ca_cfar_scale(8, 1e-3) > 0.0);
}

#[test]
fn radar_chain_can_run_end_to_end_on_cpu() {
    let backend = CpuBackend;
    let chain = RadarChain::new(backend, CfarParams::new(4, 2, 1e-3));
    let chirp = LfmChirp {
        sample_rate_hz: 10_000.0,
        pulse_width_s: 0.001,
        bandwidth_hz: 1_000.0,
        carrier_hz: 0.0,
        initial_phase_rad: 0.0,
    };
    let reference = chain.chirp(&chirp);
    let output = chain.detect(&reference, &reference);

    assert_eq!(output.reference.len(), reference.len());
    assert_eq!(output.compressed.len(), reference.len() * 2 - 1);
    assert_eq!(output.magnitudes.len(), output.compressed.len());
    assert_eq!(output.cfar.len(), output.compressed.len());
    assert_eq!(chain.backend_name(), "cpu");
}
