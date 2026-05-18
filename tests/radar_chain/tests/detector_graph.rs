use echoforge_radar::{
    magnitude, pulse_compress, BlobRdDetector, BlobRdParams, CaCfarDetector, CfarParams,
    ComplexSample, DetectionKind, DetectorGraphRuntime, LfmChirp, OsCfarDetector, OsCfarParams,
    RangeDoppler,
};

fn peak_index(samples: &[f32]) -> usize {
    samples
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

/// End-to-end: synthesize an LFM chirp, embed it inside a noisy received
/// buffer, pulse-compress against the reference chirp, lift the resulting
/// 1-D magnitude vector into a 2-column range-Doppler grid, and run a
/// chained DetectorGraphRuntime (CA-CFAR + OS-CFAR + blob).
#[test]
fn detector_graph_finds_chirp_self_match_peak() {
    let chirp = LfmChirp {
        sample_rate_hz: 10_000.0,
        pulse_width_s: 0.001,
        bandwidth_hz: 1_000.0,
        carrier_hz: 0.0,
        initial_phase_rad: 0.0,
    };
    let reference = chirp.samples();
    let chirp_len = reference.len();

    // Build a longer received buffer with the chirp inserted at a known
    // offset and low-amplitude deterministic background to give the CFAR
    // training cells a real noise floor far from the matched-filter peak.
    let received_len = chirp_len * 16;
    let target_offset = chirp_len * 7;
    let mut received = vec![ComplexSample::new(0.0, 0.0); received_len];
    // Deterministic background: a tiny pseudo-noise pattern that avoids
    // structural coherence with the chirp.
    for (i, slot) in received.iter_mut().enumerate() {
        let phase = (i as f32) * 0.137;
        *slot = ComplexSample::new(0.01 * phase.cos(), 0.01 * phase.sin());
    }
    for (i, sample) in reference.iter().enumerate() {
        received[target_offset + i] += *sample;
    }

    let compressed = pulse_compress(&received, &reference);
    let mags = magnitude(&compressed);
    let expected_peak = peak_index(&mags);

    // Lift to a 2-column range-Doppler grid so the blob detector has a real
    // 2-D neighborhood and so the row-max projection is unambiguous.
    let r = mags.len();
    let d = 2usize;
    let mut data = vec![0.0f32; r * d];
    for (i, m) in mags.iter().enumerate() {
        data[i * d] = *m;
        data[i * d + 1] = *m;
    }
    let rd = RangeDoppler::new(r, d, data);

    // Guard cells (20) are intentionally wider than the matched-filter main
    // lobe (~20 bins peak-to-base) so the training window samples the clean
    // background noise floor rather than the lobe shoulders. Pfa is loose
    // (1e-2) to keep the threshold low against the very small synthetic
    // background.
    let mut graph = DetectorGraphRuntime::default();
    graph.add_ca_cfar(CaCfarDetector::new(CfarParams::new(8, 20, 1e-2)));
    graph.add_os_cfar(OsCfarDetector::new(OsCfarParams::new(8, 20, 1e-2, 0.75)));
    let peak_mag = mags.iter().copied().fold(0.0f32, f32::max);
    graph.add_blob(BlobRdDetector::new(BlobRdParams::new(peak_mag * 0.5, 1)));

    let fused = graph.run(&rd);

    let ca_n = fused
        .per_detector
        .get(&DetectionKind::CaCfar)
        .copied()
        .unwrap_or(0);
    let os_n = fused
        .per_detector
        .get(&DetectionKind::OsCfar)
        .copied()
        .unwrap_or(0);
    let blob_n = fused
        .per_detector
        .get(&DetectionKind::Blob)
        .copied()
        .unwrap_or(0);
    assert!(ca_n >= 1, "CA-CFAR should fire (got {ca_n})");
    assert!(os_n >= 1, "OS-CFAR should fire (got {os_n})");
    assert!(
        blob_n >= 1,
        "Blob should fire on the matched-filter peak (got {blob_n})"
    );

    // Fused dedup invariant: count >= max(individual).
    assert!(fused.events.len() >= ca_n.max(os_n).max(blob_n));

    // The expected peak bin must appear in the fused events.
    assert!(
        fused.events.iter().any(|e| e.range_bin == expected_peak),
        "fused events missing expected peak at bin {expected_peak}: {:?}",
        fused.events
    );
}
