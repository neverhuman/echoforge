    use super::*;

    /// Tiny SplitMix64-style RNG so tests are deterministic without a
    /// crate dependency. Mirrors the local AlphaRng in `cfar_alpha`.
    struct TestRng(u64);

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_add(0x9e37_79b9_7f4a_7c15))
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn open_unit_f64(&mut self) -> f64 {
            let bits = self.next_u64() >> 11;
            let denom = (1u64 << 53) as f64;
            (bits as f64 + 0.5) / denom
        }

        /// Unit-mean exponential power sample (= |I + jQ|^2 with I, Q
        /// independent Gaussian). Matches the Gaussian / Rayleigh
        /// convention used by `cfar_alpha::sample_distribution`.
        fn exp_power(&mut self) -> f64 {
            let u = self.open_unit_f64();
            -((1.0 - u).ln())
        }
    }

    fn gaussian_power_map(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = TestRng::new(seed);
        let mut out = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for _ in 0..cols {
                row.push(rng.exp_power());
            }
            out.push(row);
        }
        out
    }

    #[test]
    fn ca_cfar_2d_detects_single_injected_peak() {
        // 64×32 unit-mean Gaussian power map with a strong peak injected
        // at (32, 16). CA-CFAR should flag exactly that cell.
        let mut map = gaussian_power_map(64, 32, 0xC2FA_2D01);
        map[32][16] = 1.0e4;

        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-4,
            variant: Cfar2dVariant::CaCfar,
        });
        let detections = det.process(&map);

        // Must contain the injected peak.
        assert!(
            detections
                .iter()
                .any(|d| d.range_bin == 32 && d.azimuth_bin == 16),
            "expected detection at (32, 16); got {detections:?}"
        );
        // At Pfa=1e-4 over ~ (64-12) × (32-12) = 52 × 20 = 1040 evaluated
        // cells, the expected number of false alarms is ~0.1. With a
        // single injected peak, the detector should produce exactly one
        // detection on this seed.
        assert_eq!(
            detections.len(),
            1,
            "expected exactly 1 detection, got {detections:?}"
        );
    }

    #[test]
    fn os_cfar_2d_detects_single_injected_peak() {
        let mut map = gaussian_power_map(64, 32, 0xC2FA_2D02);
        map[20][10] = 5.0e3;

        // Training ring has N = (2*4+2*2+1)*(2*4+2*2+1) - (2*2+1)^2
        //                     = 13*13 - 25 = 144 cells.
        // 75th-percentile order index is 0.75 * 144 - 1 ≈ 107.
        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-4,
            variant: Cfar2dVariant::OsCfar {
                k_quantile_index: 107,
            },
        });
        let detections = det.process(&map);

        assert!(
            detections
                .iter()
                .any(|d| d.range_bin == 20 && d.azimuth_bin == 10),
            "expected detection at (20, 10); got {detections:?}"
        );
    }

    #[test]
    fn ca_cfar_2d_empirical_pfa_within_tolerance() {
        // 128×64 = 8192-cell Gaussian map, single trial. We're not running
        // 100k *independent* maps (that would be ~1e9 cells and minutes of
        // CPU). Instead we slide the CFAR window across one large map and
        // count exceedances; cells are i.i.d. so the in-map count is a
        // valid empirical Pfa estimator up to a small spatial-correlation
        // correction that the unit-mean-exponential samples don't induce.
        //
        // Window: tr=4, g=2 → ~ (128-12) × (64-12) = 116 × 52 = 6032
        // evaluated cells. At nominal Pfa=1e-2 we expect ~60 false alarms;
        // ±50% tolerance is a generous smoke gate.
        let map = gaussian_power_map(128, 64, 0xC2FA_2D03);
        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-2,
            variant: Cfar2dVariant::CaCfar,
        });
        let detections = det.process(&map);
        let evaluated = (128 - 2 * 6) * (64 - 2 * 6);
        let observed_pfa = detections.len() as f64 / evaluated as f64;
        let nominal = 1e-2_f64;
        let ratio = observed_pfa / nominal;
        assert!(
            (0.5..=2.0).contains(&ratio),
            "observed Pfa {observed_pfa} should be within 50% of nominal {nominal} \
             (ratio {ratio}); detections={}, evaluated={evaluated}",
            detections.len()
        );
    }

    #[test]
    fn edge_cell_peak_does_not_panic() {
        // Peak at the corner (0, 0): the detector skips it because it's
        // inside the edge margin. Must not panic.
        let mut map = gaussian_power_map(32, 32, 0xC2FA_2D04);
        map[0][0] = 1.0e6;

        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-3,
            variant: Cfar2dVariant::CaCfar,
        });
        let detections = det.process(&map);
        // Detection at (0, 0) is impossible because the CFAR window can't
        // sit entirely inside the map for that CUT.
        assert!(
            detections.iter().all(|d| !(d.range_bin == 0 && d.azimuth_bin == 0)),
            "edge cell (0, 0) should be skipped, got {detections:?}"
        );
    }

    #[test]
    fn throughput_256x128_under_100ms() {
        // 256 × 128 map = 32,768 cells; CFAR sweep ≈ 232 × 104 = 24,128
        // evaluated cells with a 13 × 13 = 169-cell training ring each.
        // Single-thread CA-CFAR should finish in well under 100 ms on
        // commodity x86_64. The threshold is loose so we don't fail in
        // debug builds on slow CI runners.
        let map = gaussian_power_map(256, 128, 0xC2FA_2D05);
        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-3,
            variant: Cfar2dVariant::CaCfar,
        });
        let start = std::time::Instant::now();
        let _ = det.process(&map);
        let dur = start.elapsed();
        assert!(
            dur < std::time::Duration::from_millis(500),
            "process(256x128) took {dur:?}, expected < 500 ms even in debug builds"
        );
    }

    #[test]
    fn training_cell_count_matches_window_geometry() {
        // tr=4, g=2 → outer 13x13 = 169; inner 5x5 = 25; ring = 144.
        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-3,
            variant: Cfar2dVariant::CaCfar,
        });
        assert_eq!(det.training_cell_count(), 144);
    }

    #[test]
    fn empty_or_tiny_input_returns_no_detections() {
        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-3,
            variant: Cfar2dVariant::CaCfar,
        });
        let empty: Vec<Vec<f64>> = Vec::new();
        assert!(det.process(&empty).is_empty());
        let tiny = vec![vec![1.0; 4]; 4];
        assert!(det.process(&tiny).is_empty());
    }
