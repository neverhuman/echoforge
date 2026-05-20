    use super::super::kinematic_gate::KinematicSample;
    use super::*;

    fn cruise_window(samples: Vec<(f64, f64, f64)>) -> KinematicObservation {
        KinematicObservation::new(
            samples
                .into_iter()
                .map(|(t, v, h)| KinematicSample::new(t, v, h))
                .collect(),
            12_000.0,
            20.0,
        )
    }

    fn steady_cruise_window(speed: f64, alt: f64, n: usize) -> KinematicObservation {
        let samples: Vec<(f64, f64, f64)> = (0..n)
            .map(|k| (k as f64, speed, alt))
            .collect();
        cruise_window(samples)
    }

    #[test]
    fn cruise_detector_accepts_piston_50mps_steady() {
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(50.0, 800.0, 8);
        let dec = detector.evaluate(&obs, None, None);
        assert_eq!(dec.propulsion_class, PropulsionClass::Piston);
        assert!(dec.detected, "steady piston cruise must detect; {}", dec.note);
    }

    #[test]
    fn cruise_detector_accepts_jet_120mps_steady() {
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(120.0, 800.0, 8);
        let dec = detector.evaluate(&obs, None, None);
        assert_eq!(dec.propulsion_class, PropulsionClass::Jet);
        assert!(dec.detected, "steady jet cruise must detect; {}", dec.note);
    }

    #[test]
    fn cruise_detector_rejects_ambiguous_75mps() {
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(75.0, 800.0, 8);
        let dec = detector.evaluate(&obs, None, None);
        assert_eq!(dec.propulsion_class, PropulsionClass::Ambiguous);
        assert!(!dec.detected, "ambiguous cluster must NOT detect");
    }

    #[test]
    fn cruise_detector_rejects_bird_15mps() {
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(15.0, 800.0, 8);
        let dec = detector.evaluate(&obs, None, None);
        assert_eq!(dec.propulsion_class, PropulsionClass::BirdLike);
        assert!(!dec.detected);
    }

    #[test]
    fn cruise_detector_os_cfar_threshold_check_blocks_noise_floor() {
        // Provide a flat-noise spectrum; OS-CFAR threshold must reject
        // (peak ~= noise estimate × ~1 << alpha).
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(50.0, 800.0, 8);
        let spec = vec![1.0f32; 128];
        let dec = detector.evaluate(&obs, Some(&spec), Some(5.0));
        assert!(!dec.detected, "flat noise must not pass OS-CFAR");
        assert!(!dec.cfar_passed);
    }

    #[test]
    fn cruise_detector_os_cfar_threshold_passes_strong_peak() {
        // Spike one bin to 1000 × the floor; CFAR must accept.
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(50.0, 800.0, 8);
        let mut spec = vec![1.0f32; 128];
        spec[64] = 1000.0;
        let dec = detector.evaluate(&obs, Some(&spec), Some(5.0));
        assert!(dec.cfar_passed, "strong peak must clear OS-CFAR");
        assert_eq!(dec.dominant_doppler_bin, Some(64));
    }

    #[test]
    fn cruise_detector_blade_pass_micro_doppler_piston() {
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(50.0, 800.0, 8);
        // 256 bins × 1 Hz = 256 Hz Nyquist. Spike bin 180 (180 Hz)
        // which is squarely in [127.5, 253] Hz blade-pass window.
        let mut spec = vec![1.0f32; 256];
        spec[64] = 1000.0; // body Doppler peak (CFAR target)
        spec[180] = 50.0; // blade-pass line
        let dec = detector.evaluate(&obs, Some(&spec), Some(1.0));
        assert!(
            dec.micro_doppler_confirmed,
            "blade-pass at 180 Hz must be confirmed for piston cluster"
        );
    }
