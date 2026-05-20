    use super::*;
    use std::f32::consts::PI as PI32;

    fn channels_from_plane_wave(
        n_channels: usize,
        n_samples: usize,
        spacing_m: f64,
        freq_hz: f64,
        az_deg: f64,
    ) -> Vec<Vec<ComplexSample>> {
        let lambda = C_M_PER_S / freq_hz;
        let k = 2.0 * std::f64::consts::PI / lambda;
        let sin_theta = az_deg.to_radians().sin();
        (0..n_channels)
            .map(|ch| {
                let pos = (ch as f64) - (n_channels as f64 - 1.0) * 0.5;
                let phase = (k * spacing_m * pos * sin_theta) as f32;
                let phasor = ComplexSample::new(phase.cos(), phase.sin());
                vec![phasor; n_samples]
            })
            .collect()
    }

    #[test]
    fn sum_beamformer_handles_empty_input() {
        let bf = SumBeamformer::new();
        let out = bf.beamform(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn sum_beamformer_adds_channels_sample_wise() {
        let channels = vec![
            vec![
                ComplexSample::new(1.0, 0.0),
                ComplexSample::new(2.0, 1.0),
                ComplexSample::new(0.0, -1.0),
            ],
            vec![
                ComplexSample::new(0.0, 1.0),
                ComplexSample::new(-1.0, 0.0),
                ComplexSample::new(3.0, 4.0),
            ],
            vec![
                ComplexSample::new(0.5, 0.5),
                ComplexSample::new(0.0, 0.0),
                ComplexSample::new(-2.0, -3.0),
            ],
        ];
        let bf = SumBeamformer::new();
        let out = bf.beamform(&channels);
        assert_eq!(out.len(), 3);
        assert!((out[0].re - 1.5).abs() < 1e-6);
        assert!((out[0].im - 1.5).abs() < 1e-6);
        assert!((out[1].re - 1.0).abs() < 1e-6);
        assert!((out[1].im - 1.0).abs() < 1e-6);
        assert!((out[2].re - 1.0).abs() < 1e-6);
        assert!((out[2].im - 0.0).abs() < 1e-6);
    }

    #[test]
    fn delay_and_sum_passes_on_steering_signal() {
        let n_channels = 8;
        let n_samples = 4;
        let spacing = 0.015;
        let freq = 10_000_000_000.0;
        let target = 25.0;

        let on_channels = channels_from_plane_wave(n_channels, n_samples, spacing, freq, target);
        let bf = DelayAndSumBeamformer::for_ula(n_channels, spacing, freq, target);
        let out = bf.beamform(&on_channels);
        let on_mag: f32 = out.iter().map(|c| c.norm()).sum::<f32>() / n_samples as f32;
        assert!(on_mag > 0.95, "on-steering magnitude too low: {on_mag}");
    }

    #[test]
    fn delay_and_sum_rejects_off_steering_signal() {
        let n_channels = 16;
        let n_samples = 4;
        let spacing = 0.015;
        let freq = 10_000_000_000.0;
        let target = 0.0;
        let interferer_az = 30.0;

        let on_channels = channels_from_plane_wave(n_channels, n_samples, spacing, freq, target);
        let off_channels =
            channels_from_plane_wave(n_channels, n_samples, spacing, freq, interferer_az);
        let bf = DelayAndSumBeamformer::for_ula(n_channels, spacing, freq, target);
        let on_out = bf.beamform(&on_channels);
        let off_out = bf.beamform(&off_channels);

        let on_mag: f32 = on_out.iter().map(|c| c.norm()).sum::<f32>() / n_samples as f32;
        let off_mag: f32 = off_out.iter().map(|c| c.norm()).sum::<f32>() / n_samples as f32;
        assert!(
            on_mag > off_mag * 4.0,
            "on={on_mag} should dominate off={off_mag}"
        );
    }

    #[test]
    fn delay_and_sum_empty_input_returns_empty() {
        let bf = DelayAndSumBeamformer::new(vec![Complex::new(1.0, 0.0); 4]);
        let out = bf.beamform(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn capon_unimplemented_applies_supplied_weights() {
        let weights = vec![
            Complex::new(0.5, 0.0),
            Complex::new(0.0, 0.5),
            Complex::new(-0.5, 0.0),
        ];
        let channels = vec![
            vec![ComplexSample::new(1.0, 0.0)],
            vec![ComplexSample::new(1.0, 0.0)],
            vec![ComplexSample::new(1.0, 0.0)],
        ];
        let bf = CaponUnimplementedBeamformer::new(weights);
        let out = bf.beamform(&channels);
        assert_eq!(out.len(), 1);
        // (1+0j)*0.5 + (1+0j)*0.5j + (1+0j)*(-0.5) = 0.0 + 0.5j
        assert!(out[0].re.abs() < 1e-6, "re={}", out[0].re);
        assert!((out[0].im - 0.5).abs() < 1e-6, "im={}", out[0].im);
    }

    #[test]
    fn capon_unimplemented_handles_empty_input() {
        let bf = CaponUnimplementedBeamformer::new(vec![Complex::new(1.0, 0.0); 2]);
        assert!(bf.beamform(&[]).is_empty());
    }

    #[test]
    fn steering_vector_has_unit_magnitudes() {
        let sv = steering_vector(12, 0.015, 10_000_000_000.0, 17.0);
        assert_eq!(sv.len(), 12);
        for entry in &sv {
            let mag = (entry.re * entry.re + entry.im * entry.im).sqrt();
            assert!((mag - 1.0).abs() < 1e-9, "magnitude {mag} not unit");
        }
    }

    #[test]
    fn steering_vector_zero_az_is_all_ones() {
        let sv = steering_vector(8, 0.015, 10_000_000_000.0, 0.0);
        for entry in &sv {
            assert!((entry.re - 1.0).abs() < 1e-9);
            assert!(entry.im.abs() < 1e-9);
        }
    }

    #[test]
    fn steering_vector_phase_progression_is_linear() {
        let sv = steering_vector(4, 0.015, 10_000_000_000.0, 12.0);
        // Phases should be in arithmetic progression — successive deltas
        // are constant for a ULA. Compare deltas (modulo wrap) within a
        // tight tolerance.
        let phases: Vec<f32> = sv
            .iter()
            .map(|c| (c.im as f32).atan2(c.re as f32))
            .collect();
        let mut deltas: Vec<f32> = phases.windows(2).map(|w| w[1] - w[0]).collect();
        // Unwrap deltas around ±π so wrap-arounds don't fool the diff.
        for d in &mut deltas {
            while *d > PI32 {
                *d -= 2.0 * PI32;
            }
            while *d < -PI32 {
                *d += 2.0 * PI32;
            }
        }
        let reference = deltas[0];
        for d in &deltas {
            assert!(
                (d - reference).abs() < 1e-3,
                "delta {d} differs from reference {reference}"
            );
        }
    }

    #[test]
    fn steering_vector_single_element_returns_unit() {
        let sv = steering_vector(1, 0.015, 10_000_000_000.0, 45.0);
        assert_eq!(sv.len(), 1);
        assert!((sv[0].re - 1.0).abs() < 1e-9);
        assert!(sv[0].im.abs() < 1e-9);
    }
