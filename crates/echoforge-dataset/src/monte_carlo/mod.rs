pub mod config;
pub mod error;
pub mod scene_config;

mod helpers;
mod run;
mod sampling;

pub use config::{
    MonteCarloBenchmarkReport, MonteCarloDemoConfig, MonteCarloDemoReport, MonteCarloRuntimePolicy,
    StageTiming, DEFAULT_PRESET, DEFAULT_TARGET_LABEL,
};
pub use error::DatasetError;
pub use run::run_monte_carlo_demo;
pub use scene_config::{
    embedded_airspace_config, known_presets, AirspaceMonteCarloConfig, BehaviorConfig,
    ContestedAirspaceConfig, DimensionsConfig, EnvironmentProfileConfig, KinematicsConfig,
    MicroMotionConfig, ObjectClassConfig, PresetConfig, RfiConfig, SensorArchetypeConfig,
    SensorObservableConfig, WeatherConfig,
};

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use echoforge_core::models::{DatasetCard, RadarEpisode};

    use super::*;

    #[test]
    fn embedded_config_contains_many_airspace_objects() {
        let config = embedded_airspace_config().expect("config parses");
        assert!(config.object_classes.len() >= 6);
        assert!(config
            .environment_profiles
            .iter()
            .any(|profile| profile.contested_airspace.cochannel_emitters[1] >= 8));
        assert!(config
            .guardrails
            .iter()
            .any(|item| item == "public-proxy assumptions only"));
    }

    #[test]
    fn tempdir_generation_writes_required_files_and_models_validate() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let output = tmp_dir.path().join("demo");
        let mut config = MonteCarloDemoConfig::low_altitude_fixed_wing_default(output.clone());
        config.episodes = 3;
        config.pulse_count = 8;
        config.generated_at = "2026-05-18T00:00:00Z".to_string();

        let report = run_monte_carlo_demo(config).expect("run demo");
        assert_eq!(report.episode_count, 3);
        assert!(report.leakage_clean);
        assert!(output.join("dataset_card.json").exists());
        assert!(output.join("manifest.json").exists());
        assert!(output
            .join("episodes/episode_000001/products/iq_complex.zarr/.zarray")
            .exists());
        assert!(output
            .join("episodes/episode_000001/products/link_diagnostics.json")
            .exists());

        let dataset_card: DatasetCard =
            serde_json::from_slice(&fs::read(output.join("dataset_card.json")).unwrap()).unwrap();
        dataset_card.validate().expect("dataset card validates");
        for index in 1..=3 {
            let episode_path = output
                .join("episodes")
                .join(format!("episode_{index:06}"))
                .join("radar_episode.json");
            let episode: RadarEpisode =
                serde_json::from_slice(&fs::read(episode_path).unwrap()).unwrap();
            episode.validate().expect("radar episode validates");
        }
    }

    #[test]
    fn same_seed_and_timestamp_keep_manifest_stable() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let mut a = MonteCarloDemoConfig::low_altitude_fixed_wing_default(tmp_dir.path().join("a"));
        a.episodes = 2;
        a.pulse_count = 6;
        let mut b = a.clone();
        b.output_dir = tmp_dir.path().join("b");

        run_monte_carlo_demo(a.clone()).expect("first run");
        run_monte_carlo_demo(b.clone()).expect("second run");

        let manifest_a = fs::read_to_string(a.output_dir.join("manifest.json")).unwrap();
        let manifest_b = fs::read_to_string(b.output_dir.join("manifest.json")).unwrap();
        assert_eq!(manifest_a, manifest_b);
    }

    #[test]
    fn refuses_existing_or_source_owned_output() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let existing = tmp_dir.path().join("existing");
        fs::create_dir_all(&existing).unwrap();
        let mut config = MonteCarloDemoConfig::low_altitude_fixed_wing_default(existing);
        config.episodes = 1;
        assert!(matches!(
            run_monte_carlo_demo(config),
            Err(DatasetError::RefusingOutputPath(_))
        ));

        let mut bad =
            MonteCarloDemoConfig::low_altitude_fixed_wing_default(PathBuf::from("crates/demo"));
        bad.episodes = 1;
        assert!(matches!(
            run_monte_carlo_demo(bad),
            Err(DatasetError::RefusingOutputPath(_))
        ));
    }

    #[test]
    fn known_presets_include_diversified_roster() {
        let presets = known_presets().expect("presets parse");
        let expected = [
            "low-altitude-fixed-wing-takeoff-v1",
            "mixed-low-altitude-hard-negatives-v1",
            "multirotor-hover-rural-v1",
            "birds-and-balloons-v1",
            "infrastructure-glint-clutter-only-v1",
        ];
        for id in expected {
            assert!(
                presets.iter().any(|preset| preset == id),
                "preset {id} missing from known_presets list: {presets:?}"
            );
        }
        assert!(
            !presets.iter().any(|preset| preset.contains("iranian")),
            "no preset id should mention iranian: {presets:?}"
        );
    }

    #[test]
    fn embedded_config_object_classes_are_neutrally_named() {
        let config = embedded_airspace_config().expect("config parses");
        for class in &config.object_classes {
            assert!(
                !class.id.to_ascii_lowercase().contains("iranian"),
                "object class id should be neutral: {}",
                class.id
            );
            assert!(
                !class.display_name.to_ascii_lowercase().contains("iranian"),
                "object class display_name should be neutral: {}",
                class.display_name
            );
        }
    }

    #[test]
    fn superseded_iranian_takeoff_default_still_resolves_to_neutral_preset() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let cfg = MonteCarloDemoConfig::iranian_takeoff_default(tmp_dir.path().join("prior"));
        assert_eq!(cfg.preset, DEFAULT_PRESET);
        assert_eq!(cfg.target_label, DEFAULT_TARGET_LABEL);
    }

    #[test]
    fn additional_presets_run_end_to_end() {
        for preset_id in [
            "multirotor-hover-rural-v1",
            "birds-and-balloons-v1",
            "infrastructure-glint-clutter-only-v1",
        ] {
            let tmp_dir = tempfile::tempdir().expect("tempdir");
            let mut cfg = MonteCarloDemoConfig::low_altitude_fixed_wing_default(
                tmp_dir.path().join(preset_id),
            );
            cfg.preset = preset_id.to_string();
            cfg.episodes = 2;
            cfg.pulse_count = 6;
            let report = run_monte_carlo_demo(cfg).expect("preset should resolve and run");
            assert_eq!(report.episode_count, 2, "{preset_id}");
            assert!(report.leakage_clean, "{preset_id}");
        }
    }
}
