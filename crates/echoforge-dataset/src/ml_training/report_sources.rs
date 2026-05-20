//! External calibration source definitions — extracted from report.rs for LOC compliance.

use crate::ml_training::types::ExternalCalibrationSource;

pub(super) fn external_calibration_sources() -> Vec<ExternalCalibrationSource> {
    vec![
        ExternalCalibrationSource {
            id: "scientific-data-2026-drone-radar-rf".to_string(),
            title: "Time-synchronized multi-sensor drone radar/RF dataset".to_string(),
            url: "https://www.nature.com/articles/s41597-026-06802-6".to_string(),
            role: "calibration_or_evaluation_metadata_only".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes:
                "Do not copy source data into EchoForge output unless the local operator verifies dataset terms."
                    .to_string(),
            expected_feature_mappings: vec![
                "range_doppler_proxy".to_string(),
                "doppler_spectrum".to_string(),
                "power_spectral_density".to_string(),
            ],
        },
        ExternalCalibrationSource {
            id: "rdrd-rad-dar-public-drone-radar".to_string(),
            title: "RDRD/RAD-DAR public drone radar dataset metadata (pending)".to_string(),
            url: "local-path-config-required".to_string(),
            role: "optional_local_calibration_mapping".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes:
                "Metadata hook only; configure a local path after source and license review.".to_string(),
            expected_feature_mappings: vec![
                "range_doppler_map".to_string(),
                "micro_doppler_spectrum".to_string(),
            ],
        },
        ExternalCalibrationSource {
            id: "rahman-robertson-drone-bird-micro-doppler".to_string(),
            title: "Radar micro-Doppler signatures of drones and birds".to_string(),
            url: "https://research-repository.st-andrews.ac.uk/handle/10023/16577".to_string(),
            role: "micro_doppler_format_motivation".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes: "Publication metadata only; no measured traces are vendored.".to_string(),
            expected_feature_mappings: vec![
                "propeller_or_wingbeat_peak_hz_proxy".to_string(),
                "micro_doppler_bandwidth_hz_proxy".to_string(),
            ],
        },
        ExternalCalibrationSource {
            id: "eusipco-2020-micro-doppler-representations".to_string(),
            title: "Comparison of micro-Doppler signal representations".to_string(),
            url: "https://eurasip.org/Proceedings/Eusipco/Eusipco2020/pdfs/0001561.pdf"
                .to_string(),
            role: "representation_family_motivation".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes: "Publication metadata only; no paper data are vendored.".to_string(),
            expected_feature_mappings: vec![
                "stft_spectrogram".to_string(),
                "weighted_spectrum".to_string(),
                "cepstrum".to_string(),
                "cadence_velocity".to_string(),
            ],
        },
        ExternalCalibrationSource {
            id: "low-grazing-uav-detection-cfar-micro-doppler".to_string(),
            title: "Low-grazing UAV detection literature on CFAR, clutter, and trajectory extraction"
                .to_string(),
            url: "https://arxiv.org/abs/1902.05483".to_string(),
            role: "cfar_tbd_label_motivation".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes: "Publication metadata only.".to_string(),
            expected_feature_mappings: vec![
                "cfar_statistic".to_string(),
                "tbd_track_score".to_string(),
                "clutter_pressure".to_string(),
            ],
        },
        ExternalCalibrationSource {
            id: "learned-radar-representation-2301-02451".to_string(),
            title: "Learned radar representations and data-driven detector reference".to_string(),
            url: "https://arxiv.org/abs/2301.02451".to_string(),
            role: "low_level_tensor_retention_motivation".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes: "Publication metadata only.".to_string(),
            expected_feature_mappings: vec![
                "iq_complex".to_string(),
                "range_time".to_string(),
                "learned_windows".to_string(),
            ],
        },
        ExternalCalibrationSource {
            id: "learned-radar-representation-2402-12970".to_string(),
            title: "Data-driven radar detector reference".to_string(),
            url: "https://arxiv.org/abs/2402.12970".to_string(),
            role: "low_level_tensor_retention_motivation".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes: "Publication metadata only.".to_string(),
            expected_feature_mappings: vec![
                "range_doppler_time".to_string(),
                "normalized_windows".to_string(),
            ],
        },
    ]
}
