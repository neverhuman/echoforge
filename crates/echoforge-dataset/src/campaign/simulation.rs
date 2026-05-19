use std::collections::BTreeMap;

use echoforge_radar::{
    apply_clutter_to_profile, apply_receiver_impairments, apply_rfi_to_profile,
    sample_clutter_frame, sample_receiver_impairments, sample_rfi_frame, ClutterProfile,
    RfiProfile, ReceiverImpairmentProfile,
};

use super::detectors::{
    CfarTrackerBaseline, FeatureTreeClassifier, StreamingDetector, TemporalTinyModel,
};
use super::rng::SplitMix64;
use super::types::{
    CampaignBucket, CampaignClass, CampaignConfig, CampaignRecordPlan,
    DimensionsSample, FirstTriggerEvent, FrameFeature, FrameLabel, ModelPredictionRow,
};
use crate::monte_carlo::DatasetError;
use echoforge_radar::SyntheticEpisode;

// ── Class envelope ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(super) struct ClassEnvelope {
    pub dimensions_m: DimensionsSample,
    pub rcs_dbsm: f64,
    pub speed_mps: f64,
    pub initial_range_m: f64,
    pub heading_deg: f64,
    pub acceleration_mps2: f64,
    pub climb_rate_mps: f64,
    pub max_altitude_m: f64,
    pub radial_velocity_bias_mps: f64,
    pub attitude_jitter_deg: f64,
    pub propulsor_hz: f64,
    pub micro_doppler_hz: f64,
    pub base_snr_db: f32,
    pub clutter_profile: ClutterProfile,
    pub rfi_profile: RfiProfile,
    pub receiver_profile: ReceiverImpairmentProfile,
}

pub(super) fn class_envelope(class: &CampaignClass, rng: &mut SplitMix64) -> ClassEnvelope {
    let mut clutter = ClutterProfile::moderate_mixed();
    let mut rfi = RfiProfile::contested_low_altitude();
    let mut receiver = ReceiverImpairmentProfile::public_proxy_default();
    let (dimensions, rcs, speed, snr, micro, prop, altitude, range, radial) = match class.bucket {
        CampaignBucket::PositiveOwaDeltaPusher => (
            DimensionsSample {
                length: rng.range_f64(3.3, 3.7),
                wingspan: rng.range_f64(2.3, 2.7),
                height: rng.range_f64(0.35, 0.75),
            },
            rng.range_f64(-16.0, -2.0),
            rng.range_f64(45.0, 60.0),
            rng.range_f32(7.0, 14.0),
            rng.range_f64(28.0, 95.0),
            rng.range_f64(75.0, 145.0),
            rng.range_f64(80.0, 450.0),
            rng.range_f64(4_800.0, 8_500.0),
            rng.range_f64(-55.0, -24.0),
        ),
        CampaignBucket::SmallUav => (
            DimensionsSample {
                length: rng.range_f64(0.35, 2.4),
                wingspan: rng.range_f64(0.35, 4.0),
                height: rng.range_f64(0.08, 0.7),
            },
            rng.range_f64(-32.0, -8.0),
            rng.range_f64(0.0, 32.0),
            rng.range_f32(0.0, 11.0),
            rng.range_f64(40.0, 240.0),
            rng.range_f64(0.0, 250.0),
            rng.range_f64(10.0, 260.0),
            rng.range_f64(700.0, 4_500.0),
            rng.range_f64(-22.0, 22.0),
        ),
        CampaignBucket::Biological => (
            DimensionsSample {
                length: rng.range_f64(0.03, 1.2),
                wingspan: rng.range_f64(0.04, 2.4),
                height: rng.range_f64(0.01, 0.45),
            },
            rng.range_f64(-48.0, -14.0),
            rng.range_f64(1.0, 24.0),
            rng.range_f32(-4.0, 9.0),
            rng.range_f64(3.0, 70.0),
            rng.range_f64(2.0, 45.0),
            rng.range_f64(5.0, 700.0),
            rng.range_f64(300.0, 3_500.0),
            rng.range_f64(-18.0, 18.0),
        ),
        CampaignBucket::WindborneDebris => (
            DimensionsSample {
                length: rng.range_f64(0.2, 5.0),
                wingspan: rng.range_f64(0.2, 8.0),
                height: rng.range_f64(0.2, 5.0),
            },
            rng.range_f64(-34.0, -6.0),
            rng.range_f64(0.0, 14.0),
            rng.range_f32(-5.0, 8.0),
            rng.range_f64(0.0, 8.0),
            rng.range_f64(0.0, 2.0),
            rng.range_f64(5.0, 1_200.0),
            rng.range_f64(300.0, 5_000.0),
            rng.range_f64(-8.0, 8.0),
        ),
        CampaignBucket::InfrastructureTerrain => {
            clutter.turbines = 0.6;
            clutter.buildings = 0.7;
            clutter.power_lines = 0.55;
            (
                DimensionsSample {
                    length: rng.range_f64(5.0, 80.0),
                    wingspan: rng.range_f64(2.0, 80.0),
                    height: rng.range_f64(5.0, 160.0),
                },
                rng.range_f64(0.0, 34.0),
                rng.range_f64(0.0, 260.0),
                rng.range_f32(-4.0, 14.0),
                rng.range_f64(0.0, 45.0),
                rng.range_f64(0.0, 120.0),
                rng.range_f64(20.0, 12_000.0),
                rng.range_f64(800.0, 18_000.0),
                rng.range_f64(-140.0, 140.0),
            )
        }
        CampaignBucket::GroundMoversMultipath => {
            clutter.roads_vehicles = 0.75;
            clutter.urban_multipath = 0.65;
            (
                DimensionsSample {
                    length: rng.range_f64(2.0, 14.0),
                    wingspan: rng.range_f64(1.5, 3.5),
                    height: rng.range_f64(1.0, 4.2),
                },
                rng.range_f64(-5.0, 18.0),
                rng.range_f64(0.0, 32.0),
                rng.range_f32(-2.0, 16.0),
                rng.range_f64(2.0, 40.0),
                rng.range_f64(4.0, 22.0),
                rng.range_f64(0.0, 3.0),
                rng.range_f64(200.0, 4_000.0),
                rng.range_f64(-25.0, 25.0),
            )
        }
        CampaignBucket::WeatherRfiSensorArtifacts => {
            clutter.rain = 0.75;
            clutter.dust_haze = 0.65;
            clutter.terrain_only_scene = 0.8;
            rfi.burst_probability = 0.07;
            rfi.narrowband_cw_power = 0.35;
            rfi.cochannel_emitters = 8;
            receiver.dropped_pulse_probability = 0.03;
            receiver.clipping_level = 1.2;
            (
                DimensionsSample {
                    length: 0.0,
                    wingspan: 0.0,
                    height: 0.0,
                },
                rng.range_f64(-45.0, -18.0),
                rng.range_f64(0.0, 12.0),
                rng.range_f32(-8.0, 8.0),
                rng.range_f64(0.0, 6.0),
                0.0,
                rng.range_f64(0.0, 2_500.0),
                rng.range_f64(100.0, 8_000.0),
                rng.range_f64(-6.0, 6.0),
            )
        }
    };

    ClassEnvelope {
        dimensions_m: dimensions,
        rcs_dbsm: rcs,
        speed_mps: speed,
        initial_range_m: range,
        heading_deg: rng.range_f64(-18.0, 18.0),
        acceleration_mps2: rng.range_f64(0.0, 1.2),
        climb_rate_mps: rng.range_f64(0.0, 4.0),
        max_altitude_m: altitude.max(1.0),
        radial_velocity_bias_mps: radial,
        attitude_jitter_deg: rng.range_f64(0.2, 4.0),
        propulsor_hz: prop,
        micro_doppler_hz: micro,
        base_snr_db: snr,
        clutter_profile: clutter,
        rfi_profile: rfi,
        receiver_profile: receiver,
    }
}

// ── Stressor application ──────────────────────────────────────────────────────

pub(super) fn apply_campaign_stressors(
    episode: &mut SyntheticEpisode,
    envelope: &ClassEnvelope,
    seed: u64,
) {
    apply_clutter_to_profile(
        &mut episode.integrated_range_profile,
        envelope.clutter_profile,
        seed ^ 0x1111,
    );
    apply_rfi_to_profile(
        &mut episode.integrated_range_profile,
        envelope.rfi_profile,
        seed ^ 0x2222,
    );
    for (pulse_index, pulse) in episode.iq.iter_mut().enumerate() {
        apply_receiver_impairments(pulse, envelope.receiver_profile, seed ^ 0x3333, pulse_index);
    }
    for (row_index, row) in episode.range_doppler_proxy.iter_mut().enumerate() {
        apply_rfi_to_profile(row, envelope.rfi_profile, seed ^ 0x4444 ^ row_index as u64);
    }
}

// ── Frame feature generation ──────────────────────────────────────────────────

pub(super) fn build_frame_features(
    config: &CampaignConfig,
    plan: &CampaignRecordPlan,
    envelope: &ClassEnvelope,
    frame_count: usize,
    cpi_pulses: usize,
) -> (Vec<FrameFeature>, Option<usize>) {
    let mut features = Vec::with_capacity(frame_count);
    let mut first_detectable = None;
    let mut persistence_frames = 0usize;
    let mut prev_snr = envelope.base_snr_db;

    for frame_index in 0..frame_count {
        let time_s = frame_index as f64 / config.frame_rate_hz;
        let progress = (time_s / config.time_window_s).clamp(0.0, 1.0);
        let mut frame_rng =
            SplitMix64::new(plan.seed ^ (frame_index as u64).wrapping_mul(0x1360_0d5));
        let clutter = sample_clutter_frame(envelope.clutter_profile, plan.seed, frame_index);
        let rfi = sample_rfi_frame(envelope.rfi_profile, plan.seed, frame_index, 128);
        let receiver =
            sample_receiver_impairments(envelope.receiver_profile, plan.seed, frame_index);
        let positive_rise = if plan.class.is_shahed_public_proxy {
            8.0 * (1.0 - (-time_s / 16.0).exp()) as f32
        } else {
            0.0
        };
        let family_noise = frame_rng.range_f32(-1.5, 1.5);
        let snr_db = (envelope.base_snr_db + positive_rise + family_noise
            - 4.0 * rfi.pressure
            - 2.0 * clutter.false_alarm_pressure)
            .clamp(-12.0, 30.0);
        if snr_db >= 7.0 && !receiver.dropped_pulse {
            persistence_frames += 1;
        } else {
            persistence_frames = persistence_frames.saturating_sub(1);
        }

        let first_gate =
            snr_db >= 7.0 && persistence_frames >= 2 && clutter.false_alarm_pressure < 0.85;
        if first_detectable.is_none() && first_gate {
            first_detectable = Some(frame_index);
        }

        let range_direction = if plan.class.is_shahed_public_proxy {
            -0.78
        } else {
            frame_rng.range_f64(-0.35, 0.35)
        };
        let range_m =
            (envelope.initial_range_m + range_direction * envelope.speed_mps * time_s).max(50.0);
        let radial_velocity = envelope.radial_velocity_bias_mps
            + frame_rng.range_f64(-2.0, 2.0)
            + if plan.class.bucket == CampaignBucket::InfrastructureTerrain {
                18.0 * (progress * std::f64::consts::PI).sin()
            } else {
                0.0
            };
        let altitude = if plan.class.is_shahed_public_proxy {
            (30.0 + 160.0 * progress + 20.0 * (progress * std::f64::consts::PI).sin())
                .min(envelope.max_altitude_m)
        } else {
            envelope.max_altitude_m * (0.15 + 0.8 * frame_rng.unit_f64())
        };
        let doppler_spread = (clutter.doppler_spread_hz
            + envelope.micro_doppler_hz as f32 * 0.12
            + frame_rng.range_f32(0.0, 8.0))
        .clamp(0.0, 320.0);
        let blob_area = (2.0
            + snr_db.max(0.0) * 0.35
            + clutter.false_alarm_pressure * 5.0
            + frame_rng.range_f32(0.0, 3.0))
        .clamp(0.0, 80.0);
        let micro_mod = (envelope.micro_doppler_hz as f32
            * (0.75 + 0.25 * (2.0 * std::f64::consts::PI * progress).sin() as f32)
            + frame_rng.range_f32(-4.0, 4.0))
        .max(0.0);

        features.push(FrameFeature {
            frame_index,
            time_s,
            cpi_pulses,
            range_m,
            range_rate_mps: radial_velocity,
            radial_velocity_mps: radial_velocity,
            altitude_m: altitude,
            snr_db,
            snr_trend_db: snr_db - prev_snr,
            doppler_spread_hz: doppler_spread,
            blob_area_bins: blob_area,
            micro_doppler_modulation: micro_mod,
            track_persistence_s: persistence_frames as f32 / config.frame_rate_hz as f32,
            clutter_pressure: clutter.false_alarm_pressure,
            rfi_pressure: rfi.pressure,
            receiver_dropout: receiver.dropped_pulse,
            phase_noise_rad: receiver.phase_offset_rad,
            amplitude_scintillation: envelope.receiver_profile.amplitude_scintillation_sigma,
        });
        prev_snr = snr_db;
    }

    (features, first_detectable)
}

// ── Frame label generation ────────────────────────────────────────────────────

pub(super) fn build_frame_labels(
    config: &CampaignConfig,
    plan: &CampaignRecordPlan,
    features: &[FrameFeature],
    first_detectable_frame: Option<usize>,
    first_model_trigger_frame: Option<usize>,
) -> Vec<FrameLabel> {
    features
        .iter()
        .map(|feature| FrameLabel {
            frame_index: feature.frame_index,
            time_s: feature.time_s,
            target_family: plan.class.target_family.clone(),
            is_shahed_public_proxy: plan.class.is_shahed_public_proxy,
            is_hard_negative: plan.class.is_hard_negative,
            hard_negative_family: plan.class.hard_negative_family.clone(),
            scenario_phase: scenario_phase(plan.class.is_shahed_public_proxy, feature.time_s),
            first_detectable_frame,
            first_model_trigger_frame,
            confidence_threshold: config.trigger_confidence,
        })
        .collect()
}

fn scenario_phase(is_positive: bool, time_s: f64) -> String {
    if !is_positive {
        return "confuser_or_artifact_motion".to_string();
    }
    if time_s < 8.0 {
        "rail_or_booster_launch_proxy".to_string()
    } else if time_s < 18.0 {
        "transition_to_pusher_prop".to_string()
    } else if time_s < 72.0 {
        "low_altitude_cruise".to_string()
    } else {
        "shallow_climb_turn".to_string()
    }
}

// ── Streaming model runner ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(super) struct ModelRun {
    pub events: Vec<FirstTriggerEvent>,
    pub predictions: Vec<ModelPredictionRow>,
    pub max_confidence_by_model: BTreeMap<String, f32>,
    pub first_trigger_by_model: BTreeMap<String, Option<usize>>,
}

pub(super) fn run_streaming_models(
    record_id: &str,
    features: &[FrameFeature],
    threshold: f32,
) -> Result<ModelRun, DatasetError> {
    let mut detectors: Vec<Box<dyn StreamingDetector>> = vec![
        Box::new(CfarTrackerBaseline::new(threshold)),
        Box::new(FeatureTreeClassifier::new(threshold)),
        Box::new(TemporalTinyModel::new(threshold)),
    ];
    let mut events = Vec::new();
    let mut predictions = Vec::with_capacity(features.len() * detectors.len());
    let mut max_confidence_by_model = BTreeMap::<String, f32>::new();
    let mut first_trigger_by_model = BTreeMap::<String, Option<usize>>::new();

    for detector in detectors.iter() {
        first_trigger_by_model.insert(detector.model_id().to_string(), None);
        max_confidence_by_model.insert(detector.model_id().to_string(), 0.0);
    }

    for frame in features {
        for detector in detectors.iter_mut() {
            let state = detector.update(frame);
            max_confidence_by_model
                .entry(state.model_id.clone())
                .and_modify(|value| *value = (*value).max(state.confidence))
                .or_insert(state.confidence);
            let triggered_on_this_frame = state.first_trigger_event.is_some();
            if let Some(event) = state.first_trigger_event.clone() {
                first_trigger_by_model.insert(state.model_id.clone(), Some(event.frame_index));
                events.push(event);
            }
            predictions.push(ModelPredictionRow {
                record_id: record_id.to_string(),
                model_id: state.model_id,
                frame_index: frame.frame_index,
                time_s: frame.time_s,
                confidence: state.confidence,
                class_probabilities_json: serde_json::to_string(&state.class_probabilities)?,
                triggered_on_this_frame,
            });
        }
    }

    Ok(ModelRun {
        events,
        predictions,
        max_confidence_by_model,
        first_trigger_by_model,
    })
}
