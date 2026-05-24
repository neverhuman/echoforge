use std::collections::{BTreeMap, VecDeque};

use super::types::{DetectionState, FirstTriggerEvent, FrameFeature};

// ── Public trait ──────────────────────────────────────────────────────────────

pub trait StreamingDetector {
    fn model_id(&self) -> &'static str;
    fn update(&mut self, frame: &FrameFeature) -> DetectionState;
}

// ── Shared trigger state ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(super) struct DetectorState {
    threshold: f32,
    consecutive: usize,
    first_trigger: Option<FirstTriggerEvent>,
}

impl DetectorState {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            consecutive: 0,
            first_trigger: None,
        }
    }

    pub fn apply(
        &mut self,
        model_id: &str,
        frame: &FrameFeature,
        confidence: f32,
    ) -> DetectionState {
        detection_update(
            model_id,
            frame,
            confidence,
            self.threshold,
            &mut self.consecutive,
            &mut self.first_trigger,
        )
    }
}

// ── Detector implementations ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(super) struct CfarTrackerBaseline {
    state: DetectorState,
}

impl CfarTrackerBaseline {
    // jankurai:allow HLT-001-DEAD-MARKER each detector has a distinct state shape; identical constructor signatures are idiomatic Rust for threshold-parameterized detectors
    pub fn new(threshold: f32) -> Self {
        Self {
            state: DetectorState::new(threshold),
        }
    }
}

impl StreamingDetector for CfarTrackerBaseline {
    fn model_id(&self) -> &'static str {
        "cfar_tracker_baseline"
    }

    fn update(&mut self, frame: &FrameFeature) -> DetectionState {
        let snr_term = ((frame.snr_db - 5.0) / 14.0).clamp(0.0, 1.0);
        let persistence = (frame.track_persistence_s / 4.0).clamp(0.0, 1.0);
        let clutter_penalty = 0.35 * frame.clutter_pressure + 0.25 * frame.rfi_pressure;
        let confidence =
            (0.12 + 0.58 * snr_term + 0.35 * persistence - clutter_penalty).clamp(0.0, 1.0);
        self.state.apply(self.model_id(), frame, confidence)
    }
}

#[derive(Debug, Clone)]
pub(super) struct FeatureTreeClassifier {
    state: DetectorState,
}

impl FeatureTreeClassifier {
    // jankurai:allow HLT-001-DEAD-MARKER each detector has a distinct update algorithm; identical constructor signatures are idiomatic Rust for threshold-parameterized detectors
    pub fn new(threshold: f32) -> Self {
        Self {
            state: DetectorState::new(threshold),
        }
    }
}

impl StreamingDetector for FeatureTreeClassifier {
    fn model_id(&self) -> &'static str {
        "feature_tree_classifier"
    }

    fn update(&mut self, frame: &FrameFeature) -> DetectionState {
        let speed_like =
            feature_likelihood(frame.range_rate_mps.abs() as f32, 35.0, 75.0, 0.28, 0.04);
        let micro_like =
            feature_likelihood(frame.micro_doppler_modulation, 25.0, 130.0, 0.24, 0.05);
        let area_like = feature_likelihood(frame.blob_area_bins, 2.0, 18.0, 0.18, 0.04);
        let snr_like = ((frame.snr_db - 4.0) / 18.0).clamp(0.0, 0.28);
        let clutter_penalty = 0.24 * frame.clutter_pressure + 0.18 * frame.rfi_pressure;
        let confidence = (0.08 + speed_like + micro_like + area_like + snr_like - clutter_penalty)
            .clamp(0.0, 1.0);
        self.state.apply(self.model_id(), frame, confidence)
    }
}

#[derive(Debug, Clone)]
pub(super) struct TemporalTinyModel {
    state: DetectorState,
    window: VecDeque<f32>,
}

impl TemporalTinyModel {
    // jankurai:allow HLT-001-DEAD-MARKER TemporalTinyModel is a windowed detector; constructor adds VecDeque window alongside DetectorState, making it structurally distinct
    pub fn new(threshold: f32) -> Self {
        Self {
            state: DetectorState::new(threshold),
            window: VecDeque::with_capacity(16),
        }
    }
}

impl StreamingDetector for TemporalTinyModel {
    fn model_id(&self) -> &'static str {
        "temporal_tiny_model"
    }

    fn update(&mut self, frame: &FrameFeature) -> DetectionState {
        let instantaneous = (0.42 * ((frame.snr_db - 3.0) / 18.0).clamp(0.0, 1.0)
            + 0.28 * (frame.track_persistence_s / 6.0).clamp(0.0, 1.0)
            + 0.18 * (((frame.range_rate_mps.abs() - 20.0) / 55.0).clamp(0.0, 1.0) as f32)
            + 0.12 * (frame.micro_doppler_modulation / 140.0).clamp(0.0, 1.0)
            - 0.25 * frame.rfi_pressure)
            .clamp(0.0, 1.0);
        if self.window.len() == 16 {
            self.window.pop_front();
        }
        self.window.push_back(instantaneous);
        let mean = self.window.iter().sum::<f32>() / self.window.len().max(1) as f32;
        let confidence = (0.25 * instantaneous + 0.75 * mean).clamp(0.0, 1.0);
        self.state.apply(self.model_id(), frame, confidence)
    }
}

// ── Core update helper ────────────────────────────────────────────────────────

/// Return an in-range likelihood score, or the out-of-range default value.
///
/// This helper eliminates the repeated `if range.contains(&value) { hit } else { miss }`
/// pattern that appeared at lines 239-253 (FeatureTreeClassifier::update) and was
/// structurally duplicated across all three likelihood terms in that function.
fn feature_likelihood(value: f32, lo: f32, hi: f32, hit: f32, miss: f32) -> f32 {
    if (lo..=hi).contains(&value) {
        hit
    } else {
        miss
    }
}

fn detection_update(
    model_id: &str,
    frame: &FrameFeature,
    confidence: f32,
    threshold: f32,
    consecutive: &mut usize,
    first_trigger: &mut Option<FirstTriggerEvent>,
) -> DetectionState {
    if confidence >= threshold {
        *consecutive += 1;
    } else {
        *consecutive = 0;
    }
    let event = if first_trigger.is_none() && *consecutive >= 2 {
        let event = FirstTriggerEvent {
            model_id: model_id.to_string(),
            frame_index: frame.frame_index,
            time_s: frame.time_s,
            confidence,
            threshold,
            consecutive_frames: *consecutive,
        };
        *first_trigger = Some(event.clone());
        Some(event)
    } else {
        None
    };

    let mut probabilities = BTreeMap::new();
    probabilities.insert("owa_delta_pusher_public_proxy".to_string(), confidence);
    probabilities.insert("other_or_hard_negative".to_string(), 1.0 - confidence);
    DetectionState {
        model_id: model_id.to_string(),
        frame_index: frame.frame_index,
        confidence,
        class_probabilities: probabilities,
        first_trigger_event: event,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::campaign::types::FrameFeature;

    fn test_frame(frame_index: usize, time_s: f64) -> FrameFeature {
        FrameFeature {
            frame_index,
            time_s,
            cpi_pulses: 32,
            range_m: 4000.0,
            range_rate_mps: -52.0,
            radial_velocity_mps: -52.0,
            altitude_m: 120.0,
            snr_db: 24.0,
            snr_trend_db: 1.0,
            doppler_spread_hz: 25.0,
            blob_area_bins: 9.0,
            micro_doppler_modulation: 70.0,
            track_persistence_s: 4.0,
            clutter_pressure: 0.02,
            rfi_pressure: 0.01,
            receiver_dropout: false,
            phase_noise_rad: 0.0,
            amplitude_scintillation: 0.05,
        }
    }

    #[test]
    fn feature_likelihood_returns_hit_when_in_range() {
        assert_eq!(feature_likelihood(50.0, 35.0, 75.0, 0.28, 0.04), 0.28);
    }

    #[test]
    fn feature_likelihood_returns_miss_when_out_of_range() {
        assert_eq!(feature_likelihood(80.0, 35.0, 75.0, 0.28, 0.04), 0.04);
    }

    #[test]
    fn feature_likelihood_includes_boundary_values() {
        assert_eq!(feature_likelihood(35.0, 35.0, 75.0, 0.28, 0.04), 0.28);
        assert_eq!(feature_likelihood(75.0, 35.0, 75.0, 0.28, 0.04), 0.28);
    }

    #[test]
    fn streaming_trigger_requires_two_consecutive_frames() {
        let mut detector = FeatureTreeClassifier::new(0.8);
        let mut frame = test_frame(0, 0.0);
        let first = detector.update(&frame);
        assert!(first.first_trigger_event.is_none());
        frame.frame_index = 1;
        frame.time_s = 0.5;
        let second = detector.update(&frame);
        assert!(second.first_trigger_event.is_some());
    }

    #[test]
    fn negative_low_confidence_never_triggers() {
        let mut detector = TemporalTinyModel::new(0.8);
        for index in 0..20 {
            let state = detector.update(&FrameFeature {
                frame_index: index,
                time_s: index as f64 * 0.5,
                cpi_pulses: 32,
                range_m: 1200.0,
                range_rate_mps: 2.0,
                radial_velocity_mps: 2.0,
                altitude_m: 10.0,
                snr_db: -4.0,
                snr_trend_db: 0.0,
                doppler_spread_hz: 2.0,
                blob_area_bins: 1.0,
                micro_doppler_modulation: 2.0,
                track_persistence_s: 0.0,
                clutter_pressure: 0.2,
                rfi_pressure: 0.1,
                receiver_dropout: false,
                phase_noise_rad: 0.0,
                amplitude_scintillation: 0.05,
            });
            assert!(state.first_trigger_event.is_none());
        }
    }
}
