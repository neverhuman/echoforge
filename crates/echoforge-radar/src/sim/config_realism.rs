//! Realism extension config types for radar synthesis.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PropagationAnomaly {
    pub min_elevation_deg: f64,
    pub excess_loss_db: f64,
    pub ducting_gain_db: f64,
}

impl PropagationAnomaly {
    pub fn bounded(self) -> Self {
        Self {
            min_elevation_deg: self.min_elevation_deg.clamp(-10.0, 45.0),
            excess_loss_db: self.excess_loss_db.clamp(0.0, 80.0),
            ducting_gain_db: self.ducting_gain_db.clamp(0.0, 40.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrackArtifactProfile {
    pub false_track_rate_per_hour: f64,
    pub fragmentation_probability: f64,
    pub missed_update_probability: f64,
}

impl TrackArtifactProfile {
    pub fn bounded(self) -> Self {
        Self {
            false_track_rate_per_hour: self.false_track_rate_per_hour.clamp(0.0, 10_000.0),
            fragmentation_probability: self.fragmentation_probability.clamp(0.0, 1.0),
            missed_update_probability: self.missed_update_probability.clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientEvent {
    pub kind: TransientEventKind,
    pub start_s: f64,
    pub duration_s: f64,
    pub strength: f32,
    #[serde(default)]
    pub target_ref: Option<usize>,
}

impl TransientEvent {
    pub fn active_at(&self, t_s: f64) -> bool {
        self.duration_s > 0.0 && t_s >= self.start_s && t_s < self.start_s + self.duration_s
    }

    pub fn bounded_strength(&self) -> f32 {
        self.strength.clamp(0.0, 16.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransientEventKind {
    RfiBurst,
    Dropout,
    Glint,
    MultipathGhost,
    ClutterSurge,
    WeatherVolume,
}
