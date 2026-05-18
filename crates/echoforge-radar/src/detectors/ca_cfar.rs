//! CA-CFAR (Cell-Averaging Constant False Alarm Rate) detector wrapper.
//!
//! This file is a thin `Detector`-trait wrapper around the existing
//! `crate::cfar::ca_cfar_1d` implementation, which remains the source of
//! truth and is still exported at the crate root for backward compatibility.

use crate::cfar::{ca_cfar_1d, CfarParams};

use super::{magnitude_to_db, DetectionEvent, DetectionKind, Detector};

#[derive(Debug, Clone, Copy)]
pub struct CaCfarDetector {
    pub params: CfarParams,
}

impl CaCfarDetector {
    pub fn new(params: CfarParams) -> Self {
        Self { params }
    }
}

impl Detector for CaCfarDetector {
    type Input = Vec<f32>;

    fn detect(&self, input: &Self::Input) -> Vec<DetectionEvent> {
        ca_cfar_1d(input, self.params)
            .into_iter()
            .filter(|d| d.evaluated && d.detected)
            .map(|d| {
                DetectionEvent::new(
                    d.index,
                    None,
                    magnitude_to_db(d.statistic as f64),
                    DetectionKind::CaCfar,
                )
            })
            .collect()
    }

    fn kind(&self) -> DetectionKind {
        DetectionKind::CaCfar
    }
}
