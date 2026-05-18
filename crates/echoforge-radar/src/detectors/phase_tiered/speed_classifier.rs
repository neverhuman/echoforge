//! Two-cluster (piston / jet) propulsion-class speed classifier for the
//! Shahed-136-class one-way-attack drone family.
//!
//! Public-proxy clusters per the Wave-A
//! `shahed-public-proxy-flight-envelope-v2` dossier
//! (`object-packs/public-proxy-v1/physics_dossier.md`):
//!   * **Piston** (Shahed-136 piston variant): cruise 50–55 m/s,
//!     max 55–60 m/s, min sustainable 35–40 m/s → use [40, 60] m/s.
//!   * **Jet** (Shahed-238 variant): cruise ~110–145 m/s
//!     (400–520 km/h) → use [100, 150] m/s.
//!
//! Any speed in the explicit unmodeled gap (60, 100) m/s is reported as
//! `Ambiguous` rather than silently bucketed into a cluster. Speeds
//! below 25 m/s are reported as `BirdLike` and speeds above 200 m/s as
//! `AircraftLike` so reviewers can see explicit out-of-class rejections
//! instead of false-positive Shahed labels.
//!
//! Strict-open posture: the cluster bounds reflect *public-proxy
//! expected* behaviour and do NOT claim platform-specific signature truth.

/// Coarse propulsion-class classification of a measured speed in m/s.
/// `Piston` / `Jet` mean "consistent with the dossier's piston / jet
/// Shahed cluster"; `Ambiguous` means "in the explicit dossier gap";
/// `BirdLike` / `AircraftLike` / `None` are out-of-class rejections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PropulsionClass {
    /// Speed consistent with the piston Shahed cruise cluster.
    Piston,
    /// Speed consistent with the jet Shahed cruise cluster.
    Jet,
    /// Speed in the explicit unmodeled gap between piston and jet
    /// cruise envelopes (60–100 m/s by default).
    Ambiguous,
    /// Speed below the piston cluster minimum (typical bird /
    /// small-target band; default <25 m/s).
    BirdLike,
    /// Speed above the jet cluster maximum (typical manned-aircraft
    /// or fighter regime; default >200 m/s).
    AircraftLike,
    /// Default value — no speed available or below 0.
    None,
}

/// Two-cluster speed classifier. The default Shahed-class configuration
/// uses piston [40, 60] m/s and jet [100, 150] m/s; the gap (60, 100) is
/// reported as `Ambiguous` and out-of-band speeds as `BirdLike` / `AircraftLike`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeedClassifier {
    pub piston_min: f64,
    pub piston_max: f64,
    pub jet_min: f64,
    pub jet_max: f64,
    /// Anything strictly below this is `BirdLike`. Default 25 m/s.
    pub bird_max: f64,
    /// Anything strictly above this is `AircraftLike`. Default 200 m/s.
    pub aircraft_min: f64,
}

impl SpeedClassifier {
    /// Default classifier for the Shahed-136 class, sourced from the
    /// `shahed-public-proxy-flight-envelope-v2` dossier:
    /// piston cluster [40, 60] m/s, jet cluster [100, 150] m/s.
    pub fn shahed_class_default() -> Self {
        Self {
            piston_min: 40.0,
            piston_max: 60.0,
            jet_min: 100.0,
            jet_max: 150.0,
            bird_max: 25.0,
            aircraft_min: 200.0,
        }
    }

    /// Classify a measured speed in m/s. Negative speeds are mapped to
    /// `None` (the classifier treats |speed|; callers that intend signed
    /// closing speed should pass `speed.abs()`).
    pub fn classify(&self, speed_mps: f64) -> PropulsionClass {
        if !speed_mps.is_finite() || speed_mps < 0.0 {
            return PropulsionClass::None;
        }
        if speed_mps < self.bird_max {
            return PropulsionClass::BirdLike;
        }
        if speed_mps >= self.piston_min && speed_mps <= self.piston_max {
            return PropulsionClass::Piston;
        }
        if speed_mps >= self.jet_min && speed_mps <= self.jet_max {
            return PropulsionClass::Jet;
        }
        if speed_mps > self.aircraft_min {
            return PropulsionClass::AircraftLike;
        }
        // Catches the explicit dossier gap (60, 100) and the small
        // residual sub-gap (25 to 40) between bird and piston cluster.
        PropulsionClass::Ambiguous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_classifier_piston_50mps() {
        let c = SpeedClassifier::shahed_class_default();
        assert_eq!(c.classify(50.0), PropulsionClass::Piston);
    }

    #[test]
    fn speed_classifier_jet_120mps() {
        let c = SpeedClassifier::shahed_class_default();
        assert_eq!(c.classify(120.0), PropulsionClass::Jet);
    }

    #[test]
    fn speed_classifier_ambiguous_75mps() {
        let c = SpeedClassifier::shahed_class_default();
        // 75 m/s is in the dossier gap (60, 100).
        assert_eq!(c.classify(75.0), PropulsionClass::Ambiguous);
    }

    #[test]
    fn speed_classifier_bird_15mps() {
        let c = SpeedClassifier::shahed_class_default();
        assert_eq!(c.classify(15.0), PropulsionClass::BirdLike);
    }

    #[test]
    fn speed_classifier_aircraft_300mps() {
        let c = SpeedClassifier::shahed_class_default();
        assert_eq!(c.classify(300.0), PropulsionClass::AircraftLike);
    }

    #[test]
    fn speed_classifier_piston_min_boundary_inclusive() {
        let c = SpeedClassifier::shahed_class_default();
        // 40 m/s is the dossier's minimum sustainable airspeed top — must
        // round to Piston, not Ambiguous.
        assert_eq!(c.classify(40.0), PropulsionClass::Piston);
    }

    #[test]
    fn speed_classifier_jet_max_boundary_inclusive() {
        let c = SpeedClassifier::shahed_class_default();
        // 150 m/s is the upper bound of the Shahed-238 dossier band.
        assert_eq!(c.classify(150.0), PropulsionClass::Jet);
    }

    #[test]
    fn speed_classifier_negative_and_nan_return_none() {
        let c = SpeedClassifier::shahed_class_default();
        assert_eq!(c.classify(-5.0), PropulsionClass::None);
        assert_eq!(c.classify(f64::NAN), PropulsionClass::None);
    }
}
