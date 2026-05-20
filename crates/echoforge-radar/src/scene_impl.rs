use crate::sim::{TakeoffProfile, TargetState};

/// Dispatch body for [`super::TargetKinematics::state_at`].
/// Extracted here to keep `scene.rs` within the 350-LOC clean-code limit.
pub(super) fn target_kinematics_state_at(
    kinematics: &super::TargetKinematics,
    t_s: f64,
    initial_range_m: f64,
    antenna_alt_agl_m: f64,
) -> TargetState {
    let _ = antenna_alt_agl_m;
    match kinematics {
        super::TargetKinematics::FromTakeoffProfile(profile) => {
            TakeoffProfile::state_at(profile, t_s)
        }

        super::TargetKinematics::Bird {
            cruise_speed_mps,
            altitude_agl_m,
            heading_deg,
            ..
        } => straight_line_state(
            t_s,
            initial_range_m,
            *cruise_speed_mps,
            *heading_deg,
            *altitude_agl_m,
        ),

        super::TargetKinematics::GroundVehicle {
            speed_mps,
            heading_deg,
            initial_range_m: vehicle_initial_range_m,
        } => {
            let heading_rad = heading_deg.to_radians();
            let along_track_m = speed_mps * t_s;
            let range_m = (vehicle_initial_range_m - along_track_m * heading_rad.cos()).max(0.0);
            let radial_velocity_mps = speed_mps * heading_rad.cos();
            TargetState {
                time_s: t_s,
                range_m,
                altitude_m: 0.0,
                radial_velocity_mps,
                pitch_deg: 0.0,
                yaw_deg: 0.0,
                course_deg: *heading_deg,
                propulsor_phase_rad: 0.0,
            }
        }

        super::TargetKinematics::WindTurbine {
            hub_range_m,
            hub_altitude_agl_m,
            ..
        } => TargetState {
            time_s: t_s,
            range_m: *hub_range_m,
            altitude_m: *hub_altitude_agl_m,
            radial_velocity_mps: 0.0,
            pitch_deg: 0.0,
            yaw_deg: 0.0,
            course_deg: 0.0,
            propulsor_phase_rad: 0.0,
        },

        super::TargetKinematics::MultipathGhost { .. } => panic!(
            "TargetKinematics::MultipathGhost::state_at called directly; \
             callers must first resolve the parent entity's state and \
             apply the multipath geometry at synthesize_scene level. \
             See crate::sim::synthesize_scene for the dispatch."
        ),

        super::TargetKinematics::Balloon {
            drift_speed_mps,
            drift_heading_deg,
            altitude_agl_m,
            tethered,
        } => {
            if *tethered {
                TargetState {
                    time_s: t_s,
                    range_m: initial_range_m,
                    altitude_m: *altitude_agl_m,
                    radial_velocity_mps: 0.0,
                    pitch_deg: 0.0,
                    yaw_deg: 0.0,
                    course_deg: 0.0,
                    propulsor_phase_rad: 0.0,
                }
            } else {
                straight_line_state(
                    t_s,
                    initial_range_m,
                    *drift_speed_mps,
                    *drift_heading_deg,
                    *altitude_agl_m,
                )
            }
        }

        super::TargetKinematics::Kite {
            anchor_range_m,
            altitude_agl_m,
            wind_gust_amplitude_mps,
        } => {
            let gust_carrier_hz = 0.5;
            let radial_velocity_mps = wind_gust_amplitude_mps
                * (2.0 * std::f64::consts::PI * gust_carrier_hz * t_s).sin();
            TargetState {
                time_s: t_s,
                range_m: *anchor_range_m,
                altitude_m: *altitude_agl_m,
                radial_velocity_mps,
                pitch_deg: 0.0,
                yaw_deg: 0.0,
                course_deg: 0.0,
                propulsor_phase_rad: 0.0,
            }
        }

        super::TargetKinematics::Helicopter {
            cruise_speed_mps,
            altitude_agl_m,
            heading_deg,
            ..
        } => straight_line_state(
            t_s,
            initial_range_m,
            *cruise_speed_mps,
            *heading_deg,
            *altitude_agl_m,
        ),
    }
}

pub(super) fn straight_line_state(
    t_s: f64,
    initial_range_m: f64,
    speed_mps: f64,
    heading_deg: f64,
    altitude_agl_m: f64,
) -> TargetState {
    let heading_rad = heading_deg.to_radians();
    let along_track_m = speed_mps * t_s;
    let range_m = (initial_range_m - along_track_m * heading_rad.cos()).max(0.0);
    let radial_velocity_mps = speed_mps * heading_rad.cos();
    TargetState {
        time_s: t_s,
        range_m,
        altitude_m: altitude_agl_m,
        radial_velocity_mps,
        pitch_deg: 0.0,
        yaw_deg: 0.0,
        course_deg: heading_deg,
        propulsor_phase_rad: 0.0,
    }
}
