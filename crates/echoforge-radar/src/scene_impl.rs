use crate::sim::TargetState;

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
        propulsor_phase_rad: 0.0,
    }
}
