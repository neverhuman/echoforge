//! Per-class envelope sampling for campaign simulation — extracted for LOC compliance.

use echoforge_radar::{ClutterProfile, ReceiverImpairmentProfile, RfiProfile};

use super::ClassEnvelope;
use crate::campaign::rng::SplitMix64;
use crate::campaign::types::{CampaignBucket, CampaignClass, DimensionsSample};

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
