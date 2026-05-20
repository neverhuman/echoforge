use super::rng::{child_seed, deterministic_shuffle};
use super::types::{CampaignBucket, CampaignClass, CampaignConfig, CampaignRecordPlan};
use super::{DEFAULT_CAMPAIGN_REQUEST_ID, NEUTRAL_CAMPAIGN_ID, OWA_DELTA_OBJECT_ID};
use crate::monte_carlo::DatasetError;

// ── Validation ────────────────────────────────────────────────────────────────

pub(super) fn validate_campaign_config(config: &CampaignConfig) -> Result<(), DatasetError> {
    if config.campaign != DEFAULT_CAMPAIGN_REQUEST_ID && config.campaign != NEUTRAL_CAMPAIGN_ID {
        return Err(DatasetError::InvalidConfig(format!(
            "unknown campaign {}; expected {}",
            config.campaign, DEFAULT_CAMPAIGN_REQUEST_ID
        )));
    }
    if !(1..=50_000).contains(&config.records) {
        return Err(DatasetError::InvalidConfig(
            "records must be in the range 1..=50000".to_string(),
        ));
    }
    if config.shahed_min > config.records {
        return Err(DatasetError::InvalidConfig(
            "shahed-min cannot exceed records".to_string(),
        ));
    }
    if config.time_window_s <= 0.0 || config.frame_rate_hz <= 0.0 {
        return Err(DatasetError::InvalidConfig(
            "time-window-s and frame-rate-hz must be positive".to_string(),
        ));
    }
    if !(0.05..=0.99).contains(&config.trigger_confidence) {
        return Err(DatasetError::InvalidConfig(
            "trigger-confidence must be in the range 0.05..=0.99".to_string(),
        ));
    }
    if let Some(workers) = config.workers {
        if workers == 0 {
            return Err(DatasetError::InvalidConfig(
                "workers must be at least 1".to_string(),
            ));
        }
    }
    Ok(())
}

// ── Plan construction ─────────────────────────────────────────────────────────

pub(super) fn build_campaign_plan(
    config: &CampaignConfig,
) -> Result<Vec<CampaignRecordPlan>, DatasetError> {
    let classes = campaign_classes();
    let positive = classes
        .iter()
        .find(|class| class.bucket == CampaignBucket::PositiveOwaDeltaPusher)
        .expect("positive class exists")
        .clone();
    let target_positive = config
        .shahed_target
        .max(config.shahed_min)
        .min(config.records);
    let remainder = config.records - target_positive;
    let bucket_targets = weighted_remainder_counts(remainder);
    let mut assignments = Vec::with_capacity(config.records);
    assignments.extend(std::iter::repeat(positive).take(target_positive));

    for (bucket, count) in bucket_targets {
        let bucket_classes = classes
            .iter()
            .filter(|class| class.bucket == bucket)
            .cloned()
            .collect::<Vec<_>>();
        if bucket_classes.is_empty() && count > 0 {
            return Err(DatasetError::InvalidConfig(format!(
                "no campaign classes for bucket {bucket:?}"
            )));
        }
        for index in 0..count {
            assignments.push(bucket_classes[index % bucket_classes.len()].clone());
        }
    }

    deterministic_shuffle(&mut assignments, config.seed);
    Ok(assignments
        .into_iter()
        .enumerate()
        .map(|(index, class)| {
            let record_id = format!("record_{:06}", index + 1);
            CampaignRecordPlan {
                index,
                record_id,
                seed: child_seed(config.seed, index as u64),
                class,
            }
        })
        .collect())
}

fn weighted_remainder_counts(remainder: usize) -> Vec<(CampaignBucket, usize)> {
    let weights = [
        (CampaignBucket::SmallUav, 30usize),
        (CampaignBucket::Biological, 20),
        (CampaignBucket::WindborneDebris, 15),
        (CampaignBucket::InfrastructureTerrain, 15),
        (CampaignBucket::GroundMoversMultipath, 10),
        (CampaignBucket::WeatherRfiSensorArtifacts, 10),
    ];
    let mut counts = Vec::with_capacity(weights.len());
    let mut assigned = 0usize;
    for (bucket, weight) in weights {
        let count = remainder * weight / 100;
        assigned += count;
        counts.push((bucket, count));
    }
    let mut cursor = 0usize;
    while assigned < remainder {
        counts[cursor].1 += 1;
        assigned += 1;
        cursor = (cursor + 1) % counts.len();
    }
    counts
}

pub(super) fn campaign_classes() -> Vec<CampaignClass> {
    let positive = campaign_class(
        OWA_DELTA_OBJECT_ID,
        "Delta Pusher Fixed-Wing OWA Public Proxy",
        "owa_delta_pusher_public_proxy",
        CampaignBucket::PositiveOwaDeltaPusher,
        "positive_public_proxy",
        true,
        false,
    );
    let mut classes = vec![positive];
    classes.extend(hard_negative_classes());
    classes
}

fn hard_negative_classes() -> Vec<CampaignClass> {
    [
        ("low-altitude-fixed-wing-takeoff-v1", "Low-Altitude Fixed-Wing UAV Proxy", "fixed_wing_uav", CampaignBucket::SmallUav, "small_fixed_wing_uav"),
        ("commercial-quadrotor-low-altitude-v1", "Commercial Quadrotor Low-Altitude Proxy", "quadrotor_uav", CampaignBucket::SmallUav, "quadrotor"),
        ("hexarotor-heavy-lift-low-altitude-v1", "Hexarotor Heavy-Lift Low-Altitude Proxy", "hexarotor_uav", CampaignBucket::SmallUav, "hexarotor"),
        ("rc-plane-hobby-glider-v1", "RC Plane and Hobby Glider Proxy", "rc_fixed_wing_or_hobby_glider", CampaignBucket::SmallUav, "rc_plane_hobby_glider"),
        ("bird-large-and-flock-v1", "Large Bird and Flock Proxy", "bird_or_flock", CampaignBucket::Biological, "bird_or_flock"),
        ("bat-and-insect-cloud-v1", "Bat and Insect Cloud Biological Proxy", "bat_or_insect_cloud", CampaignBucket::Biological, "bat_or_insect_cloud"),
        ("balloon-kite-debris-v1", "Balloon, Kite, and Windborne Debris Proxy", "windborne_slow_object", CampaignBucket::WindborneDebris, "balloon_kite_windborne_debris"),
        ("wind-turbine-industrial-glint-v1", "Wind Turbine and Industrial Glint Proxy", "static_or_rotating_infrastructure", CampaignBucket::InfrastructureTerrain, "infrastructure_glint"),
        ("commercial-aircraft-corridor-clutter-v1", "Commercial Aircraft Corridor Clutter Proxy", "commercial_aircraft_corridor", CampaignBucket::InfrastructureTerrain, "commercial_aircraft_corridor"),
        ("ground-vehicle-roadside-v1", "Ground Vehicle and Roadside Multipath Proxy", "ground_vehicle", CampaignBucket::GroundMoversMultipath, "ground_movers_multipath"),
        ("weather-terrain-only-scene-v1", "Weather and Terrain-Only Scene Proxy", "weather_terrain_only", CampaignBucket::WeatherRfiSensorArtifacts, "weather_rfi_sensor_artifacts"),
    ]
    .into_iter()
    .map(|(id, display_name, target_family, bucket, hard_negative_family)| {
        campaign_class(id, display_name, target_family, bucket, hard_negative_family, false, true)
    })
    .collect()
}

fn campaign_class(
    id: &str,
    display_name: &str,
    target_family: &str,
    bucket: CampaignBucket,
    hard_negative_family: &str,
    is_shahed_public_proxy: bool,
    is_hard_negative: bool,
) -> CampaignClass {
    CampaignClass {
        id: id.to_string(),
        display_name: display_name.to_string(),
        target_family: target_family.to_string(),
        bucket,
        hard_negative_family: hard_negative_family.to_string(),
        is_shahed_public_proxy,
        is_hard_negative,
    }
}

