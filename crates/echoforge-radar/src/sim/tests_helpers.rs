use super::*;

pub(super) fn high_snr_episode(pulse_count: usize, seed: u64) -> SyntheticEpisode {
    let config = RadarSimConfig {
        pulse_count,
        target_snr_db: 28.0,
        ..RadarSimConfig::default()
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.awgn_sigma = 0.025;
    synthesize_takeoff_episode(config, TakeoffProfile::default(), noise, EpisodeSeed(seed))
}
