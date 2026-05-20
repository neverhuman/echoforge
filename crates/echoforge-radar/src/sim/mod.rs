mod config;
mod dft;
mod episode;
mod helpers;
mod synthesize;

pub use config::{NoiseProfile, RadarSimConfig, TakeoffProfile};
pub use dft::{slow_time_complex_dft, slow_time_dft_magnitude};
pub use episode::{DetectionRecord, EpisodeSeed, SyntheticEpisode, TargetState};
pub use helpers::range_bin_to_m;
pub use synthesize::{synthesize_scene, synthesize_takeoff_episode};

#[cfg(test)]
mod tests_helpers;

#[cfg(test)]
mod tests_propeller;

#[cfg(test)]
mod tests_synthesis;

#[cfg(test)]
mod tests_dft;

#[cfg(test)]
mod tests_pol_clutter_a;

#[cfg(test)]
mod tests_pol_clutter_b;
