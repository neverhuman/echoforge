use std::fs;
use std::path::Path;

use echoforge_radar::SyntheticEpisode;
use ndarray::{ArrayD, IxDyn};
use serde::Serialize;

use crate::monte_carlo::DatasetError;

pub fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<(), DatasetError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(path, bytes)?;
    Ok(())
}

pub fn write_text(path: &Path, value: &str) -> Result<(), DatasetError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, value)?;
    Ok(())
}

pub fn write_episode_tensors(
    products_dir: &Path,
    episode: &SyntheticEpisode,
) -> Result<(), DatasetError> {
    fs::create_dir_all(products_dir)?;

    let pulse_count = episode.iq.len();
    let sample_count = episode.iq.first().map(|pulse| pulse.len()).unwrap_or(0);
    let iq_values = episode
        .iq
        .iter()
        .flat_map(|pulse| pulse.iter().copied())
        .collect::<Vec<_>>();
    let iq = ArrayD::from_shape_vec(IxDyn(&[pulse_count, sample_count]), iq_values)
        .map_err(|err| DatasetError::Tensor(err.to_string()))?;
    echoforge_sig::tensor::write_complex64(&products_dir.join("iq_complex.zarr"), &iq)?;

    let range_profile = ArrayD::from_shape_vec(
        IxDyn(&[episode.integrated_range_profile.len()]),
        episode.integrated_range_profile.clone(),
    )
    .map_err(|err| DatasetError::Tensor(err.to_string()))?;
    echoforge_sig::tensor::write_f32(&products_dir.join("range_profile.zarr"), &range_profile)?;

    let doppler_bins = episode.range_doppler_proxy.len();
    let range_bins = episode
        .range_doppler_proxy
        .first()
        .map(|row| row.len())
        .unwrap_or(0);
    let rd_values = episode
        .range_doppler_proxy
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let rd = ArrayD::from_shape_vec(IxDyn(&[doppler_bins, range_bins]), rd_values)
        .map_err(|err| DatasetError::Tensor(err.to_string()))?;
    echoforge_sig::tensor::write_f32(&products_dir.join("range_doppler_proxy.zarr"), &rd)?;

    Ok(())
}
