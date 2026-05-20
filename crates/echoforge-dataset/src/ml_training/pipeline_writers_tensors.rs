//! Tensor writers: multi-view projections, learned-window products.
//! Extracted from pipeline_writers.rs for LOC compliance.

use std::fs;
use std::path::Path;

use crate::export::write_json_pretty as write_json;
use crate::monte_carlo::DatasetError;
use crate::ml_training::types::{LearnedWindowEntry, LearnedWindowManifest, MlFrameFeatureRow};
use crate::ml_training::util::write_f32_tensor;

pub fn write_multi_view_products(
    record_dir: &Path,
    features: &[MlFrameFeatureRow],
) -> Result<(), DatasetError> {
    let dir = record_dir.join("multi_view");
    fs::create_dir_all(&dir)?;
    let frames = features.len();
    let range_bins = 24usize;
    let doppler_bins = 24usize;
    let rd_range_bins = 12usize;
    let rd_doppler_bins = 12usize;

    let range_time = build_1d_projection(frames, range_bins, features, |f| {
        let center = ((f.range_m / 9_000.0) * (range_bins as f64 - 1.0))
            .clamp(0.0, range_bins as f64 - 1.0) as f32;
        let energy = f.range_time_energy;
        let noise = f.rfi_pressure;
        move |bin: usize| {
            let dist = bin as f32 - center;
            (energy * (-dist * dist / 18.0).exp() + 0.02 * noise).max(0.0)
        }
    });
    write_f32_tensor(&dir.join("range_time.zarr"), &[frames, range_bins], range_time)?;

    let doppler_time = build_1d_projection(frames, doppler_bins, features, |f| {
        let center = (((f.radial_velocity_mps + 160.0) / 320.0) * (doppler_bins as f64 - 1.0))
            .clamp(0.0, doppler_bins as f64 - 1.0) as f32;
        let energy = f.doppler_time_energy;
        let dropout = f.dropout_fraction;
        move |bin: usize| {
            let dist = bin as f32 - center;
            (energy * (-dist * dist / 14.0).exp() + 0.015 * dropout).max(0.0)
        }
    });
    write_f32_tensor(
        &dir.join("doppler_time.zarr"),
        &[frames, doppler_bins],
        doppler_time,
    )?;

    let mut rdt = Vec::with_capacity(frames * rd_range_bins * rd_doppler_bins);
    for feature in features {
        let range_center = ((feature.range_m / 9_000.0) * (rd_range_bins as f64 - 1.0))
            .clamp(0.0, rd_range_bins as f64 - 1.0) as f32;
        let doppler_center = (((feature.radial_velocity_mps + 160.0) / 320.0)
            * (rd_doppler_bins as f64 - 1.0))
            .clamp(0.0, rd_doppler_bins as f64 - 1.0) as f32;
        for d in 0..rd_doppler_bins {
            for r in 0..rd_range_bins {
                let rd = d as f32 - doppler_center;
                let rr = r as f32 - range_center;
                rdt.push(
                    (feature.range_doppler_time_energy * (-(rr * rr + rd * rd) / 12.0).exp()
                        + 0.01 * feature.rfi_pressure)
                        .max(0.0),
                );
            }
        }
    }
    write_f32_tensor(
        &dir.join("range_doppler_time.zarr"),
        &[frames, rd_doppler_bins, rd_range_bins],
        rdt,
    )?;
    use serde_json::json;
    write_json(
        &dir.join("range_angle_schema_pending.json"),
        &json!({
            "status": "unavailable",
            "reason": "range-angle and range-azimuth-Doppler tensors require future public-proxy MIMO channel synthesis; v1 records the schema as pending only",
            "reserved_shapes": {
                "range_angle": ["frames", "angle_bins", "range_bins"],
                "range_azimuth_doppler": ["frames", "azimuth_bins", "doppler_bins", "range_bins"]
            }
        }),
    )?;
    Ok(())
}

pub fn write_learned_windows(
    record_dir: &Path,
    record_id: &str,
    features: &[MlFrameFeatureRow],
) -> Result<(), DatasetError> {
    let dir = record_dir.join("learned_windows");
    fs::create_dir_all(&dir)?;
    let feature_order = vec![
        "normalized_snr".to_string(),
        "range_norm".to_string(),
        "velocity_norm".to_string(),
        "clutter_proxy".to_string(),
        "rfi_pressure".to_string(),
        "dropout_fraction".to_string(),
        "micro_doppler_energy".to_string(),
        "tbd_track_score".to_string(),
    ];
    let mut entries = Vec::new();
    for window in [8usize, 16, 32] {
        let (values, windows) = learned_window_values(features, window);
        let path = format!("window_{window}.zarr");
        write_f32_tensor(
            &dir.join(&path),
            &[windows, window, feature_order.len()],
            values,
        )?;
        entries.push(LearnedWindowEntry {
            window_frames: window,
            path,
            shape: vec![windows, window, feature_order.len()],
        });
    }
    write_json(
        &dir.join("manifest.json"),
        &LearnedWindowManifest {
            record_id: record_id.to_string(),
            windows: entries,
            feature_order,
        },
    )?;
    Ok(())
}

fn learned_window_values(features: &[MlFrameFeatureRow], window: usize) -> (Vec<f32>, usize) {
    let stride = (window / 2).max(1);
    let windows = if features.len() <= window {
        1
    } else {
        ((features.len() - window) / stride) + 1
    };
    let mut values = Vec::with_capacity(windows * window * 8);
    for w in 0..windows {
        let start = (w * stride).min(features.len().saturating_sub(1));
        for offset in 0..window {
            let feature = features
                .get((start + offset).min(features.len().saturating_sub(1)))
                .expect("features nonempty");
            values.extend_from_slice(&[
                feature.normalized_snr,
                (feature.range_m as f32 / 10_000.0).clamp(0.0, 1.0),
                ((feature.radial_velocity_mps as f32 + 180.0) / 360.0).clamp(0.0, 1.0),
                ((feature.local_noise_floor_db + 60.0) / 48.0).clamp(0.0, 1.0),
                feature.rfi_pressure,
                feature.dropout_fraction,
                feature.micro_doppler_energy,
                feature.tbd_track_score,
            ]);
        }
    }
    (values, windows)
}

fn build_1d_projection<F, G>(
    frames: usize,
    bins: usize,
    features: &[MlFrameFeatureRow],
    per_feature: F,
) -> Vec<f32>
where
    F: Fn(&MlFrameFeatureRow) -> G,
    G: Fn(usize) -> f32,
{
    let mut out = Vec::with_capacity(frames * bins);
    for feature in features {
        let pixel = per_feature(feature);
        for bin in 0..bins {
            out.push(pixel(bin));
        }
    }
    out
}