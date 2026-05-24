//! Utility functions: CSV/tensor I/O, DSP helpers, PRNG utilities, class
//! tables, and record-plan construction.

use std::fs;
use std::path::Path;
use std::time::Instant;

use ndarray::{ArrayD, IxDyn};

use crate::monte_carlo::DatasetError;
use crate::rng::{child_seed, deterministic_shuffle};
use crate::split::SplitKind;

use super::config::MlTrainingDataConfig;
use super::types::{MlClass, MlRecordPlan};

#[path = "util_classes.rs"]
mod util_classes;

// ---------------------------------------------------------------------------
// Record-plan construction
// ---------------------------------------------------------------------------

pub(super) fn build_record_plan(
    config: &MlTrainingDataConfig,
    _frame_count: usize,
) -> Result<Vec<MlRecordPlan>, DatasetError> {
    let classes = ml_classes();
    if let (Some(pos_per_class), Some(neg_per_sensor)) = (
        config.positives_per_class_per_sensor,
        config.negatives_per_sensor,
    ) {
        return build_exact_sensor_class_plan(config, &classes, pos_per_class, neg_per_sensor);
    }

    let positive_count = ((config.records as f64 * config.positive_fraction).round() as usize)
        .max(1)
        .min(config.records);
    let positives = positive_classes(config, &classes)?;
    let sensors = nonempty_sensor_ids(config);
    let phases = nonempty_phase_targets(config);
    let positive_assignments = (0..positive_count)
        .map(|index| positives[index % positives.len()].clone())
        .collect::<Vec<_>>();
    let negative_classes = classes
        .iter()
        .filter(|class| class.is_hard_negative)
        .cloned()
        .collect::<Vec<_>>();

    let mut assignments = Vec::with_capacity(config.records);
    assignments.extend(positive_assignments);
    for index in 0..(config.records - positive_count) {
        assignments.push(negative_classes[index % negative_classes.len()].clone());
    }
    deterministic_shuffle(&mut assignments, config.seed ^ 0x004d_4c74_7261_696e);

    let mut plans = assignments
        .into_iter()
        .enumerate()
        .map(|(index, class)| plan_for_assignment(config, index, class, &sensors, &phases))
        .collect::<Vec<_>>();
    assign_exact_splits(&mut plans, config.seed ^ 0x0073_706c_6974);
    Ok(plans)
}

fn build_exact_sensor_class_plan(
    config: &MlTrainingDataConfig,
    classes: &[MlClass],
    pos_per_class: usize,
    neg_per_sensor: usize,
) -> Result<Vec<MlRecordPlan>, DatasetError> {
    let positives = positive_classes(config, classes)?;
    let sensors = nonempty_sensor_ids(config);
    let phases = nonempty_phase_targets(config);
    let negative_classes = classes
        .iter()
        .filter(|class| class.is_hard_negative)
        .cloned()
        .collect::<Vec<_>>();

    let mut assignments = Vec::<(MlClass, String)>::with_capacity(config.records);
    for sensor in &sensors {
        for class in &positives {
            for _ in 0..pos_per_class {
                assignments.push((class.clone(), sensor.clone()));
            }
        }
        for index in 0..neg_per_sensor {
            assignments.push((
                negative_classes[index % negative_classes.len()].clone(),
                sensor.clone(),
            ));
        }
    }
    deterministic_shuffle(&mut assignments, config.seed ^ 0x004d_4c74_7261_696e);

    let mut plans = assignments
        .into_iter()
        .enumerate()
        .map(|(index, (class, sensor))| {
            plan_for_exact_assignment(config, index, class, sensor, &phases)
        })
        .collect::<Vec<_>>();
    assign_exact_splits(&mut plans, config.seed ^ 0x0073_706c_6974);
    Ok(plans)
}

fn positive_classes(
    config: &MlTrainingDataConfig,
    classes: &[MlClass],
) -> Result<Vec<MlClass>, DatasetError> {
    config
        .positive_class_ids
        .iter()
        .map(|class_id| {
            classes
                .iter()
                .find(|class| class.class_id == *class_id && class.is_public_proxy_positive)
                .cloned()
                .ok_or_else(|| {
                    DatasetError::InvalidConfig(format!(
                        "unknown positive public-proxy class id {class_id}"
                    ))
                })
        })
        .collect()
}

fn nonempty_sensor_ids(config: &MlTrainingDataConfig) -> Vec<String> {
    if config.sensor_ids.is_empty() {
        vec!["generic-x-band-public-proxy".to_string()]
    } else {
        config.sensor_ids.clone()
    }
}

fn nonempty_phase_targets(config: &MlTrainingDataConfig) -> Vec<String> {
    if config.phase_targets.is_empty() {
        vec!["cruise_altitude".to_string()]
    } else {
        config.phase_targets.clone()
    }
}

fn plan_for_assignment(
    config: &MlTrainingDataConfig,
    index: usize,
    class: MlClass,
    sensors: &[String],
    phases: &[String],
) -> MlRecordPlan {
    let sensor = sensors[index % sensors.len()].clone();
    plan_for_exact_assignment(config, index, class, sensor, phases)
}

fn plan_for_exact_assignment(
    config: &MlTrainingDataConfig,
    index: usize,
    class: MlClass,
    sensor_id: String,
    phases: &[String],
) -> MlRecordPlan {
    let scenario_seed = child_seed(config.seed, index as u64);
    let object_seed = child_seed(
        stable_hash_str(&format!("{}:{sensor_id}", class.class_id)),
        scenario_seed,
    );
    MlRecordPlan {
        record_index: index,
        record_id: format!("record_{:06}", index + 1),
        sensor_id,
        phase_target: phases[index % phases.len()].clone(),
        scenario_seed,
        object_seed,
        split: SplitKind::Train,
        class,
    }
}

pub(super) fn assign_exact_splits(plans: &mut [MlRecordPlan], seed: u64) {
    let total = plans.len();
    let train = ((total as f64) * 0.70).round() as usize;
    let validation = ((total as f64) * 0.15).round() as usize;
    let train = train.min(total);
    let validation = validation.min(total.saturating_sub(train));
    let mut order = (0..total).collect::<Vec<_>>();
    deterministic_shuffle(&mut order, seed);
    for (rank, plan_index) in order.into_iter().enumerate() {
        plans[plan_index].split = if rank < train {
            SplitKind::Train
        } else if rank < train + validation {
            SplitKind::Validation
        } else {
            SplitKind::Test
        };
    }
}

// ---------------------------------------------------------------------------
// Class tables
// ---------------------------------------------------------------------------

pub(super) fn ml_classes() -> Vec<MlClass> {
    util_classes::ml_classes()
}

// ---------------------------------------------------------------------------
// CSV / tensor I/O
// ---------------------------------------------------------------------------

pub(super) fn write_csv<T: serde::Serialize>(path: &Path, rows: &[T]) -> Result<(), DatasetError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = csv::Writer::from_path(path)?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn write_f32_tensor(
    path: &Path,
    shape: &[usize],
    values: Vec<f32>,
) -> Result<(), DatasetError> {
    let expected = shape.iter().product::<usize>();
    if values.len() != expected {
        return Err(DatasetError::Tensor(format!(
            "tensor {} has {} values; expected {} for shape {:?}",
            path.display(),
            values.len(),
            expected,
            shape
        )));
    }
    let array = ArrayD::from_shape_vec(IxDyn(shape), values)
        .map_err(|err| DatasetError::Tensor(err.to_string()))?;
    echoforge_sig::tensor::write_f32(path, &array)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// DSP helpers
// ---------------------------------------------------------------------------

pub(super) fn stft_spectrogram(signal: &[f32], window: usize, hop: usize, bins: usize) -> Vec<f32> {
    let windows = if signal.len() <= window {
        1
    } else {
        ((signal.len() - window) / hop.max(1)) + 1
    };
    let mut out = Vec::with_capacity(windows * bins);
    for w in 0..windows {
        let start = w * hop.max(1);
        let slice = (0..window)
            .map(|i| {
                signal
                    .get((start + i).min(signal.len().saturating_sub(1)))
                    .copied()
                    .unwrap_or(0.0)
                    * hann(i, window)
            })
            .collect::<Vec<_>>();
        out.extend(dft_magnitude(&slice, bins));
    }
    out
}

pub(super) fn weighted_spectrum(signal: &[f32], bins: usize) -> Vec<f32> {
    let mut weighted = Vec::with_capacity(signal.len());
    for (index, value) in signal.iter().enumerate() {
        weighted.push(*value * hann(index, signal.len().max(1)));
    }
    dft_magnitude(&weighted, bins)
}

pub(super) fn cepstrum_proxy(spectrum: &[f32], bins: usize) -> Vec<f32> {
    let log_power = spectrum
        .iter()
        .map(|value| (value.max(1e-6)).ln())
        .collect::<Vec<_>>();
    dft_magnitude(&log_power, bins)
}

pub(super) fn cadence_velocity(
    spectrum: &[f32],
    velocity_mps: f32,
    cadence_bins: usize,
    velocity_bins: usize,
) -> Vec<f32> {
    let peak = spectrum.iter().copied().fold(0.0, f32::max).max(1e-6);
    let velocity_center = ((velocity_mps + 160.0) / 320.0 * (velocity_bins as f32 - 1.0))
        .clamp(0.0, velocity_bins as f32 - 1.0);
    let mut out = Vec::with_capacity(cadence_bins * velocity_bins);
    for c in 0..cadence_bins {
        let spectral = spectrum
            .get(c.min(spectrum.len().saturating_sub(1)))
            .copied()
            .unwrap_or(0.0)
            / peak;
        for v in 0..velocity_bins {
            let dist = v as f32 - velocity_center;
            out.push((spectral * (-dist * dist / 18.0).exp()).clamp(0.0, 1.0));
        }
    }
    out
}

pub(super) fn dft_magnitude(signal: &[f32], bins: usize) -> Vec<f32> {
    if signal.is_empty() {
        return vec![0.0; bins];
    }
    (0..bins)
        .map(|k| {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (n, value) in signal.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / signal.len() as f32;
                re += *value * angle.cos();
                im += *value * angle.sin();
            }
            (re * re + im * im).sqrt() / signal.len() as f32
        })
        .collect()
}

pub(super) fn entropy(values: &[f32]) -> f32 {
    let sum = values.iter().map(|value| value.max(0.0)).sum::<f32>();
    if sum <= 0.0 {
        return 0.0;
    }
    let entropy = values
        .iter()
        .map(|value| {
            let p = value.max(0.0) / sum;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum::<f32>();
    entropy / (values.len().max(2) as f32).ln()
}

pub(super) fn hann(index: usize, len: usize) -> f32 {
    if len <= 1 {
        return 1.0;
    }
    0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / (len - 1) as f32).cos()
}

// ---------------------------------------------------------------------------
// Seed / shuffle helpers
// ---------------------------------------------------------------------------

#[expect(dead_code)]
pub(super) fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

pub(super) fn stable_hash_str(input: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

pub(super) fn relative_record_path(record_id: &str, suffix: &str) -> String {
    format!("records/{record_id}/{suffix}")
}
