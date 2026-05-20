//! Translation of a radar [`SyntheticEpisode`] into a wire [`ScanFrame`],
//! plus a lightweight nearest-neighbour track associator.

use echoforge_radar::SyntheticEpisode;

use super::control::ScenarioSpec;
use super::encode::{quantize_column, quantize_rd};
use super::frames::{PpiBlip, RdDetection, ScanFrame, ScanMeta, Telemetry, TrackRow};

const C_M_PER_S: f64 = 299_792_458.0;

/// Range/Doppler axis calibration derived from a synthesised episode.
pub struct FrameGeometry {
    pub sample_count: usize,
    pub zero_delay_bin: usize,
    pub doppler_bins: usize,
    pub range_max_m: f64,
    pub doppler_max_hz: f64,
    carrier_hz: f64,
    pri_s: f64,
    sample_rate_hz: f64,
}

impl FrameGeometry {
    pub fn from_episode(ep: &SyntheticEpisode) -> Self {
        let compressed_len = ep.range_doppler_proxy.first().map(|r| r.len()).unwrap_or(0);
        let sample_count = compressed_len.div_ceil(2);
        let zero_delay_bin = sample_count.saturating_sub(1);
        let doppler_bins = ep.range_doppler_proxy.len();
        let sample_rate_hz = ep.config.sample_rate_hz;
        let pri_s = ep.config.pri_s.max(1e-9);
        let range_max_m =
            (sample_count.saturating_sub(1)) as f64 * C_M_PER_S / (2.0 * sample_rate_hz);
        Self {
            sample_count,
            zero_delay_bin,
            doppler_bins,
            range_max_m,
            doppler_max_hz: 1.0 / (2.0 * pri_s),
            carrier_hz: ep.config.carrier_hz.max(1.0),
            pri_s,
            sample_rate_hz,
        }
    }

    fn cropped_bin_for_range(&self, range_m: f64) -> usize {
        let step = C_M_PER_S / (2.0 * self.sample_rate_hz);
        ((range_m / step).round() as usize).min(self.sample_count.saturating_sub(1))
    }

    /// Radial velocity (m/s) for an fftshifted Doppler-bin index.
    fn velocity_for_shifted_bin(&self, bin: usize) -> f64 {
        let n = self.doppler_bins.max(1) as f64;
        let centred = bin as f64 - (self.doppler_bins / 2) as f64;
        let f_doppler = centred / (n * self.pri_s);
        f_doppler * C_M_PER_S / (2.0 * self.carrier_hz)
    }
}

/// fftshift the Doppler (outer) dimension and crop to positive range.
fn shift_and_crop(proxy: &[Vec<f32>], zero_delay_bin: usize) -> Vec<Vec<f32>> {
    let n = proxy.len();
    if n == 0 {
        return Vec::new();
    }
    let half = n / 2;
    (0..n)
        .map(|i| {
            let row = &proxy[(i + half) % n];
            row.get(zero_delay_bin..)
                .map(|s| s.to_vec())
                .unwrap_or_default()
        })
        .collect()
}

fn db(value: f64) -> f64 {
    10.0 * value.max(1e-30).log10()
}

/// Per-frame inputs that the episode itself does not carry.
pub struct FrameContext<'a> {
    pub scenario: &'a ScenarioSpec,
    pub frame_index: u64,
    pub sim_time_s: f64,
    pub wall_time_ms: u64,
    pub beam_azimuth_deg: f64,
    pub rd_range_bins: usize,
    pub spectrogram_bins: usize,
    pub frame_compute_ms: f64,
    pub scan_rate_hz: f64,
}

/// Build a [`ScanFrame`] from a synthesised episode.
pub fn episode_to_scan(
    episode: &SyntheticEpisode,
    ctx: &FrameContext,
    tracker: &mut TrackBook,
) -> ScanFrame {
    let geom = FrameGeometry::from_episode(episode);
    let shifted = shift_and_crop(&episode.range_doppler_proxy, geom.zero_delay_bin);
    let range_doppler = quantize_rd(&shifted, ctx.rd_range_bins);

    // PPI blips — one per scene entity at truth geometry.
    let antenna_alt = episode.config.radar_altitude_agl_m;
    let mut ppi = Vec::with_capacity(ctx.scenario.entities.len());
    for (idx, entity) in ctx.scenario.entities.iter().enumerate() {
        let state =
            entity
                .kinematics
                .state_at(ctx.sim_time_s, entity.display_initial_range_m, antenna_alt);
        let snr_db = episode
            .per_target_snr_db
            .get(idx)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let detected = episode
            .detections
            .iter()
            .any(|d| (d.range_m - state.range_m).abs() < (150.0_f64).max(0.03 * state.range_m));
        ppi.push(PpiBlip {
            entity_id: idx,
            range_m: state.range_m,
            azimuth_deg: entity.bearing_deg,
            amplitude_db: if snr_db.is_finite() { snr_db } else { -120.0 },
            snr_db: if snr_db.is_finite() { snr_db } else { -120.0 },
            detected,
            class_label: entity.label.clone(),
        });
    }

    // CFAR detections placed on the range-Doppler plane.
    let mut detections = Vec::with_capacity(episode.detections.len());
    let mut assoc_inputs = Vec::with_capacity(episode.detections.len());
    for d in &episode.detections {
        if d.range_bin < geom.zero_delay_bin {
            continue;
        }
        let cropped = d.range_bin - geom.zero_delay_bin;
        let (doppler_bin, peak) = shifted
            .iter()
            .enumerate()
            .map(|(i, row)| (i, row.get(cropped).copied().unwrap_or(0.0)))
            .fold((0usize, 0.0f32), |a, b| if b.1 > a.1 { b } else { a });
        let display_bin = if geom.sample_count > 0 {
            (cropped * ctx.rd_range_bins / geom.sample_count)
                .min(ctx.rd_range_bins.saturating_sub(1))
        } else {
            0
        };
        let snr_db = if d.noise_estimate > 0.0 {
            db((d.statistic / d.noise_estimate) as f64)
        } else {
            0.0
        };
        let velocity = geom.velocity_for_shifted_bin(doppler_bin);
        detections.push(RdDetection {
            range_bin: display_bin,
            range_m: d.range_m,
            doppler_bin,
            magnitude_db: 20.0 * (peak.max(1e-12)).log10() as f64,
            snr_db,
        });
        assoc_inputs.push(DetectionInput {
            range_m: d.range_m,
            velocity_mps: velocity,
            snr_db,
            confidence: d.confidence,
        });
    }

    // Micro-Doppler column at the brightest range gate.
    let primary_bin = episode
        .detections
        .iter()
        .max_by(|a, b| a.statistic.total_cmp(&b.statistic))
        .map(|d| d.range_bin.saturating_sub(geom.zero_delay_bin))
        .unwrap_or_else(|| {
            let r = ppi.first().map(|b| b.range_m).unwrap_or(0.0);
            geom.cropped_bin_for_range(r)
        })
        .min(geom.sample_count.saturating_sub(1));
    let column: Vec<f32> = shifted
        .iter()
        .map(|row| row.get(primary_bin).copied().unwrap_or(0.0))
        .collect();
    let micro_doppler = quantize_column(&column, geom.doppler_max_hz, ctx.spectrogram_bins);

    // Tracks.
    let tracks = tracker.update(&assoc_inputs, ctx.scenario, ctx.beam_azimuth_deg);

    let lb = &episode.diagnostic_link_budget;
    let telemetry = Telemetry {
        snr_db: lb.snr_db,
        received_power_dbw: db(lb.received_power_w),
        noise_power_dbw: db(lb.noise_power_w),
        free_space_path_loss_db: lb.free_space_path_loss_db,
        atmospheric_loss_db: lb.atmospheric_loss_db,
        rain_loss_db: lb.rain_loss_db,
        propagation_factor_db: lb.propagation_factor_db,
        coherent_integration_gain_db: lb.coherent_integration_gain_db,
        above_horizon: lb.above_horizon,
        detections_this_frame: episode.detections.len(),
        frame_compute_ms: ctx.frame_compute_ms,
        scan_rate_hz: ctx.scan_rate_hz,
    };

    ScanFrame {
        meta: ScanMeta {
            frame_index: ctx.frame_index,
            sim_time_s: ctx.sim_time_s,
            wall_time_ms: ctx.wall_time_ms,
            beam_azimuth_deg: ctx.beam_azimuth_deg,
            ppi,
            detections,
            tracks,
            telemetry,
        },
        range_doppler,
        micro_doppler,
    }
}

/// One detection fed to the [`TrackBook`].
pub struct DetectionInput {
    pub range_m: f64,
    pub velocity_mps: f64,
    pub snr_db: f64,
    pub confidence: f32,
}

struct LiveTrack {
    id: usize,
    range_m: f64,
    velocity_mps: f64,
    snr_db: f64,
    confidence: f32,
    age: u32,
    misses: u32,
}

/// A nearest-neighbour, range-gated track-while-scan associator. Real
/// frame-to-frame association — the radar crate's fusion adapter only
/// labels detections within a single frame.
pub struct TrackBook {
    tracks: Vec<LiveTrack>,
    next_id: usize,
}

impl Default for TrackBook {
    fn default() -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
        }
    }
}

impl TrackBook {
    const RANGE_GATE_M: f64 = 280.0;
    const MAX_MISSES: u32 = 4;
    const SMOOTH: f64 = 0.45;

    pub fn update(
        &mut self,
        detections: &[DetectionInput],
        scenario: &ScenarioSpec,
        beam_azimuth_deg: f64,
    ) -> Vec<TrackRow> {
        let mut matched = vec![false; self.tracks.len()];
        let mut used = vec![false; detections.len()];
        for (di, det) in detections.iter().enumerate() {
            let best = self
                .tracks
                .iter()
                .enumerate()
                .filter(|(ti, _)| !matched[*ti])
                .map(|(ti, t)| (ti, (t.range_m - det.range_m).abs()))
                .filter(|(_, dist)| *dist < Self::RANGE_GATE_M)
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((ti, _)) = best {
                let t = &mut self.tracks[ti];
                t.range_m += Self::SMOOTH * (det.range_m - t.range_m);
                t.velocity_mps += Self::SMOOTH * (det.velocity_mps - t.velocity_mps);
                t.snr_db = det.snr_db;
                t.confidence = det.confidence;
                t.age += 1;
                t.misses = 0;
                matched[ti] = true;
                used[di] = true;
            }
        }
        for (ti, t) in self.tracks.iter_mut().enumerate() {
            if !matched[ti] {
                t.misses += 1;
            }
        }
        self.tracks.retain(|t| t.misses <= Self::MAX_MISSES);
        for (di, det) in detections.iter().enumerate() {
            if used[di] {
                continue;
            }
            self.tracks.push(LiveTrack {
                id: self.next_id,
                range_m: det.range_m,
                velocity_mps: det.velocity_mps,
                snr_db: det.snr_db,
                confidence: det.confidence,
                age: 1,
                misses: 0,
            });
            self.next_id += 1;
        }
        self.tracks
            .iter()
            .map(|t| {
                let (label, bearing) = nearest_entity(scenario, t.range_m, beam_azimuth_deg);
                TrackRow {
                    track_id: t.id,
                    range_m: t.range_m,
                    azimuth_deg: bearing,
                    radial_velocity_mps: t.velocity_mps,
                    snr_db: t.snr_db,
                    confidence: t.confidence,
                    class_label: label,
                    age_frames: t.age,
                }
            })
            .collect()
    }
}

/// Associate a track range with the nearest scenario entity for a class
/// label and bearing; falls back to the scan azimuth when no entity is
/// close.
fn nearest_entity(scenario: &ScenarioSpec, range_m: f64, beam_azimuth_deg: f64) -> (String, f64) {
    scenario
        .entities
        .iter()
        .map(|e| (e, (e.display_initial_range_m - range_m).abs()))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .filter(|(_, dist)| *dist < 2500.0)
        .map(|(e, _)| (e.label.clone(), e.bearing_deg))
        .unwrap_or_else(|| ("unresolved".to_string(), beam_azimuth_deg))
}
