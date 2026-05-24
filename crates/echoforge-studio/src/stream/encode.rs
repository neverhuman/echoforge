//! Binary (de)serialisation for [`ScanFrame`] and the quantisation of
//! the heavy range-Doppler / micro-Doppler arrays.
//!
//! Wire layout of an encoded scan frame (all integers little-endian):
//!
//! ```text
//!  off  size  field
//!   0    4    magic            u32  = SCAN_FRAME_MAGIC
//!   4    4    json_len         u32
//!   8    2    rd_range_bins    u16
//!  10    2    rd_doppler_bins  u16
//!  12    4    rd_db_min        f32
//!  16    4    rd_db_max        f32
//!  20    2    spec_bins        u16
//!  22    4    spec_db_min      f32
//!  26    4    spec_db_max      f32
//!  30    4    spec_doppler_hz  f32
//!  34   json_len   ScanMeta as JSON
//!  ..   rd cells   rd_range_bins * rd_doppler_bins  u8  (row-major [doppler][range])
//!  ..   spec       spec_bins u8
//! ```

use thiserror::Error;

use super::frames::{
    MicroDopplerColumn, RangeDopplerGrid, ScanFrame, ScanMeta, SCAN_FRAME_MAGIC, SCAN_HEADER_LEN,
};

/// Display dynamic range (dB) below the per-frame peak. Cells dimmer
/// than this are clamped to the floor so a few zero cells cannot wash
/// out the heatmap.
pub const DISPLAY_DYNAMIC_RANGE_DB: f32 = 60.0;

/// Magnitude floor before the dB conversion, guarding `log10(0)`.
const MAG_FLOOR: f32 = 1e-12;

#[derive(Debug, Error)]
pub enum StreamError {
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("frame decode error: {0}")]
    Decode(String),
}

/// Convert a slice of linear magnitudes to a quantised `u8` block over a
/// `DISPLAY_DYNAMIC_RANGE_DB` window anchored at the peak. Returns
/// `(db_min, db_max, bytes)`.
fn quantize_db(magnitudes: &[f32]) -> (f32, f32, Vec<u8>) {
    let db: Vec<f32> = magnitudes
        .iter()
        .map(|&m| 20.0 * m.max(MAG_FLOOR).log10())
        .collect();
    let db_max = db
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(-300.0);
    let db_min = db_max - DISPLAY_DYNAMIC_RANGE_DB;
    let span = (db_max - db_min).max(f32::EPSILON);
    let bytes = db
        .iter()
        .map(|&v| {
            let norm = ((v - db_min) / span).clamp(0.0, 1.0);
            (norm * 255.0).round() as u8
        })
        .collect();
    (db_min, db_max, bytes)
}

/// Downsample (max-pool) and quantise a `[doppler][range]` magnitude
/// proxy into a [`RangeDopplerGrid`]. Max-pooling preserves peak cells —
/// critical for keeping CFAR detections visible after downsampling.
pub fn quantize_rd(proxy: &[Vec<f32>], target_range_bins: usize) -> RangeDopplerGrid {
    let doppler_bins = proxy.len();
    let raw_range = proxy.first().map(|row| row.len()).unwrap_or(0);
    if doppler_bins == 0 || raw_range == 0 {
        return RangeDopplerGrid {
            range_bins: 0,
            doppler_bins: 0,
            db_min: 0.0,
            db_max: 0.0,
            cells: Vec::new(),
        };
    }
    let range_bins = target_range_bins.clamp(1, raw_range);
    let mut pooled: Vec<f32> = Vec::with_capacity(doppler_bins * range_bins);
    for row in proxy {
        for out in 0..range_bins {
            let lo = out * raw_range / range_bins;
            let hi = ((out + 1) * raw_range / range_bins)
                .max(lo + 1)
                .min(raw_range);
            let peak = row[lo..hi].iter().copied().fold(0.0_f32, f32::max);
            pooled.push(peak);
        }
    }
    let (db_min, db_max, cells) = quantize_db(&pooled);
    RangeDopplerGrid {
        range_bins,
        doppler_bins,
        db_min,
        db_max,
        cells,
    }
}

/// Linearly resample to `target_bins`, dB-convert and quantise one
/// Doppler spectrum into a [`MicroDopplerColumn`]. The caller supplies
/// the spectrum already arranged on the desired axis (the mapping layer
/// fftshifts so zero Doppler is centred).
pub fn quantize_column(
    spectrum: &[f32],
    doppler_max_hz: f64,
    target_bins: usize,
) -> MicroDopplerColumn {
    let bins = target_bins.max(1);
    let resampled: Vec<f32> = if spectrum.is_empty() {
        vec![0.0; bins]
    } else if spectrum.len() == 1 {
        vec![spectrum[0]; bins]
    } else {
        (0..bins)
            .map(|i| {
                let pos = i as f32 * (spectrum.len() - 1) as f32 / (bins - 1).max(1) as f32;
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(spectrum.len() - 1);
                let frac = pos - lo as f32;
                spectrum[lo] * (1.0 - frac) + spectrum[hi] * frac
            })
            .collect()
    };
    let (db_min, db_max, column) = quantize_db(&resampled);
    MicroDopplerColumn {
        bins,
        db_min,
        db_max,
        doppler_max_hz,
        column,
    }
}

/// Encode a [`ScanFrame`] into its binary wire representation.
pub fn encode_scan_frame(frame: &ScanFrame) -> Result<Vec<u8>, StreamError> {
    let json = serde_json::to_vec(&frame.meta)?;
    let rd = &frame.range_doppler;
    let spec = &frame.micro_doppler;
    let mut buf =
        Vec::with_capacity(SCAN_HEADER_LEN + json.len() + rd.cells.len() + spec.column.len());
    buf.extend_from_slice(&SCAN_FRAME_MAGIC.to_le_bytes());
    buf.extend_from_slice(&(json.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(rd.range_bins as u16).to_le_bytes());
    buf.extend_from_slice(&(rd.doppler_bins as u16).to_le_bytes());
    buf.extend_from_slice(&rd.db_min.to_le_bytes());
    buf.extend_from_slice(&rd.db_max.to_le_bytes());
    buf.extend_from_slice(&(spec.bins as u16).to_le_bytes());
    buf.extend_from_slice(&spec.db_min.to_le_bytes());
    buf.extend_from_slice(&spec.db_max.to_le_bytes());
    buf.extend_from_slice(&(spec.doppler_max_hz as f32).to_le_bytes());
    debug_assert_eq!(buf.len(), SCAN_HEADER_LEN);
    buf.extend_from_slice(&json);
    buf.extend_from_slice(&rd.cells);
    buf.extend_from_slice(&spec.column);
    Ok(buf)
}

fn read_u32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn read_u16(bytes: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([bytes[off], bytes[off + 1]])
}

fn read_f32(bytes: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

/// Decode a binary scan frame. Mirrors [`encode_scan_frame`] — used by
/// the integration tests and any Rust consumer of the stream.
pub fn decode_scan_frame(bytes: &[u8]) -> Result<ScanFrame, StreamError> {
    if bytes.len() < SCAN_HEADER_LEN {
        return Err(StreamError::Decode(format!(
            "buffer too short: {} < {SCAN_HEADER_LEN}",
            bytes.len()
        )));
    }
    let magic = read_u32(bytes, 0);
    if magic != SCAN_FRAME_MAGIC {
        return Err(StreamError::Decode(format!("bad magic: {magic:#x}")));
    }
    let json_len = read_u32(bytes, 4) as usize;
    let rd_range = read_u16(bytes, 8) as usize;
    let rd_doppler = read_u16(bytes, 10) as usize;
    let rd_db_min = read_f32(bytes, 12);
    let rd_db_max = read_f32(bytes, 16);
    let spec_bins = read_u16(bytes, 20) as usize;
    let spec_db_min = read_f32(bytes, 22);
    let spec_db_max = read_f32(bytes, 26);
    let spec_doppler_hz = read_f32(bytes, 30) as f64;

    let rd_len = rd_range * rd_doppler;
    let total = SCAN_HEADER_LEN + json_len + rd_len + spec_bins;
    if bytes.len() < total {
        return Err(StreamError::Decode(format!(
            "buffer too short for payload: {} < {total}",
            bytes.len()
        )));
    }
    let json_start = SCAN_HEADER_LEN;
    let rd_start = json_start + json_len;
    let spec_start = rd_start + rd_len;
    let meta: ScanMeta = serde_json::from_slice(&bytes[json_start..rd_start])?;
    let cells = bytes[rd_start..spec_start].to_vec();
    let column = bytes[spec_start..spec_start + spec_bins].to_vec();

    Ok(ScanFrame {
        meta,
        range_doppler: RangeDopplerGrid {
            range_bins: rd_range,
            doppler_bins: rd_doppler,
            db_min: rd_db_min,
            db_max: rd_db_max,
            cells,
        },
        micro_doppler: MicroDopplerColumn {
            bins: spec_bins,
            db_min: spec_db_min,
            db_max: spec_db_max,
            doppler_max_hz: spec_doppler_hz,
            column,
        },
    })
}
