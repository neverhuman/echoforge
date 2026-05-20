//! Replay driver.
//!
//! The in-repo bundle fixtures are *thin* — a `manifest.json` plus QA
//! JSON, with no frame tensors. Rather than leave the console blank,
//! replay streams a deterministic re-synthesis whose seed is derived
//! from the bundle digest, so replaying bundle A and bundle B yield
//! distinct, reproducible streams. A `Status` frame states this plainly
//! so nothing is misrepresented as recorded data.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::mpsc;

use super::control::{ControlCommand, ScenarioSpec};
use super::engine::SimEngine;
use super::frames::{ControlFrame, OutboundFrame, StatusFrame};
use super::live::{drive, DriverSource};

/// FNV-1a 64-bit digest.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Read a bundle's `manifest.json` and return `(digest, display_name)`.
fn load_bundle_digest(path: &Path) -> Result<(u64, String), String> {
    let manifest = if path.is_dir() {
        path.join("manifest.json")
    } else {
        path.to_path_buf()
    };
    let bytes = std::fs::read(&manifest).map_err(|e| e.to_string())?;
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "bundle".to_string());
    Ok((fnv1a(&bytes), name))
}

/// Replay driver — streams a deterministic re-synthesis seeded from the
/// requested bundle.
pub(super) async fn run_replay(
    engine: Arc<SimEngine>,
    scenario: ScenarioSpec,
    bundle_path: PathBuf,
    session_id: u64,
    ctrl: mpsc::Receiver<ControlCommand>,
) {
    let (seed_base, startup_message) = match load_bundle_digest(&bundle_path) {
        Ok((digest, name)) => (
            digest,
            format!(
                "replay: bundle '{name}' loaded — thin QA bundle carries no \
                 frame tensors; streaming a deterministic re-synthesis seeded \
                 from the bundle digest"
            ),
        ),
        Err(err) => {
            engine.broadcast(OutboundFrame::Control(ControlFrame::Status(
                StatusFrame::warn(
                    "replay_unavailable",
                    format!(
                        "bundle '{}' could not be read ({err}); streaming the \
                         fallback scenario instead",
                        bundle_path.display()
                    ),
                ),
            )));
            (
                scenario.base_seed,
                format!("replay fallback: {}", scenario.label),
            )
        }
    };
    let source = DriverSource {
        label: "replay",
        seed_base,
        startup_message,
    };
    drive(engine, scenario, session_id, ctrl, source).await;
}
