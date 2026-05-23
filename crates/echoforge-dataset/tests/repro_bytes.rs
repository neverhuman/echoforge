//! Byte-stability lane for the Monte-Carlo demo output tree.
//!
//! Runs `run_monte_carlo_demo` twice into two separate `tempfile::TempDir`
//! roots with identical config (preset / seed / generated_at / episodes /
//! pulse_count / sample_rate) and walks both trees comparing the sha256
//! of every file. Any mismatch (missing file on either side or differing
//! bytes) fails the test.
//!
//! This is the Rust-side enforcement for the `repro-bytes-lane`
//! packet. The shell-side companion (`tools/repro_bytes.mjs`, wired into
//! `just repro`) does the same check against the released CLI binary so a
//! regression caught here also fails the CI lane.
//!
//! Wallclock-derivative files (`benchmark_report.md`,
//! `benchmark_report.json`) are intentionally excluded — they carry
//! stage timings and throughput measurements that vary across runs by
//! design. The intent of byte-stability is "the dataset content is
//! deterministic", not "the timing instrumentation is deterministic".
//! Keep the allowlist tight; any new wallclock file must be added here
//! explicitly with a comment justifying why it is not bit-stable.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use echoforge_dataset::{run_monte_carlo_demo, MonteCarloDemoConfig};
use sha2::{Digest, Sha256};

/// Episodes used by the byte-stability test. Kept small so the test stays
/// fast (each episode writes ~7 product files + JSON metadata); large
/// enough that worker chunking, multiple split assignments, and the
/// leakage report all exercise non-trivial code paths.
const EPISODES: usize = 4;
/// Pulse count used by the byte-stability test. Trades resolution for
/// runtime; 8 is the same value used by the existing determinism test in
/// `monte_carlo.rs` (`tempdir_generation_writes_required_files_and_models_validate`).
const PULSE_COUNT: usize = 8;

/// Files whose content is wallclock-derivative by design. These are
/// excluded from the byte-equality check; presence/absence of each entry
/// is still compared (so a file dropping out of the output tree across
/// runs is still a regression).
const WALLCLOCK_FILES: &[&str] = &[
    // Stage timings + throughput, formatted markdown.
    "benchmark_report.md",
    // Stage timings + throughput, JSON. Only emitted when `--benchmark`
    // is set; harmless to list unconditionally.
    "benchmark_report.json",
];

#[test]
fn dataset_tree_is_byte_stable_across_reruns() {
    let tmp_dir = tempfile::tempdir().expect("tempdir");
    let dir_a = tmp_dir.path().join("run-a");
    let dir_b = tmp_dir.path().join("run-b");

    let config_a = make_config(dir_a.clone());
    let config_b = make_config(dir_b.clone());

    run_monte_carlo_demo(config_a).expect("first run completes");
    run_monte_carlo_demo(config_b).expect("second run completes");

    let hashes_a = hash_tree(&dir_a);
    let hashes_b = hash_tree(&dir_b);

    let only_in_a: Vec<&String> = hashes_a
        .keys()
        .filter(|k| !hashes_b.contains_key(*k))
        .collect();
    let only_in_b: Vec<&String> = hashes_b
        .keys()
        .filter(|k| !hashes_a.contains_key(*k))
        .collect();
    assert!(
        only_in_a.is_empty() && only_in_b.is_empty(),
        "demo output tree file set drifted between reruns: only_in_a={only_in_a:?}, only_in_b={only_in_b:?}"
    );

    let mut mismatches = Vec::new();
    let mut compared = 0usize;
    for (rel, hash_a) in &hashes_a {
        if WALLCLOCK_FILES.contains(&rel.as_str()) {
            continue;
        }
        let hash_b = hashes_b
            .get(rel)
            .expect("relative path present in both trees (set equality checked above)");
        compared += 1;
        if hash_a != hash_b {
            mismatches.push((rel.clone(), hash_a.clone(), hash_b.clone()));
        }
    }
    assert!(
        mismatches.is_empty(),
        "demo output tree bytes drifted between reruns: {} mismatch(es) across {} compared file(s): {:?}",
        mismatches.len(),
        compared,
        mismatches
    );

    assert!(
        compared > 0,
        "expected at least one non-wallclock file in the demo output tree; got 0 compared (total tree size {})",
        hashes_a.len()
    );
}

fn make_config(out: PathBuf) -> MonteCarloDemoConfig {
    let mut cfg = MonteCarloDemoConfig::low_altitude_fixed_wing_default(out);
    cfg.episodes = EPISODES;
    cfg.pulse_count = PULSE_COUNT;
    // Pin seed + timestamp so both runs share every reproducibility input.
    cfg.seed = 20_260_518;
    cfg.generated_at = "2026-05-18T00:00:00Z".to_string();
    cfg
}

/// Walks `root` recursively and returns `relative_path -> sha256_hex`
/// for every regular file. Paths are normalized to forward slashes so
/// the comparison is platform-stable.
fn hash_tree(root: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    visit(root, root, &mut out);
    out
}

fn visit(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|err| {
        panic!("read_dir({}) failed: {err}", dir.display());
    });
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let file_type = entry.file_type().expect("file_type");
        if file_type.is_dir() {
            visit(root, &path, out);
        } else if file_type.is_file() {
            let bytes = fs::read(&path).unwrap_or_else(|err| {
                panic!("read({}) failed: {err}", path.display());
            });
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            let digest = hasher.finalize();
            let rel = path
                .strip_prefix(root)
                .expect("path under root")
                .to_string_lossy()
                .replace('\\', "/");
            out.insert(rel, hex_encode(&digest));
        }
        // Symlinks / other types are ignored — the demo writer never
        // creates them, so seeing one would itself be a regression.
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}
