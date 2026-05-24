from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from detection.detection_common_helpers import slice_metrics
from detection.ml_training_v2_config import (
    FRAME_COUNT,
    FRAME_INDEX,
    MICRO_DOPPLER_BANDWIDTH_PROXY_CLAMP_HZ,
    MICRO_DOPPLER_PEAK_PROXY_CLAMP_HZ,
)
from detection.ml_training_v2_diagnostics import (
    aggregate_frame_features,
    micro_doppler_saturation_guard,
)
from detection.ml_training_v2_generators import build_strata, simulate_record
from detection.ml_training_v2_real_priors import adjust_range, knob_prior
from detection.real_data.adapters import build_dataset_report, dry_run_fetch
from detection.real_data.catalog import load_catalog, validate_catalog
from detection.real_data.feature_policy import observable_only_violations
from detection.real_data.git_guard import find_tracked_real_data
from detection.real_data.realism_gate import evaluate_realism_gate


CATALOG = Path(__file__).resolve().parents[1] / "catalog.json"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _catalog_entry(dataset_id: str) -> dict[str, object]:
    for entry in load_catalog(CATALOG)["datasets"]:
        if entry["dataset_id"] == dataset_id:
            return dict(entry)
    raise AssertionError(f"missing catalog entry: {dataset_id}")


class RealDataCatalogTests(unittest.TestCase):
    def test_catalog_has_required_fields(self) -> None:
        catalog = load_catalog(CATALOG)
        errors = validate_catalog(catalog)
        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(catalog["datasets"]), 10)

    def test_dry_run_has_no_side_effects(self) -> None:
        entry = load_catalog(CATALOG)["datasets"][0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = dry_run_fetch(entry, root)
            self.assertEqual(payload["network_side_effects"], "none")
            self.assertEqual(payload["status"], "manual_required")
            self.assertEqual(list(root.iterdir()), [])

    def test_missing_dataset_emits_reference_only_report(self) -> None:
        entry = load_catalog(CATALOG)["datasets"][0]
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            out_root = Path(tmp) / "out"
            report = build_dataset_report(
                entry, raw_root=raw_root, out_root=out_root, run_id="unit"
            )
            self.assertEqual(report.status, "reference_only")
            manifest = json.loads((report.output_dir / "anchor_manifest.json").read_text())
            self.assertEqual(manifest["calibration_anchor_status"], "reference_only")
            with (report.output_dir / "calibration_distance.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue(all(row["status"] == "reference_only" for row in rows))

    def test_local_observations_become_candidate_without_copying_traces(self) -> None:
        entry = dict(load_catalog(CATALOG)["datasets"][0])
        entry["reference_feature_bounds"] = {"snr_db": {"mean_min": 8.0, "mean_max": 12.0}}
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            dataset_root = raw_root / entry["dataset_id"]
            dataset_root.mkdir(parents=True)
            (dataset_root / "observations.csv").write_text(
                "feature,value\nsnr_db,9.0\nsnr_db,11.0\n", encoding="utf-8"
            )
            report = build_dataset_report(
                entry, raw_root=raw_root, out_root=Path(tmp) / "out", run_id="unit"
            )
            self.assertEqual(report.status, "measured_anchor_candidate")
            distribution = json.loads(
                (report.output_dir / "feature_distributions.json").read_text()
            )
            self.assertEqual(distribution["features"]["snr_db"]["count"], 2.0)
            with (report.output_dir / "calibration_distance.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["status"], "candidate_pass")

    def test_kth_adapter_normalizes_micro_doppler_anchor_rows(self) -> None:
        entry = _catalog_entry("kth-drone-bird-human-77ghz")
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            dataset_root = raw_root / str(entry["dataset_id"])
            dataset_root.mkdir(parents=True)
            (dataset_root / "kth_observations.csv").write_text(
                "\n".join(
                    [
                        "sample_id,class_family,micro_doppler_peak_hz,doppler_bandwidth_hz,micro_doppler_energy,spectral_entropy",
                        "raw-sample-1,drone,122,61,0.72,0.41",
                        "raw-sample-2,bird,18,92,0.50,0.77",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            report = build_dataset_report(
                entry, raw_root=raw_root, out_root=Path(tmp) / "out", run_id="unit"
            )
            self.assertEqual(report.status, "measured_anchor_candidate")
            with (report.output_dir / "anchor_observations.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertNotIn("raw-sample-1", {row["sample_id_hash"] for row in rows})
            self.assertTrue(
                {row["split_role"] for row in rows} <= {"calibration", "benchmark", "holdout"}
            )
            coefficients = json.loads(
                (report.output_dir / "calibration_coefficients.json").read_text()
            )
            self.assertEqual(coefficients["simulator_priors"], {})
            self.assertIn("bird_flock", coefficients["simulator_priors_by_family"])
            self.assertIn("single_bird", coefficients["simulator_priors_by_family"])
            self.assertIn("rc_fixed_wing", coefficients["simulator_priors_by_family"])
            self.assertNotIn("public_proxy_fixed_wing", coefficients["simulator_priors_by_family"])
            self.assertIn("micro_bw", coefficients["simulator_priors_by_family"]["bird_flock"])

    def test_kth_raw_npy_extracts_segment_features_and_preserves_split_edge(self) -> None:
        entry = _catalog_entry("kth-drone-bird-human-77ghz")
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            dataset_root = raw_root / str(entry["dataset_id"])
            dataset_root.mkdir(parents=True)
            t = np.arange(256, dtype=np.float64) / 17_000.0
            tone_a = np.exp(1j * 2.0 * np.pi * 320.0 * t)
            tone_b = np.exp(1j * 2.0 * np.pi * 640.0 * t)
            seg_a = np.tile(tone_a, (5, 1)).reshape(-1)
            seg_b = np.tile(tone_b, (5, 1)).reshape(-1)
            fixture = np.empty((2, 6), dtype=object)
            fixture[0] = [
                "D1",
                np.stack([seg_a, seg_b]),
                np.asarray([120.0, 121.0]),
                np.asarray([0.0, 0.1]),
                np.asarray([1, 2]),
                np.asarray([False, True]),
            ]
            fixture[1] = [
                "seagull",
                np.stack([seg_b, seg_a]),
                np.asarray([45.0, 46.0]),
                np.asarray([0.0, 0.1]),
                np.asarray([3, 1]),
                np.asarray([False, False]),
            ]
            np.save(dataset_root / "data_SAAB_SIRS_77GHz_FMCW.npy", fixture)

            report = build_dataset_report(
                entry, raw_root=raw_root, out_root=Path(tmp) / "out", run_id="unit"
            )

            self.assertEqual(report.status, "measured_anchor_candidate")
            with (report.output_dir / "anchor_observations.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 32)
            self.assertEqual({row["class_family"] for row in rows}, {"bird", "drone"})
            self.assertEqual(
                {row["split_role"] for row in rows},
                {"calibration", "benchmark", "holdout"},
            )
            self.assertIn("edge_truncated", {row["observable_name"] for row in rows})
            self.assertTrue(
                any(
                    row["observable_name"] == "micro_doppler_peak_hz"
                    and float(row["value"]) > 250.0
                    for row in rows
                )
            )
            shape_report = json.loads((report.output_dir / "raw_shape_report.json").read_text())
            self.assertEqual(shape_report["top_level_shape"], [2, 6])
            self.assertEqual(shape_report["segment_count"], 4)
            distribution = json.loads(
                (report.output_dir / "feature_distributions.json").read_text()
            )
            self.assertIn("edge", distribution["by_edge_truncation"])
            self.assertIn("non_edge", distribution["by_edge_truncation"])
            coefficients = json.loads(
                (report.output_dir / "calibration_coefficients.json").read_text()
            )
            self.assertEqual(coefficients["simulator_priors"], {})
            self.assertIn("dropout", coefficients["simulator_priors_by_family"]["bird_flock"])

    def test_kth_compare_only_rows_do_not_tune_simulator_priors(self) -> None:
        entry = _catalog_entry("kth-drone-bird-human-77ghz")
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            dataset_root = raw_root / str(entry["dataset_id"])
            dataset_root.mkdir(parents=True)
            (dataset_root / "kth_observations.csv").write_text(
                "\n".join(
                    [
                        "sample_id,class_family,range_m,return_power_db,micro_doppler_peak_hz,micro_doppler_energy",
                        "cr-1,calibration_reflector,12.0,88.0,900.0,999999.0",
                        "human-1,human,18.0,55.0,420.0,777777.0",
                        "bird-1,bird,21.0,30.0,18.0,1000000.0",
                        "bird-2,bird,22.0,31.0,24.0,2000000.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_dataset_report(
                entry, raw_root=raw_root, out_root=Path(tmp) / "out", run_id="unit"
            )
            coefficients = json.loads(
                (report.output_dir / "calibration_coefficients.json").read_text()
            )

            notes = coefficients["applicability_notes"]
            self.assertEqual(notes["observable_applicability"]["range_m"], "compare_only")
            self.assertEqual(notes["observable_applicability"]["return_power_db"], "compare_only")
            self.assertEqual(notes["class_applicability"]["human"], "compare_only")
            self.assertEqual(notes["class_applicability"]["calibration_reflector"], "compare_only")
            self.assertEqual(coefficients["simulator_priors"], {})
            families = coefficients["simulator_priors_by_family"]
            self.assertEqual(set(families), {"bird_flock", "single_bird"})
            for family, priors in families.items():
                self.assertNotIn("range_m", priors)
                self.assertNotIn("return_power_db", priors)
                micro_amp = priors["micro_amp"]
                lower, upper = micro_amp["synthetic_bounds"]
                self.assertGreaterEqual(micro_amp["target_q10"], lower)
                self.assertLessEqual(micro_amp["target_q90"], upper)
                self.assertEqual(
                    micro_amp["raw_magnitude_policy"], "rank_shape_only_not_raw_energy"
                )

    def test_ori_adapter_normalizes_track_anchor_rows(self) -> None:
        entry = _catalog_entry("open-radar-initiative-outdoor-moving-object")
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            dataset_root = raw_root / str(entry["dataset_id"])
            dataset_root.mkdir(parents=True)
            (dataset_root / "ori_tracks.csv").write_text(
                "\n".join(
                    [
                        "track_id,object_class,range_m,azimuth_deg,radial_velocity_mps,snr_db,track_gap_fraction,clutter_score",
                        "track-1,uav,800,12,14,18,0.04,0.21",
                        "track-2,person,120,3,1.5,7,0.18,0.36",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            report = build_dataset_report(
                entry, raw_root=raw_root, out_root=Path(tmp) / "out", run_id="unit"
            )
            distribution = json.loads(
                (report.output_dir / "feature_distributions.json").read_text()
            )
            self.assertEqual(distribution["features"]["snr_db"]["count"], 2.0)
            coefficients = json.loads(
                (report.output_dir / "calibration_coefficients.json").read_text()
            )
            self.assertIn("snr_center", coefficients["simulator_priors"])
            self.assertIn("dropout", coefficients["simulator_priors"])
            self.assertIn("glint", coefficients["simulator_priors"])

    def test_license_gated_dataset_stays_reference_only_without_marker(self) -> None:
        entry = _catalog_entry("han-jung-2026-timesync-drone-radar-rf")
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            dataset_root = raw_root / str(entry["dataset_id"])
            dataset_root.mkdir(parents=True)
            (dataset_root / "observations.csv").write_text(
                "feature,value\nsnr_db,30\n", encoding="utf-8"
            )
            report = build_dataset_report(
                entry, raw_root=raw_root, out_root=Path(tmp) / "out", run_id="unit"
            )
            self.assertEqual(report.status, "license_gated_reference_only")
            with (report.output_dir / "anchor_observations.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [])

    def test_gap_report_marks_shifted_distribution(self) -> None:
        entry = _catalog_entry("kth-drone-bird-human-77ghz")
        entry["reference_feature_bounds"] = {"snr_db": {"mean_min": 0.0, "mean_max": 1.0}}
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            dataset_root = raw_root / str(entry["dataset_id"])
            dataset_root.mkdir(parents=True)
            (dataset_root / "observations.csv").write_text(
                "feature,value\nsnr_db,9.0\nsnr_db,11.0\n", encoding="utf-8"
            )
            report = build_dataset_report(
                entry, raw_root=raw_root, out_root=Path(tmp) / "out", run_id="unit"
            )
            with (report.output_dir / "sim_real_gap.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("candidate_gap", {row["status"] for row in rows})
            payload = json.loads(
                (report.output_dir / "real_anchor_alignment_report.json").read_text()
            )
            self.assertIn(
                "Exact measured truth for any named object or platform",
                payload["unsupported_claims"],
            )


class RealDataPolicyTests(unittest.TestCase):
    def test_git_guard_blocks_real_data_and_generated_outputs(self) -> None:
        violations = find_tracked_real_data(
            tracked_paths=[
                "docs/real-data.md",
                "outputs/real-data/kth/run/feature_distributions.json",
                "real-data/kth/raw.mat",
                "detection/real_data/catalog.json",
            ]
        )
        self.assertEqual(
            violations,
            ["outputs/real-data/kth/run/feature_distributions.json", "real-data/kth/raw.mat"],
        )

    def test_current_repo_has_no_tracked_real_data(self) -> None:
        self.assertEqual(find_tracked_real_data(Path(__file__).resolve().parents[3]), [])

    def test_feature_policy_blocks_truth_and_proxy_columns(self) -> None:
        violations = observable_only_violations(
            [
                "snr_db_mean",
                "micro_doppler_peak_hz_proxy_mean",
                "normalized_snr_max",
                "target_family",
            ]
        )
        self.assertEqual(
            violations,
            ["micro_doppler_peak_hz_proxy_mean", "normalized_snr_max", "target_family"],
        )

    def test_realism_gate_demotes_high_auc_with_bad_operating_metrics(self) -> None:
        gate = evaluate_realism_gate(
            {
                "holdout_auc": 0.99,
                "pfa_at_1pct_budget": 0.12,
                "false_track_rate": 0.08,
                "missed_track_rate": 0.34,
                "calibration_anchor_status": "reference_only",
                "domain_holdout_status": "fail_empty",
            }
        )
        self.assertEqual(gate["status"], "no_go")
        self.assertTrue(gate["high_auc_with_failed_realism"])
        self.assertIn("pfa", {failure["gate"] for failure in gate["failures"]})

    def test_domain_holdout_metrics_do_not_fallback_to_test_split(self) -> None:
        records = pd.DataFrame(
            {
                "split": ["test", "test", "train", "train"],
                "difficulty_bucket": ["hard", "hard", "easy", "easy"],
                "holdout_role": ["seen", "seen", "seen", "seen"],
                "hard_negative_family": ["rc_fixed_wing", "kite", "rc_fixed_wing", "kite"],
            }
        )
        metrics = slice_metrics(
            records,
            np.asarray([1, 0, 1, 0]),
            records["split"].to_numpy(),
            np.asarray([0.9, 0.1, 0.8, 0.2]),
        )
        self.assertEqual(metrics["unseen_strata_holdout_count"], 0)
        self.assertEqual(metrics["unseen_strata_holdout_status"], "fail_empty")
        self.assertEqual(metrics["domain_holdout_status"], "fail")

    def test_fake_real_anchor_priors_move_synthetic_snr_without_feature_leakage(self) -> None:
        stratum = build_strata(50)[0]
        baseline_rng = np.random.default_rng(2026)
        hardened_rng = np.random.default_rng(2026)
        _, baseline_frame = simulate_record(
            baseline_rng, "baseline", 0, stratum, True, real_anchor_priors=None
        )
        priors = {
            "status": "measured_anchor_candidate",
            "simulator_priors": {
                "snr_center": {
                    "offset": 6.0,
                    "shrinkage": 1.0,
                    "source_observable": "snr_db",
                }
            },
        }
        _, hardened_frame = simulate_record(
            hardened_rng, "hardened", 0, stratum, True, real_anchor_priors=priors
        )
        baseline_snr = float(baseline_frame[:, FRAME_INDEX["snr_db"]].mean())
        hardened_snr = float(hardened_frame[:, FRAME_INDEX["snr_db"]].mean())
        self.assertGreater(hardened_snr, baseline_snr + 4.0)

        feature_df, feature_names = aggregate_frame_features(
            np.stack([baseline_frame, hardened_frame], axis=0)
        )
        self.assertEqual(len(feature_df), 2)
        violations = observable_only_violations(feature_names)
        self.assertEqual(violations, [])
        self.assertFalse(any(name.startswith("altitude_m_") for name in feature_names))

    def test_family_specific_prior_lookup_falls_back_to_global(self) -> None:
        priors = {
            "simulator_priors": {
                "micro_peak": {
                    "target_q10": 10.0,
                    "target_q90": 20.0,
                    "shrinkage": 1.0,
                }
            },
            "simulator_priors_by_family": {
                "bird_flock": {
                    "micro_peak": {
                        "target_q10": 30.0,
                        "target_q90": 40.0,
                        "shrinkage": 1.0,
                    }
                }
            },
        }
        self.assertEqual(knob_prior(priors, "micro_peak", "bird_flock")["target_q10"], 30.0)
        self.assertEqual(knob_prior(priors, "micro_peak", "single_bird")["target_q10"], 10.0)
        self.assertEqual(
            adjust_range(
                (1.0, 2.0),
                priors,
                "micro_peak",
                low=0.0,
                high=100.0,
                family="single_bird",
            ),
            (10.0, 20.0),
        )

    def test_micro_doppler_saturation_guard_fails_clamped_medians(self) -> None:
        frames = np.zeros((2, FRAME_COUNT, len(FRAME_INDEX)), dtype=np.float32)
        frames[:, :, FRAME_INDEX["micro_doppler_peak_hz_proxy"]] = MICRO_DOPPLER_PEAK_PROXY_CLAMP_HZ
        frames[:, :, FRAME_INDEX["micro_doppler_bandwidth_hz_proxy"]] = (
            MICRO_DOPPLER_BANDWIDTH_PROXY_CLAMP_HZ - 1.0
        )
        guard = micro_doppler_saturation_guard(frames)
        self.assertEqual(guard["status"], "fail")
        self.assertEqual(guard["checks"][0]["status"], "fail")
        self.assertEqual(guard["checks"][1]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
