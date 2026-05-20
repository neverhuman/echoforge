from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from detection.detection_common_helpers import slice_metrics
from detection.real_data.adapters import build_dataset_report, dry_run_fetch
from detection.real_data.catalog import load_catalog, validate_catalog
from detection.real_data.feature_policy import observable_only_violations
from detection.real_data.git_guard import find_tracked_real_data
from detection.real_data.realism_gate import evaluate_realism_gate


CATALOG = Path(__file__).resolve().parents[1] / "catalog.json"


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


if __name__ == "__main__":
    unittest.main()
