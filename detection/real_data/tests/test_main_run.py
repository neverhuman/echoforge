from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from detection.main_run_detectors import run_main_run_detectors
from detection.main_run_generation import build_main_run_dataset
from detection.main_run_types import (
    DETECTOR_VIEW_IDS,
    MODEL_FEATURE_DENYLIST,
    POSITIVE_MODEL_LABEL,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class MainRunDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.data_root = root / "training"
        cls.detector_root = root / "detection"
        cls.quality = build_main_run_dataset(
            cls.data_root,
            scenario_groups=120,
            positive_groups=12,
            seed=202605210136,
            folds=5,
            shard_size=37,
            force=True,
            smoke=True,
        )
        cls.detector_quality = run_main_run_detectors(
            cls.data_root,
            cls.detector_root,
            folds=5,
            seed=202605210136,
            force=True,
        )
        cls.scenarios = _read_csv(cls.data_root / "scenario_manifest.csv")
        cls.records = _read_csv(cls.data_root / "records.csv")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_run_counts_and_phase_windows(self) -> None:
        self.assertEqual(len(self.scenarios), 120)
        self.assertEqual(len(self.records), 360)
        self.assertEqual(self.quality["status"], "pass")
        self.assertEqual(
            self.quality["phase_counts"],
            {
                "initial_take_up": 120,
                "climb_transition": 120,
                "cruise_altitude": 120,
            },
        )
        self.assertEqual(
            self.quality["phase_positive_counts"],
            {
                "initial_take_up": 12,
                "climb_transition": 12,
                "cruise_altitude": 12,
            },
        )
        windows = {
            row["phase_id"]: (float(row["phase_start_s"]), float(row["phase_end_s"]))
            for row in self.records
        }
        self.assertEqual(windows["initial_take_up"], (0.0, 30.0))
        self.assertEqual(windows["climb_transition"], (30.0, 90.0))
        self.assertEqual(windows["cruise_altitude"], (90.0, 150.0))

    def test_shahed_only_positive_labeling(self) -> None:
        positives = [row for row in self.scenarios if row["is_positive"] == "1"]
        negatives = [row for row in self.scenarios if row["is_positive"] == "0"]
        self.assertEqual(len(positives), 12)
        self.assertTrue(all(row["target_role"] == POSITIVE_MODEL_LABEL for row in positives))
        self.assertTrue(all(row["model_label"] == POSITIVE_MODEL_LABEL for row in positives))
        self.assertTrue(all(row["target_role"] != POSITIVE_MODEL_LABEL for row in negatives))
        for record in self.records:
            if record["label_id"] == "1":
                self.assertEqual(record["model_label"], POSITIVE_MODEL_LABEL)

    def test_split_and_fold_lock_by_scenario_group(self) -> None:
        scenario_split = {
            row["scenario_group_id"]: (row["split_role"], row["cv_fold"]) for row in self.scenarios
        }
        seen_folds: set[int] = set()
        for record in self.records:
            split_role, cv_fold = scenario_split[record["scenario_group_id"]]
            self.assertEqual(record["split_role"], split_role)
            self.assertEqual(record["cv_fold"], cv_fold)
            if split_role == "holdout":
                self.assertEqual(cv_fold, "")
            else:
                seen_folds.add(int(cv_fold))
        self.assertEqual(seen_folds, {0, 1, 2, 3, 4})
        grouped = {}
        for record in self.records:
            grouped.setdefault(record["scenario_group_id"], set()).add(record["phase_id"])
        self.assertTrue(
            all(
                phases == {"initial_take_up", "climb_transition", "cruise_altitude"}
                for phases in grouped.values()
            )
        )

    def test_raw_stream_alignment(self) -> None:
        index_rows = _read_csv(self.data_root / "raw_stream_index.csv")
        self.assertEqual(len(index_rows), len(self.records))
        cache: dict[str, object] = {}
        for row in index_rows:
            shard_path = row["shard_path"]
            if shard_path not in cache:
                cache[shard_path] = np.load(self.data_root / shard_path)
            shard = cache[shard_path]
            offset = int(row["row_offset"])
            self.assertEqual(str(shard["record_ids"][offset]), row["record_id"])
            self.assertEqual(str(shard["time_lock_ids"][offset]), row["time_lock_id"])
            self.assertEqual(
                int(shard["timestamp_start_ns"][offset]), int(row["timestamp_start_ns"])
            )

    def test_detector_view_record_identity(self) -> None:
        record_index = {
            row["record_id"]: (
                row["scenario_group_id"],
                row["time_lock_id"],
                row["phase_id"],
                row["split_role"],
            )
            for row in self.records
        }
        expected_ids = set(record_index)
        for view_id in DETECTOR_VIEW_IDS:
            rows = _read_csv(self.data_root / "detector_views" / f"{view_id}.csv")
            self.assertEqual({row["record_id"] for row in rows}, expected_ids)
            for row in rows:
                self.assertEqual(
                    (
                        row["scenario_group_id"],
                        row["time_lock_id"],
                        row["phase_id"],
                        row["split_role"],
                    ),
                    record_index[row["record_id"]],
                )

    def test_leakage_guard_feature_columns(self) -> None:
        manifest = json.loads((self.data_root / "dataset_manifest.json").read_text())
        self.assertEqual(set(manifest["model_feature_denylist"]), set(MODEL_FEATURE_DENYLIST))
        quality = json.loads((self.data_root / "quality_report.json").read_text())
        self.assertEqual(quality["leakage_guard_status"], "pass")
        for view in quality["raw_stream_summary"]["detector_views"].values():
            feature_columns = set(view["feature_columns"])
            self.assertFalse(feature_columns & set(MODEL_FEATURE_DENYLIST))
        for view_id in DETECTOR_VIEW_IDS:
            rows = _read_csv(self.data_root / "detector_views" / f"{view_id}.csv")
            self.assertNotIn("restricted_truth_path", rows[0])
            self.assertNotIn("scenario_seed", rows[0])

    def test_fusion_holdout_isolation(self) -> None:
        manifest = json.loads((self.detector_root / "calibration_manifest.json").read_text())
        self.assertEqual(manifest["status"], "pass")
        self.assertEqual(manifest["used_split_roles"], ["train_cv"])
        self.assertEqual(manifest["excluded_split_roles"], ["holdout"])
        self.assertEqual(manifest["holdout_fit_record_count"], 0)
        self.assertEqual(self.detector_quality["holdout_isolation_status"], "pass")

        calibration_rows = _read_csv(self.detector_root / "calibration_folds.csv")
        self.assertTrue(all(row["fit_split_role"] == "train_cv" for row in calibration_rows))
        fusion_rows = _read_csv(self.detector_root / "fusion_predictions.csv")
        holdout_rows = [row for row in fusion_rows if row["split_role"] == "holdout"]
        self.assertTrue(holdout_rows)
        self.assertTrue(
            all(row["calibration_role"] == "holdout_scored_only" for row in holdout_rows)
        )

    def test_performance_metrics_cover_all_detector_branches(self) -> None:
        summary = json.loads((self.detector_root / "performance_summary.json").read_text())
        self.assertEqual(summary["status"], "pass")
        expected_methods = {
            "high_resolution_xku_cuas",
            "tactical_s_band_aesa",
            "gbad_3d4d_cueing",
            "distributed_acoustic_cue",
            "tabular_ml_baseline",
            "sequence_ml_proxy",
            "layered_fusion_c2",
        }
        self.assertEqual(set(summary["methods"]), expected_methods)
        self.assertEqual(set(summary["holdout"]), expected_methods)
        for method, threshold in summary["thresholds"].items():
            self.assertIn(method, expected_methods)
            self.assertEqual(threshold["threshold_source"], "train_cv_max_f1")

        rows = _read_csv(self.detector_root / "performance_metrics.csv")
        holdout_all = [
            row for row in rows if row["split_role"] == "holdout" and row["phase_id"] == "all"
        ]
        self.assertEqual({row["method"] for row in holdout_all}, expected_methods)
        for row in holdout_all:
            self.assertEqual(row["threshold_source"], "train_cv_max_f1")
            self.assertNotEqual(row["roc_auc"], "")
            self.assertNotEqual(row["average_precision"], "")


if __name__ == "__main__":
    unittest.main()
