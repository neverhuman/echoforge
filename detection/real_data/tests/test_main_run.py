from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from detection.advanced_main_run_detectors import (
    ADVANCED_METHOD_ID,
    run_advanced_main_run_detectors,
)
from detection.main_run_detectors import run_main_run_detectors
from detection.main_run_generation import build_main_run_dataset
from detection.main_run_generation import split_counts
from detection.main_run_types import (
    DETECTOR_VIEW_IDS,
    MODEL_FEATURE_DENYLIST,
    POSITIVE_MODEL_LABEL,
    RADAR_PULSES_PER_CPI,
    RADAR_RANGE_BINS,
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
        cls.advanced_root = root / "advanced_detection"
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
        cls.advanced_quality = run_advanced_main_run_detectors(
            cls.data_root,
            cls.advanced_root,
            folds=5,
            seed=202605210136,
            force=True,
            candidate_limit=96,
            evolution_rounds=2,
            search_profile="smoke",
            evolution_sample_rows=900,
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

    def test_low_prevalence_split_counts(self) -> None:
        holdout_groups, train_groups, holdout_positives, train_positives = split_counts(
            10_000,
            50,
        )
        self.assertEqual(holdout_groups, 1_500)
        self.assertEqual(train_groups, 8_500)
        self.assertEqual(holdout_positives, 8)
        self.assertEqual(train_positives, 42)

    def test_jamming_deception_rate_and_iq_shape(self) -> None:
        self.assertEqual(self.quality["jamming_deception_stress_status"], "pass")
        self.assertGreaterEqual(self.quality["jamming_deception_active_group_rate"], 0.10)
        self.assertLessEqual(self.quality["jamming_deception_active_group_rate"], 0.20)
        shard = self.data_root / "raw_complex_iq" / "active_radar_shard_0000.npz"
        with np.load(shard, allow_pickle=False) as loaded:
            iq = loaded["iq"]
        self.assertEqual(iq.shape[1:], (3, RADAR_PULSES_PER_CPI, RADAR_RANGE_BINS))

    def test_jamming_deception_features_are_detector_only(self) -> None:
        radar_rows = _read_csv(
            self.data_root / "detector_views" / "high_resolution_xku_cuas.csv"
        )
        header = set(radar_rows[0])
        self.assertIn("jd_spectral_flatness", header)
        self.assertIn("jd_range_line_occupancy", header)
        self.assertIn("jd_pulse_burstiness", header)
        self.assertFalse(
            {
                "jamming_deception_active",
                "jamming_deception_profile",
                "jamming_deception_family",
                "jamming_deception_rate_policy",
            }
            & header
        )

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

    def test_advanced_evolution_outputs_and_holdout_isolation(self) -> None:
        required_outputs = {
            "candidate_leaderboard.csv",
            "selection_lock.json",
            "advanced_feature_manifest.json",
            "evolution_trace.jsonl",
            "advanced_predictions.csv",
            "performance_metrics.csv",
            "performance_summary.json",
            "fusion_quality_report.json",
        }
        for filename in required_outputs:
            self.assertTrue((self.advanced_root / filename).exists(), filename)

        quality = json.loads((self.advanced_root / "fusion_quality_report.json").read_text())
        self.assertEqual(quality["status"], "pass")
        self.assertEqual(quality["holdout_isolation_status"], "pass")
        self.assertEqual(quality["selection_lock_status"], "written_before_holdout_scoring")
        self.assertGreaterEqual(quality["candidate_count"], 96)
        self.assertEqual(quality["candidate_selection_holdout_record_count"], 0)
        self.assertEqual(quality["calibration_holdout_record_count"], 0)
        self.assertEqual(quality["threshold_holdout_record_count"], 0)
        self.assertEqual(quality["selected_holdout_evaluation_count"], 1)
        self.assertLessEqual(
            (self.advanced_root / "selection_lock.json").stat().st_mtime_ns,
            (self.advanced_root / "advanced_predictions.csv").stat().st_mtime_ns,
        )

        manifest = json.loads((self.advanced_root / "advanced_feature_manifest.json").read_text())
        isolation = manifest["split_isolation_policy"]
        self.assertEqual(isolation["candidate_selection_split"], "train_cv")
        self.assertEqual(isolation["calibration_split"], "train_cv")
        self.assertEqual(isolation["threshold_split"], "train_cv")
        self.assertEqual(isolation["holdout_rows_used_for_feature_selection"], 0)
        self.assertEqual(isolation["holdout_rows_used_for_candidate_selection"], 0)
        self.assertEqual(isolation["holdout_rows_used_for_calibration"], 0)
        self.assertEqual(isolation["holdout_rows_used_for_threshold_selection"], 0)
        self.assertFalse(set(manifest["feature_columns"]) & set(MODEL_FEATURE_DENYLIST))

        leaderboard = _read_csv(self.advanced_root / "candidate_leaderboard.csv")
        self.assertGreaterEqual(len(leaderboard), 96)
        self.assertTrue(all(row["selection_split"] == "train_cv" for row in leaderboard))
        self.assertTrue(all(row["holdout_rows_used_for_selection"] == "0" for row in leaderboard))
        holdout_columns = {column for column in leaderboard[0] if column.startswith("holdout")}
        self.assertEqual(holdout_columns, {"holdout_rows_used_for_selection"})
        self.assertTrue(
            all(
                column.startswith("train_cv_")
                or column
                in {
                    "candidate_id",
                    "base_candidate_id",
                    "candidate_type",
                    "family",
                    "subset_name",
                    "head",
                    "calibrator",
                    "feature_count",
                    "selection_split",
                    "holdout_rows_used_for_selection",
                    "objective",
                    "initial_take_up_average_precision",
                    "threshold",
                    "calibration_info",
                }
                for column in leaderboard[0]
            )
        )
        lock = json.loads((self.advanced_root / "selection_lock.json").read_text())
        disallowed_lock_keys = {
            key
            for key in json.dumps(lock, sort_keys=True).split('"')
            if key.startswith("holdout_")
            and not key.startswith("holdout_rows_used_")
            and key != "holdout_evaluation_policy"
        }
        self.assertEqual(disallowed_lock_keys, set())
        self.assertEqual(lock["holdout_rows_used_for_selection"], 0)

        summary = json.loads((self.advanced_root / "performance_summary.json").read_text())
        self.assertEqual(
            summary["advanced_selection"]["holdout_rows_used_for_selection"],
            0,
        )
        self.assertIn(ADVANCED_METHOD_ID, summary["holdout"])

    def test_advanced_evolution_determinism_on_smoke_data(self) -> None:
        repeat_root = Path(self._tmp.name) / "advanced_detection_repeat"
        repeat_quality = run_advanced_main_run_detectors(
            self.data_root,
            repeat_root,
            folds=5,
            seed=202605210136,
            force=True,
            candidate_limit=96,
            evolution_rounds=2,
            search_profile="smoke",
            evolution_sample_rows=900,
        )
        self.assertEqual(
            repeat_quality["selected_candidate_id"],
            self.advanced_quality["selected_candidate_id"],
        )
        first = _read_csv(self.advanced_root / "candidate_leaderboard.csv")[:10]
        second = _read_csv(repeat_root / "candidate_leaderboard.csv")[:10]
        self.assertEqual(
            [(row["candidate_id"], row["objective"]) for row in first],
            [(row["candidate_id"], row["objective"]) for row in second],
        )
        predictions = _read_csv(self.advanced_root / "advanced_predictions.csv")
        repeat_predictions = _read_csv(repeat_root / "advanced_predictions.csv")
        self.assertEqual(
            [row["advanced_score"] for row in predictions],
            [row["advanced_score"] for row in repeat_predictions],
        )

    def test_advanced_feature_cache_reload_matches_fresh_scores(self) -> None:
        cache_path = Path(self._tmp.name) / "advanced_features_smoke.npz"
        fresh_root = Path(self._tmp.name) / "advanced_detection_cache_fresh"
        cached_root = Path(self._tmp.name) / "advanced_detection_cache_reload"
        fresh_quality = run_advanced_main_run_detectors(
            self.data_root,
            fresh_root,
            folds=5,
            seed=202605210136,
            force=True,
            candidate_limit=96,
            evolution_rounds=1,
            search_profile="smoke",
            feature_cache=cache_path,
            evolution_sample_rows=900,
        )
        cached_quality = run_advanced_main_run_detectors(
            self.data_root,
            cached_root,
            folds=5,
            seed=202605210136,
            force=True,
            candidate_limit=96,
            evolution_rounds=1,
            search_profile="smoke",
            feature_cache=cache_path,
            evolution_sample_rows=900,
        )
        self.assertEqual(
            fresh_quality["selected_candidate_id"], cached_quality["selected_candidate_id"]
        )
        fresh_predictions = _read_csv(fresh_root / "advanced_predictions.csv")
        cached_predictions = _read_csv(cached_root / "advanced_predictions.csv")
        self.assertEqual(
            [row["advanced_score"] for row in fresh_predictions],
            [row["advanced_score"] for row in cached_predictions],
        )

    def test_advanced_selection_ignores_holdout_label_and_feature_perturbation(self) -> None:
        perturbed_root = Path(self._tmp.name) / "training_holdout_perturbed"
        shutil.copytree(self.data_root, perturbed_root)
        records_path = perturbed_root / "records.csv"
        records = _read_csv(records_path)
        holdout_ids = {row["record_id"] for row in records if row["split_role"] == "holdout"}
        for row in records:
            if row["record_id"] in holdout_ids:
                row["label_id"] = "0" if row["label_id"] == "1" else "1"
        with records_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

        raw_index = _read_csv(perturbed_root / "raw_stream_index.csv")
        first_holdout = next(row for row in raw_index if row["record_id"] in holdout_ids)
        shard_path = perturbed_root / first_holdout["shard_path"]
        with np.load(shard_path) as loaded:
            payload = {name: loaded[name].copy() for name in loaded.files}
        payload["iq"][int(first_holdout["row_offset"])] *= -1.0
        np.savez_compressed(shard_path, **payload)

        perturbed_out = Path(self._tmp.name) / "advanced_detection_perturbed_holdout"
        perturbed_quality = run_advanced_main_run_detectors(
            perturbed_root,
            perturbed_out,
            folds=5,
            seed=202605210136,
            force=True,
            candidate_limit=96,
            evolution_rounds=2,
            search_profile="smoke",
            evolution_sample_rows=900,
        )
        self.assertEqual(
            perturbed_quality["selected_candidate_id"],
            self.advanced_quality["selected_candidate_id"],
        )
        original = _read_csv(self.advanced_root / "candidate_leaderboard.csv")[:10]
        perturbed = _read_csv(perturbed_out / "candidate_leaderboard.csv")[:10]
        self.assertEqual(
            [(row["candidate_id"], row["objective"]) for row in original],
            [(row["candidate_id"], row["objective"]) for row in perturbed],
        )

    def test_advanced_feature_policy_and_clean_room_manifest(self) -> None:
        advanced_manifest_root = Path(self._tmp.name) / "advanced_detection_manifest"
        run_advanced_main_run_detectors(
            self.data_root,
            advanced_manifest_root,
            folds=5,
            seed=202605210136,
            force=True,
            candidate_limit=128,
            evolution_rounds=1,
            search_profile="aggressive",
            evolution_sample_rows=900,
        )
        manifest = json.loads((advanced_manifest_root / "advanced_feature_manifest.json").read_text())
        self.assertEqual(manifest["candidate_generation_policy"]["search_profile"], "aggressive")
        self.assertIn("clean_room_inspiration", manifest)
        self.assertIn("split_isolation_policy", manifest)
        self.assertIn(
            "wasserstein_style_prototype_distances",
            manifest["clean_room_inspiration"]["implemented_families"],
        )
        self.assertFalse(set(manifest["feature_columns"]) & set(MODEL_FEATURE_DENYLIST))

        source_paths = [
            Path("detection/advanced_main_run_detectors.py"),
            Path("detection/run_advanced_main_run_detectors.py"),
            Path("detection/real_data/tests/test_main_run.py"),
            Path("docs/main_run_advanced_evolution.md"),
            Path("README.md"),
        ]
        external_project = "ve" + "ox"
        forbidden = (
            f"/home/ubuntu/{external_project}",
            f"import {external_project}",
            f"from {external_project}",
        )
        for source_path in source_paths:
            text = source_path.read_text(encoding="utf-8").lower()
            self.assertFalse(any(term in text for term in forbidden), source_path)


if __name__ == "__main__":
    unittest.main()
