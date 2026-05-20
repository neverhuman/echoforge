"""Acoustic cueing network product builders for the v1 benchmark."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ml_training_config import (
    ACOUSTIC_DETECTOR_FAMILY_ID,
    ACOUSTIC_FALSE_CUE_SOURCES,
    ACOUSTIC_NODE_IDS,
    FRAME_INDEX,
    PHASE_SPECS,
)
from ml_training_generators import stable_seed


def circular_mean_deg(values: list[float]) -> float:
    if not values:
        return float("nan")
    radians = np.radians(np.asarray(values, dtype=np.float64))
    angle = math.degrees(math.atan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians)))))
    return float((angle + 360.0) % 360.0)


def acoustic_false_source(rng: np.random.Generator, record: pd.Series) -> str:
    if str(record.get("interference", "")) == "rfi_burst":
        return "generator_hum"
    if str(record.get("clutter_regime", "")) in {"rain_cell", "dust_weather", "sea_glint"}:
        return str(rng.choice(["wind_gust", "thunder_rumble"]))
    if str(record.get("site_archetype_id", "")).endswith("urban_edge"):
        return str(rng.choice(["road_traffic", "industrial_machinery", "generator_hum"]))
    return str(rng.choice(ACOUSTIC_FALSE_CUE_SOURCES))


def acoustic_detection_probability(record: pd.Series, mean_range_m: float, horizon_masked_fraction: float) -> float:
    phase_id = str(record["phase_id"])
    scene_role = str(record["scene_role"])
    is_positive = bool(record["is_public_proxy_positive"])
    if is_positive or scene_role == "target_masked_counterfactual":
        base = {
            "initial_take_up": 0.70,
            "climb_transition": 0.78,
            "cruise_altitude": 0.62,
        }.get(phase_id, 0.62)
        if phase_id == "initial_take_up" and horizon_masked_fraction > 0.55:
            base += 0.10
    elif scene_role == "matched_confuser":
        base = 0.26
    else:
        base = 0.16
    range_penalty = max(0.0, (mean_range_m - 5_000.0) / 11_000.0) * 0.28
    urban_bonus = 0.05 if str(record.get("site_archetype_id", "")).endswith("urban_edge") else 0.0
    return float(np.clip(base - range_penalty + urban_bonus, 0.03, 0.92))


def acoustic_spectral_features(
    rng: np.random.Generator,
    signal_like: bool,
    false_source: str,
    micro_peak_hz: float,
    micro_bandwidth_hz: float,
    micro_energy: float,
) -> dict[str, float]:
    if signal_like:
        engine_peak = float(np.clip(85.0 + micro_peak_hz * 0.42 + rng.normal(0.0, 18.0), 45.0, 260.0))
        prop_harmonic = float(np.clip(engine_peak * rng.uniform(1.8, 2.4), 90.0, 620.0))
        entropy = float(np.clip(0.32 + rng.normal(0.0, 0.08) + micro_bandwidth_hz / 900.0, 0.12, 0.92))
        modulation = float(np.clip(0.38 + micro_energy * 0.40 + rng.normal(0.0, 0.08), 0.05, 0.98))
        acoustic_snr = float(np.clip(7.5 + micro_energy * 13.0 + rng.normal(0.0, 3.0), -4.0, 28.0))
    else:
        source_base = {
            "wind_gust": (55.0, 0.82),
            "road_traffic": (115.0, 0.72),
            "industrial_machinery": (180.0, 0.58),
            "generator_hum": (120.0, 0.48),
            "thunder_rumble": (38.0, 0.88),
            "none": (95.0, 0.70),
        }.get(false_source, (100.0, 0.70))
        engine_peak = float(np.clip(source_base[0] + rng.normal(0.0, 32.0), 18.0, 320.0))
        prop_harmonic = float(np.clip(engine_peak * rng.uniform(1.2, 3.5), 35.0, 720.0))
        entropy = float(np.clip(source_base[1] + rng.normal(0.0, 0.10), 0.18, 0.98))
        modulation = float(np.clip(0.08 + rng.beta(1.2, 5.5) * 0.42, 0.02, 0.62))
        acoustic_snr = float(np.clip(rng.normal(2.0, 4.0), -9.0, 18.0))
    return {
        "engine_band_peak_hz": engine_peak,
        "prop_harmonic_hz": prop_harmonic,
        "spectral_bandwidth_hz": float(np.clip(micro_bandwidth_hz + rng.normal(0.0, 22.0), 18.0, 420.0)),
        "spectral_entropy": entropy,
        "modulation_confidence": modulation,
        "acoustic_snr_db": acoustic_snr,
    }


def build_acoustic_cue_products(
    records: pd.DataFrame,
    frames: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    node_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    phase_metric_rows: list[dict[str, Any]] = []
    track_count_by_record: dict[str, int] = {}
    first_latency_by_record: dict[str, float] = {}
    pre_los_cue_count = 0
    false_source_counts: dict[str, int] = {source: 0 for source in ACOUSTIC_FALSE_CUE_SOURCES}

    for record_idx, record in records.iterrows():
        record_id = str(record["record_id"])
        rng = np.random.default_rng(
            stable_seed(int(record["scenario_seed"]), int(record["object_seed"]), record_idx, 66_211)
        )
        valid = valid_mask[record_idx]
        frame = frames[record_idx, valid]
        if frame.size == 0:
            continue
        mean_range_m = float(np.mean(frame[:, FRAME_INDEX["range_m"]]))
        micro_peak_hz = float(np.mean(frame[:, FRAME_INDEX["micro_doppler_peak_hz_proxy"]]))
        micro_bw_hz = float(np.mean(frame[:, FRAME_INDEX["micro_doppler_bandwidth_hz_proxy"]]))
        micro_energy = float(np.mean(frame[:, FRAME_INDEX["micro_doppler_energy"]]))
        horizon_masked = float(record["horizon_masked_fraction"])
        probability = acoustic_detection_probability(record, mean_range_m, horizon_masked)
        source_count = len(ACOUSTIC_NODE_IDS)
        node_hits = rng.random(source_count) < probability
        if int(node_hits.sum()) < 2:
            continue

        scene_role = str(record["scene_role"])
        signal_like = bool(record["is_public_proxy_positive"]) or scene_role == "target_masked_counterfactual"
        false_source = "none" if signal_like else acoustic_false_source(rng, record)
        if false_source in false_source_counts:
            false_source_counts[false_source] += 1
        spectral = acoustic_spectral_features(rng, signal_like, false_source, micro_peak_hz, micro_bw_hz, micro_energy)
        phase_id = str(record["phase_id"])
        phase_duration_s = float(record["phase_end_s"]) - float(record["phase_start_s"])
        cue_latency_s = float(np.clip(rng.gamma(2.0, 2.2), 0.5, max(1.0, min(phase_duration_s - 0.5, 28.0))))
        bearing_center = float(rng.uniform(0.0, 360.0))
        detected_nodes = [ACOUSTIC_NODE_IDS[idx] for idx, hit in enumerate(node_hits) if bool(hit)]
        if phase_id == "initial_take_up" and horizon_masked > 0.55 and signal_like:
            pre_los_cue_count += 1
        track_fragments = 1 + int(signal_like and rng.random() < (0.08 + 0.12 * horizon_masked))

        for fragment_idx in range(track_fragments):
            fragment_latency = cue_latency_s + fragment_idx * float(rng.uniform(4.0, 11.0))
            fragment_bearings: list[float] = []
            track_id = f"acu_track_{record_id}_{fragment_idx + 1:02d}"
            event_id = f"acu_event_{record_id}_{fragment_idx + 1:02d}"
            for node_idx, source_id in enumerate(detected_nodes):
                doa_uncertainty = float(np.clip(rng.normal(8.0, 2.5) + mean_range_m / 7_500.0, 2.0, 24.0))
                doa = float((bearing_center + rng.normal(0.0, doa_uncertainty)) % 360.0)
                fragment_bearings.append(doa)
                node_rows.append(
                    {
                        "record_id": record_id,
                        "phase_id": phase_id,
                        "detector_family_id": ACOUSTIC_DETECTOR_FAMILY_ID,
                        "acoustic_event_id": event_id,
                        "acoustic_track_id": track_id,
                        "source_id": source_id,
                        "cue_time_s": fragment_latency,
                        "cue_latency_s": fragment_latency,
                        "doa_bearing_deg": doa,
                        "doa_uncertainty_deg": doa_uncertainty,
                        "tdoa_pair_id": f"{source_id}:{detected_nodes[(node_idx + 1) % len(detected_nodes)]}",
                        "tdoa_delta_ms": float(rng.normal(0.0, 18.0)),
                        "tdoa_residual_ms": float(abs(rng.normal(3.0, 2.4))),
                        "engine_band_peak_hz": spectral["engine_band_peak_hz"],
                        "prop_harmonic_hz": spectral["prop_harmonic_hz"],
                        "spectral_bandwidth_hz": spectral["spectral_bandwidth_hz"],
                        "spectral_entropy": spectral["spectral_entropy"],
                        "modulation_confidence": spectral["modulation_confidence"],
                        "acoustic_snr_db": spectral["acoustic_snr_db"],
                        "false_cue_source": false_source,
                    }
                )

            source_weight = max(1.0, math.sqrt(len(detected_nodes)))
            uncertainty_major = float(np.clip(mean_range_m * rng.uniform(0.10, 0.24) / source_weight, 45.0, 2_800.0))
            uncertainty_minor = float(np.clip(uncertainty_major * rng.uniform(0.38, 0.72), 20.0, uncertainty_major))
            confidence = float(
                np.clip(
                    0.18
                    + 0.12 * len(detected_nodes)
                    + spectral["modulation_confidence"] * 0.34
                    + max(0.0, spectral["acoustic_snr_db"]) * 0.012
                    - (0.18 if false_source != "none" else 0.0),
                    0.02,
                    0.99,
                )
            )
            triangulated_range = float(np.clip(mean_range_m + rng.normal(0.0, uncertainty_major), 100.0, 22_000.0))
            track_rows.append(
                {
                    "record_id": record_id,
                    "phase_id": phase_id,
                    "detector_family_id": ACOUSTIC_DETECTOR_FAMILY_ID,
                    "acoustic_track_id": track_id,
                    "source_count": len(detected_nodes),
                    "source_ids": ";".join(detected_nodes),
                    "cue_time_s": fragment_latency,
                    "first_cue_latency_s": fragment_latency,
                    "triangulated_bearing_deg": circular_mean_deg(fragment_bearings),
                    "triangulated_range_m": triangulated_range,
                    "uncertainty_major_m": uncertainty_major,
                    "uncertainty_minor_m": uncertainty_minor,
                    "uncertainty_orientation_deg": float(rng.uniform(0.0, 180.0)),
                    "cue_confidence": confidence,
                    "engine_band_peak_hz": spectral["engine_band_peak_hz"],
                    "prop_harmonic_hz": spectral["prop_harmonic_hz"],
                    "spectral_entropy": spectral["spectral_entropy"],
                    "modulation_confidence": spectral["modulation_confidence"],
                    "acoustic_snr_db": spectral["acoustic_snr_db"],
                    "false_cue_source": false_source,
                }
            )
        track_count_by_record[record_id] = track_count_by_record.get(record_id, 0) + track_fragments
        first_latency_by_record[record_id] = min(first_latency_by_record.get(record_id, float("inf")), cue_latency_s)

    track_df = pd.DataFrame(track_rows)
    node_df = pd.DataFrame(node_rows)
    for phase_id, phase_records in records.groupby("phase_id"):
        ids = phase_records["record_id"].astype(str).to_numpy()
        labels = phase_records["scene_role"].astype(str).isin(
            ["positive_public_proxy", "target_masked_counterfactual"]
        ).to_numpy()
        has_track = np.asarray([track_count_by_record.get(str(record_id), 0) > 0 for record_id in ids], dtype=bool)
        track_counts = np.asarray([track_count_by_record.get(str(record_id), 0) for record_id in ids], dtype=np.float64)
        positive_latency = [
            first_latency_by_record[str(record_id)]
            for record_id, label in zip(ids, labels)
            if label and str(record_id) in first_latency_by_record
        ]
        positive_count = int(labels.sum())
        negative_count = int((~labels).sum())
        phase_metric_rows.append(
            {
                "phase_id": str(phase_id),
                "record_count": int(len(ids)),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "pd_any_acoustic_cue": float(has_track[labels].mean()) if positive_count else float("nan"),
                "pfa_any_acoustic_cue": float(has_track[~labels].mean()) if negative_count else float("nan"),
                "first_cue_latency_s": float(np.mean(positive_latency)) if positive_latency else float("nan"),
                "track_initiation_latency_s": float(np.mean(positive_latency)) if positive_latency else float("nan"),
                "track_fragmentation_rate": float(np.mean(np.maximum(track_counts[labels] - 1.0, 0.0))) if positive_count else float("nan"),
                "false_track_rate": float(has_track[~labels].mean()) if negative_count else float("nan"),
                "missed_track_rate": float((~has_track[labels]).mean()) if positive_count else float("nan"),
                "pre_radar_los_cue_count": int(
                    sum(
                        1
                        for _, row in phase_records.iterrows()
                        if str(row["scene_role"]) in {"positive_public_proxy", "target_masked_counterfactual"}
                        and str(row["phase_id"]) == "initial_take_up"
                        and float(row["horizon_masked_fraction"]) > 0.55
                        and track_count_by_record.get(str(row["record_id"]), 0) > 0
                    )
                ),
                "false_cue_sources_reported": ";".join(
                    source for source, count in sorted(false_source_counts.items()) if count > 0
                ),
            }
        )
    phase_df = pd.DataFrame(phase_metric_rows).sort_values("phase_id")
    required_node_cols = {
        "record_id",
        "phase_id",
        "detector_family_id",
        "source_id",
        "doa_bearing_deg",
        "tdoa_residual_ms",
        "engine_band_peak_hz",
        "prop_harmonic_hz",
        "false_cue_source",
    }
    required_track_cols = {
        "record_id",
        "phase_id",
        "detector_family_id",
        "source_count",
        "triangulated_bearing_deg",
        "uncertainty_major_m",
        "uncertainty_minor_m",
        "cue_confidence",
        "false_cue_source",
    }
    node_columns_ok = required_node_cols.issubset(set(node_df.columns))
    track_columns_ok = required_track_cols.issubset(set(track_df.columns))
    phase_ids_ok = set(phase_df["phase_id"].astype(str)) == {phase.phase_id for phase in PHASE_SPECS}
    raw_audio_columns = [
        col
        for col in list(node_df.columns) + list(track_df.columns)
        if "raw_audio" in str(col).lower() or str(col).lower() in {"audio", "audio_path", "waveform_path"}
    ]
    false_sources_present = sum(1 for count in false_source_counts.values() if count > 0) >= 2
    summary_status = (
        "pass"
        if node_columns_ok
        and track_columns_ok
        and phase_ids_ok
        and not raw_audio_columns
        and pre_los_cue_count > 0
        and false_sources_present
        else "fail"
    )
    schema = {
        "artifact": "acoustic_cueing_products",
        "detector_family_id": ACOUSTIC_DETECTOR_FAMILY_ID,
        "claim_boundary": "Synthetic passive acoustic cue products only; no raw audio, measured acoustic traces, proprietary signatures, or exact platform audio fidelity.",
        "files": {
            "node_detections": "acoustic_node_detections.csv",
            "cue_tracks": "acoustic_cue_tracks.csv",
            "phase_metrics": "acoustic_phase_metrics.csv",
        },
        "node_detection_columns": list(node_df.columns),
        "cue_track_columns": list(track_df.columns),
        "phase_metric_columns": list(phase_df.columns),
        "raw_audio_stored": False,
        "model_facing_default": False,
    }
    summary = {
        "status": summary_status,
        "detector_family_id": ACOUSTIC_DETECTOR_FAMILY_ID,
        "node_detection_count": int(len(node_df)),
        "cue_track_count": int(len(track_df)),
        "phase_ids": sorted(phase_df["phase_id"].astype(str).unique().tolist()),
        "raw_audio_stored": False,
        "raw_audio_column_violations": raw_audio_columns,
        "pre_radar_los_cue_count": int(pre_los_cue_count),
        "false_cue_source_counts": false_source_counts,
        "required_metrics": [
            "pd_any_acoustic_cue",
            "pfa_any_acoustic_cue",
            "first_cue_latency_s",
            "track_initiation_latency_s",
            "track_fragmentation_rate",
            "false_track_rate",
            "missed_track_rate",
        ],
    }
    return node_df, track_df, phase_df, schema, summary
