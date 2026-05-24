"""Manual real-data adapter scaffolding.

Adapters are intentionally explicit: this lane never downloads or normalizes
measured data as a side effect of CI. Local data is opt-in under the configured
cache root, and missing or license-gated data emits reference-only reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - import optional until raw npy is used
    np = None  # type: ignore[assignment]


OBSERVATION_FILENAME = "observations.csv"
KTH_RAW_FILENAME = "data_SAAB_SIRS_77GHz_FMCW.npy"
KTH_PRF_HZ = 17_000.0
KTH_AZIMUTH_SAMPLES = 256
KTH_RANGE_CELLS = 5
NORMALIZED_OBSERVATION_FIELDS = [
    "dataset_id",
    "sample_id_hash",
    "split_role",
    "class_family",
    "observable_name",
    "value",
    "unit",
    "radar_product_level",
    "phase_proxy",
    "license_status",
    "allowed_claim_level",
]
CALIBRATION_FIELDS = [
    "dataset_id",
    "feature",
    "metric",
    "measured_value",
    "reference_min",
    "reference_max",
    "distance",
    "status",
    "allowed_claim_level",
]
SIM_REAL_GAP_FIELDS = [
    "dataset_id",
    "observable_name",
    "metric",
    "measured_value",
    "synthetic_reference_value",
    "gap_value",
    "status",
    "allowed_claim_level",
]
LICENSE_GATED_ADAPTERS = {
    "han_jung_timesync_drone_v1",
    "raddar_rdrd_v1",
}
SIMULATOR_OBSERVABLE_MAP = {
    "micro_doppler_peak_hz": "micro_peak",
    "micro_doppler_bandwidth_hz": "micro_bw",
    "doppler_bandwidth_hz": "micro_bw",
    "micro_doppler_energy": "micro_amp",
    "spectral_entropy": "spectral_variance",
    "snr_db": "snr_center",
    "noise_floor_db": "local_noise_floor_db",
    "local_noise_floor_db": "local_noise_floor_db",
    "track_gap_fraction": "dropout",
    "dropout_fraction": "dropout",
    "clutter_score": "glint",
    "false_alarm_pressure": "glint",
    "cfar_threshold_db": "cfar_threshold",
}
KTH_DATASET_ID = "kth-drone-bird-human-77ghz"
KTH_APPLICABILITY_BY_OBSERVABLE = {
    "micro_doppler_peak_hz": "tune",
    "micro_doppler_bandwidth_hz": "tune",
    "doppler_bandwidth_hz": "tune",
    "micro_doppler_energy": "tune_rank_shape_only",
    "spectral_entropy": "tune",
    "track_gap_fraction": "tune",
    "edge_truncated": "tune",
    "range_m": "compare_only",
    "return_power_db": "compare_only",
}
KTH_CLASS_APPLICABILITY = {
    "drone": "tune",
    "bird": "tune",
    "human": "compare_only",
    "calibration_reflector": "compare_only",
}
KTH_SIMULATOR_FAMILY_TARGETS = {
    "drone": ["rc_fixed_wing"],
    "bird": ["bird_flock", "single_bird"],
}
KTH_FAMILY_PRIOR_BOUNDS = {
    "rc_fixed_wing": {
        "micro_bw": (35.0, 145.0),
        "micro_amp": (0.30, 0.85),
        "dropout": (0.04, 0.28),
    },
    "bird_flock": {
        "micro_bw": (25.0, 130.0),
        "micro_amp": (0.18, 0.68),
        "dropout": (0.08, 0.34),
    },
    "single_bird": {
        "micro_bw": (20.0, 110.0),
        "micro_amp": (0.12, 0.55),
        "dropout": (0.08, 0.34),
    },
}
KTH_PRIOR_SHRINKAGE_BY_FAMILY = {
    "rc_fixed_wing": 0.20,
    "bird_flock": 0.45,
    "single_bird": 0.40,
}
SYNTHETIC_PRIOR_CENTERS = {
    "micro_peak": 95.0,
    "micro_bw": 90.0,
    "micro_amp": 0.55,
    "spectral_variance": 0.50,
    "snr_center": 2.0,
    "local_noise_floor_db": -40.0,
    "dropout": 0.12,
    "glint": 0.16,
    "cfar_threshold": 9.5,
}


@dataclass(frozen=True)
class DatasetReport:
    dataset_id: str
    status: str
    output_dir: Path
    message: str

    def to_json(self) -> dict[str, str]:
        return {
            "dataset_id": self.dataset_id,
            "status": self.status,
            "output_dir": self.output_dir.as_posix(),
            "message": self.message,
        }


def default_raw_root() -> Path:
    configured = os.environ.get("ECHOFORGE_REAL_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "echoforge" / "real-data"


def dataset_root(raw_root: Path, dataset_id: str) -> Path:
    return raw_root.expanduser() / dataset_id


def derived_output_dir(out_root: Path, dataset_id: str, run_id: str) -> Path:
    return out_root / dataset_id / run_id


def dry_run_fetch(entry: dict[str, Any], raw_root: Path | None = None) -> dict[str, Any]:
    root = default_raw_root() if raw_root is None else Path(raw_root)
    local_root = dataset_root(root, str(entry["dataset_id"]))
    retrieval = dict(entry.get("retrieval", {}))
    return {
        "dataset_id": entry["dataset_id"],
        "status": "available_local" if local_root.exists() else "manual_required",
        "local_root": local_root.as_posix(),
        "retrieval_method": retrieval.get("method", "manual"),
        "source_url": retrieval.get("url", entry.get("source_url", "")),
        "network_side_effects": "none",
        "message": "Dry run only; no files were downloaded or written.",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sample_hash(dataset_id: str, raw_id: Any) -> str:
    digest = hashlib.sha256(f"{dataset_id}:{raw_id}".encode("utf-8")).hexdigest()
    return digest[:24]


def _split_role(dataset_id: str, raw_id: Any) -> str:
    bucket = int(_sample_hash(dataset_id, raw_id)[:8], 16) % 10
    if bucket < 6:
        return "calibration"
    if bucket < 8:
        return "benchmark"
    return "holdout"


def _kth_split_role(raw_split: Any) -> str:
    try:
        split = int(raw_split)
    except (TypeError, ValueError):
        return "unknown"
    return {
        1: "calibration",
        2: "benchmark",
        3: "holdout",
    }.get(split, "unknown")


def _kth_class_family(label: Any) -> str:
    if np is not None:
        label_array = np.asarray(label, dtype=object)
        if label_array.size == 1:
            label = label_array.reshape(-1)[0]
    normalized = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"d1", "d2", "d3", "d4", "d5", "d6"}:
        return "drone"
    if normalized in {"human_walk", "human_run", "human"}:
        return "human"
    if normalized in {"cr", "corner_reflector", "corner_reflector_calibration"}:
        return "calibration_reflector"
    kth_birds = {
        "b",
        "bird",
        "black_headed_gull",
        "heron",
        "pigeon",
        "raven",
        "seagull",
        "seagull_and_black_headed_gull",
    }
    if "bird" in normalized or normalized in kth_birds:
        return "bird"
    return normalized or "unknown"


def _clean_class_family(row: dict[str, Any], default: str = "unknown") -> str:
    raw = (
        row.get("class_family")
        or row.get("target_class")
        or row.get("object_class")
        or row.get("target_type")
        or row.get("label")
        or row.get("class")
        or default
    )
    return str(raw).strip().lower().replace(" ", "_") or default


def _normalize_observation(
    entry: dict[str, Any],
    *,
    raw_id: Any,
    class_family: str,
    observable_name: str,
    value: float,
    unit: str,
    radar_product_level: str | None = None,
    phase_proxy: str = "observable_distribution",
    split_role: str | None = None,
) -> dict[str, Any]:
    dataset_id = str(entry["dataset_id"])
    return {
        "dataset_id": dataset_id,
        "sample_id_hash": _sample_hash(dataset_id, raw_id),
        "split_role": split_role or _split_role(dataset_id, raw_id),
        "class_family": class_family,
        "observable_name": observable_name,
        "value": value,
        "unit": unit,
        "radar_product_level": radar_product_level or str(entry.get("product_level", "")),
        "phase_proxy": phase_proxy,
        "license_status": str(entry.get("license_status", "")),
        "allowed_claim_level": str(entry.get("allowed_claim_level", "")),
    }


def _observations_from_feature_value(
    entry: dict[str, Any], local_root: Path
) -> list[dict[str, Any]]:
    path = local_root / OBSERVATION_FILENAME
    if not path.exists():
        return []
    rows = _read_csv(path)
    if not rows:
        return []
    fieldnames = set(rows[0])
    if NORMALIZED_OBSERVATION_FIELDS and set(NORMALIZED_OBSERVATION_FIELDS).issubset(fieldnames):
        normalized = []
        for row in rows:
            value = _to_float(row.get("value"))
            if value is None:
                continue
            normalized.append({**row, "value": value})
        return normalized
    if not {"feature", "value"}.issubset(fieldnames):
        return []

    normalized = []
    for idx, row in enumerate(rows):
        value = _to_float(row.get("value"))
        feature = str(row.get("feature", "")).strip()
        if value is None or not feature:
            continue
        normalized.append(
            _normalize_observation(
                entry,
                raw_id=row.get("sample_id") or row.get("track_id") or idx,
                class_family=_clean_class_family(row),
                observable_name=feature,
                value=value,
                unit=str(row.get("unit", "")),
                phase_proxy=str(row.get("phase_proxy", "legacy_feature_value")),
            )
        )
    return normalized


def _as_1d_array(value: Any) -> list[Any]:
    if value is None:
        return []
    if np is None:
        return [value]
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return [array.item()]
    return list(array.reshape(-1))


def _as_segment_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if np is None:
        return [value]
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return [array.item()]
    if array.ndim == 1:
        return list(array)
    if array.shape[0] == KTH_RANGE_CELLS * KTH_AZIMUTH_SAMPLES:
        return [array[:, idx] for idx in range(array.shape[1])]
    if array.shape[-2:] == (KTH_RANGE_CELLS, KTH_AZIMUTH_SAMPLES):
        return [array[idx] for idx in range(array.shape[0])]
    if array.shape[-1] == KTH_RANGE_CELLS * KTH_AZIMUTH_SAMPLES:
        return [array[idx] for idx in range(array.shape[0])]
    return list(array.reshape(-1))


def _kth_segment_matrix(raw_segment: Any) -> Any:
    if np is None:
        raise ValueError("numpy is required to read KTH raw .npy files")
    segment = np.asarray(raw_segment)
    if segment.shape == (KTH_RANGE_CELLS, KTH_AZIMUTH_SAMPLES):
        return segment.astype(np.complex128, copy=False)
    flat = segment.reshape(-1)
    expected = KTH_RANGE_CELLS * KTH_AZIMUTH_SAMPLES
    if flat.size != expected:
        raise ValueError(f"KTH segment must contain {expected} complex samples; got {flat.size}")
    return flat.astype(np.complex128, copy=False).reshape(KTH_RANGE_CELLS, KTH_AZIMUTH_SAMPLES)


def _spectral_features(center_cell: Any, full_segment: Any) -> dict[str, float]:
    if np is None:
        raise ValueError("numpy is required to compute KTH spectral features")
    centered = np.asarray(center_cell, dtype=np.complex128)
    full = np.asarray(full_segment, dtype=np.complex128)
    spectrum = np.fft.fftshift(np.fft.fft(centered * np.hanning(centered.size)))
    power = np.abs(spectrum) ** 2
    total_power = float(power.sum())
    freqs = np.fft.fftshift(np.fft.fftfreq(centered.size, d=1.0 / KTH_PRF_HZ))
    if total_power <= 0.0:
        peak_hz = 0.0
        bandwidth_hz = 0.0
        entropy = 0.0
    else:
        non_dc = np.abs(freqs) > (KTH_PRF_HZ / centered.size)
        search_power = np.where(non_dc, power, 0.0)
        peak_hz = float(abs(freqs[int(np.argmax(search_power))]))
        order = np.argsort(np.abs(freqs))
        cumulative = np.cumsum(power[order])
        cutoff_idx = int(np.searchsorted(cumulative, total_power * 0.90, side="left"))
        cutoff_idx = min(cutoff_idx, len(order) - 1)
        bandwidth_hz = float(2.0 * abs(freqs[order[cutoff_idx]]))
        probabilities = power / total_power
        positive = probabilities[probabilities > 0.0]
        entropy = float(-np.sum(positive * np.log2(positive)) / math.log2(power.size))
    return_power = float(10.0 * math.log10(float(np.mean(np.abs(full) ** 2)) + 1e-12))
    return {
        "micro_doppler_peak_hz": peak_hz,
        "micro_doppler_bandwidth_hz": bandwidth_hz,
        "micro_doppler_energy": total_power,
        "spectral_entropy": entropy,
        "return_power_db": return_power,
    }


def _track_gap_fraction(times: list[Any]) -> float:
    values = [_to_float(value) for value in times]
    finite = [value for value in values if value is not None]
    if len(finite) < 3:
        return 0.0
    deltas = [b - a for a, b in zip(finite, finite[1:]) if b >= a]
    if not deltas:
        return 0.0
    typical = _quantile(deltas, 0.50)
    if typical <= 0.0:
        return 0.0
    gaps = sum(1 for delta in deltas if delta > typical * 2.5)
    return float(gaps / len(deltas))


def _kth_record_segments(record: Any) -> Iterable[tuple[Any, Any, Any, Any, Any, Any]]:
    label = record[0]
    segments = _as_segment_list(record[1])
    ranges = _as_1d_array(record[2])
    times = _as_1d_array(record[3])
    splits = _as_1d_array(record[4])
    edge_flags = _as_1d_array(record[5])
    count = len(segments)
    for idx in range(count):
        yield (
            label,
            segments[idx],
            ranges[idx] if idx < len(ranges) else None,
            times[idx] if idx < len(times) else None,
            splits[idx] if idx < len(splits) else None,
            edge_flags[idx] if idx < len(edge_flags) else False,
        )


def _kth_raw_shape_report(path: Path, data: Any) -> dict[str, Any]:
    if np is None:
        raise ValueError("numpy is required to read KTH raw .npy files")
    labels: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    edge_counts = {"edge": 0, "non_edge": 0}
    segment_count = 0
    for record in data:
        for label, _segment, _range_m, _time_s, split, edge in _kth_record_segments(record):
            segment_count += 1
            family = _kth_class_family(label)
            labels[family] = labels.get(family, 0) + 1
            split_role = _kth_split_role(split)
            split_counts[split_role] = split_counts.get(split_role, 0) + 1
            edge_counts["edge" if bool(edge) else "non_edge"] += 1
    return {
        "source_file": path.as_posix(),
        "top_level_shape": list(np.asarray(data, dtype=object).shape),
        "segment_count": segment_count,
        "expected_segment_shape": [KTH_RANGE_CELLS, KTH_AZIMUTH_SAMPLES],
        "class_family_counts": labels,
        "split_role_counts": split_counts,
        "edge_truncation_counts": edge_counts,
        "source_notes": {
            "radar": "SAAB SIRS 1600 FMCW",
            "carrier_frequency_ghz": 77,
            "bandwidth_mhz": 160,
            "prf_hz": KTH_PRF_HZ,
            "scan_rate_hz": 10,
            "azimuth_scan_degrees": 18,
            "azimuth_scan_time_ms": 70,
            "license": "CC-BY-4.0",
        },
    }


def _extract_kth_raw_npy(
    entry: dict[str, Any], local_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    path = local_root / KTH_RAW_FILENAME
    if not path.exists():
        return [], None
    if np is None:
        raise ValueError("numpy is required to read KTH raw .npy files")
    data = np.load(path, allow_pickle=True)
    array = np.asarray(data, dtype=object)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError(f"KTH raw .npy must be an object matrix with 6 columns; got {array.shape}")
    rows: list[dict[str, Any]] = []
    for record_idx, record in enumerate(array):
        record_times = _as_1d_array(record[3])
        gap_fraction = _track_gap_fraction(record_times)
        for segment_idx, (label, segment, range_m, _time_s, split, edge) in enumerate(
            _kth_record_segments(record)
        ):
            matrix = _kth_segment_matrix(segment)
            center_cell = matrix[KTH_RANGE_CELLS // 2, :]
            features = _spectral_features(center_cell, matrix)
            features["range_m"] = _to_float(range_m) or 0.0
            features["track_gap_fraction"] = gap_fraction
            features["edge_truncated"] = 1.0 if bool(edge) else 0.0
            class_family = _kth_class_family(label)
            split_role = _kth_split_role(split)
            raw_id = f"kth:{record_idx}:{segment_idx}"
            units = {
                "micro_doppler_peak_hz": "Hz",
                "micro_doppler_bandwidth_hz": "Hz",
                "micro_doppler_energy": "unitless",
                "spectral_entropy": "unitless",
                "return_power_db": "dB",
                "range_m": "m",
                "track_gap_fraction": "unitless",
                "edge_truncated": "bool",
            }
            for observable_name, value in features.items():
                rows.append(
                    {
                        **_normalize_observation(
                            entry,
                            raw_id=f"{raw_id}:{observable_name}",
                            class_family=class_family,
                            observable_name=observable_name,
                            value=float(value),
                            unit=units[observable_name],
                            phase_proxy="kth_raw_segment_fft",
                            split_role=split_role,
                        ),
                        "_edge_truncated": bool(edge),
                    }
                )
    return rows, _kth_raw_shape_report(path, array)


def _extract_mapped_csv(
    entry: dict[str, Any],
    path: Path,
    mapping: dict[str, tuple[str, str, str]],
    *,
    id_columns: tuple[str, ...],
    default_phase_proxy: str,
) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        raw_id = next((row[col] for col in id_columns if row.get(col)), idx)
        class_family = _clean_class_family(row)
        for source_name, (observable_name, unit, phase_proxy) in mapping.items():
            value = _to_float(row.get(source_name))
            if value is None:
                continue
            normalized.append(
                _normalize_observation(
                    entry,
                    raw_id=f"{raw_id}:{source_name}",
                    class_family=class_family,
                    observable_name=observable_name,
                    value=value,
                    unit=unit,
                    phase_proxy=phase_proxy or default_phase_proxy,
                )
            )
    return normalized


def _extract_kth_observations(
    entry: dict[str, Any], local_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    raw_rows, shape_report = _extract_kth_raw_npy(entry, local_root)
    if raw_rows:
        return raw_rows, shape_report
    mapping = {
        "micro_doppler_peak_hz": ("micro_doppler_peak_hz", "Hz", "micro_doppler"),
        "doppler_peak_hz": ("micro_doppler_peak_hz", "Hz", "micro_doppler"),
        "micro_doppler_bandwidth_hz": (
            "micro_doppler_bandwidth_hz",
            "Hz",
            "micro_doppler",
        ),
        "doppler_bandwidth_hz": ("doppler_bandwidth_hz", "Hz", "micro_doppler"),
        "micro_doppler_energy": ("micro_doppler_energy", "unitless", "micro_doppler"),
        "spectral_entropy": ("spectral_entropy", "unitless", "micro_doppler"),
        "range_m": ("range_m", "m", "kth_short_range_context"),
        "return_power_db": ("return_power_db", "dB", "kth_power_context"),
    }
    for name in ("kth_observations.csv", "observations_kth.csv", OBSERVATION_FILENAME):
        path = local_root / name
        if path.exists():
            extracted = _extract_mapped_csv(
                entry,
                path,
                mapping,
                id_columns=("sample_id", "record_id", "file_id"),
                default_phase_proxy="micro_doppler",
            )
            if extracted:
                return extracted, None
    return _observations_from_feature_value(entry, local_root), None


def _extract_ori_observations(entry: dict[str, Any], local_root: Path) -> list[dict[str, Any]]:
    mapping = {
        "range_m": ("range_m", "m", "track_product"),
        "azimuth_deg": ("azimuth_deg", "deg", "track_product"),
        "radial_velocity_mps": ("radial_velocity_mps", "m/s", "track_product"),
        "snr_db": ("snr_db", "dB", "track_product"),
        "doppler_bandwidth_hz": ("doppler_bandwidth_hz", "Hz", "doppler_spectrum"),
        "track_gap_fraction": ("track_gap_fraction", "unitless", "track_lifecycle"),
        "noise_floor_db": ("noise_floor_db", "dB", "track_product"),
        "local_noise_floor_db": ("local_noise_floor_db", "dB", "track_product"),
        "clutter_score": ("clutter_score", "unitless", "clutter"),
        "false_alarm_pressure": ("false_alarm_pressure", "unitless", "clutter"),
    }
    for name in ("ori_tracks.csv", "tracks.csv", "observations_ori.csv", OBSERVATION_FILENAME):
        path = local_root / name
        if path.exists():
            extracted = _extract_mapped_csv(
                entry,
                path,
                mapping,
                id_columns=("track_id", "sample_id", "record_id"),
                default_phase_proxy="track_product",
            )
            if extracted:
                return extracted
    return _observations_from_feature_value(entry, local_root)


def _license_verified(local_root: Path) -> bool:
    return (local_root / "LICENSE_VERIFIED").exists() or (
        local_root / "license_verified.json"
    ).exists()


def _extract_observations(
    entry: dict[str, Any], local_root: Path
) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    adapter_name = str(entry.get("adapter_name", ""))
    if adapter_name in LICENSE_GATED_ADAPTERS and not _license_verified(local_root):
        return [], "license_gated_reference_only", None
    if adapter_name == "kth_microdoppler_v1":
        observations, shape_report = _extract_kth_observations(entry, local_root)
        return observations, "measured_anchor_candidate", shape_report
    if adapter_name == "ori_outdoor_track_products_v1":
        return _extract_ori_observations(entry, local_root), "measured_anchor_candidate", None
    return _observations_from_feature_value(entry, local_root), "measured_anchor_candidate", None


def _reference_only_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for feature in entry.get("reference_features", ["unspecified"]):
        rows.append(
            {
                "dataset_id": entry["dataset_id"],
                "feature": feature,
                "metric": "distribution_distance",
                "measured_value": "",
                "reference_min": "",
                "reference_max": "",
                "distance": "",
                "status": "reference_only",
                "allowed_claim_level": entry["allowed_claim_level"],
            }
        )
    return rows


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(ordered[low])
    weight = pos - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _bootstrap_median_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) < 3:
        median = _quantile(values, 0.50)
        return median, median
    digest = hashlib.sha256(",".join(f"{value:.9g}" for value in values).encode("utf-8")).digest()
    state = int.from_bytes(digest[:8], "big")
    samples = []
    for _ in range(64):
        draw = []
        for _ in values:
            state = (6364136223846793005 * state + 1442695040888963407) % (2**64)
            draw.append(values[state % len(values)])
        samples.append(_quantile(draw, 0.50))
    return _quantile(samples, 0.05), _quantile(samples, 0.95)


def _group_observations(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        observable = str(row.get("observable_name", "")).strip()
        value = _to_float(row.get("value"))
        if observable and value is not None:
            grouped.setdefault(observable, []).append(value)
    return grouped


def _summarize_observations(grouped: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for feature, values in sorted(grouped.items()):
        if not values:
            continue
        ci_low, ci_high = _bootstrap_median_interval(values)
        summary[feature] = {
            "count": float(len(values)),
            "mean": float(mean(values)),
            "std": float(pstdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)),
            "q10": _quantile(values, 0.10),
            "q25": _quantile(values, 0.25),
            "q50": _quantile(values, 0.50),
            "q75": _quantile(values, 0.75),
            "q90": _quantile(values, 0.90),
            "max": float(max(values)),
            "bootstrap_median_ci_low": ci_low,
            "bootstrap_median_ci_high": ci_high,
        }
    return summary


def _distribution_payload(
    entry: dict[str, Any],
    normalized_rows: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    grouped = _group_observations(normalized_rows)
    by_class: dict[str, dict[str, list[float]]] = {}
    by_edge: dict[str, dict[str, list[float]]] = {}
    by_split: dict[str, dict[str, list[float]]] = {}
    for row in normalized_rows:
        class_family = str(row.get("class_family", "unknown"))
        observable = str(row.get("observable_name", ""))
        value = _to_float(row.get("value"))
        if observable and value is not None:
            by_class.setdefault(class_family, {}).setdefault(observable, []).append(value)
            edge_state = "edge" if row.get("_edge_truncated") else "non_edge"
            by_edge.setdefault(edge_state, {}).setdefault(observable, []).append(value)
            split_role = str(row.get("split_role", "unknown"))
            by_split.setdefault(split_role, {}).setdefault(observable, []).append(value)
    return {
        "dataset_id": entry["dataset_id"],
        "status": status,
        "features": _summarize_observations(grouped),
        "by_class_family": {
            family: _summarize_observations(features)
            for family, features in sorted(by_class.items())
        },
        "by_edge_truncation": {
            edge_state: _summarize_observations(features)
            for edge_state, features in sorted(by_edge.items())
        },
        "by_split_role": {
            split_role: _summarize_observations(features)
            for split_role, features in sorted(by_split.items())
        },
        "row_schema": NORMALIZED_OBSERVATION_FIELDS,
    }


def _candidate_distance_rows(
    entry: dict[str, Any], summary: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    bounds = entry.get("reference_feature_bounds", {})
    rows: list[dict[str, Any]] = []
    for feature, stats in summary.items():
        feature_bounds = bounds.get(feature, {}) if isinstance(bounds, dict) else {}
        lower = feature_bounds.get("mean_min")
        upper = feature_bounds.get("mean_max")
        measured = stats["mean"]
        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            distance = (
                0.0
                if lower <= measured <= upper
                else min(abs(measured - lower), abs(measured - upper))
            )
            status = "candidate_pass" if distance == 0.0 else "candidate_gap"
        else:
            distance = ""
            status = "measured_anchor_candidate_no_reference_bounds"
        rows.append(
            {
                "dataset_id": entry["dataset_id"],
                "feature": feature,
                "metric": "mean_distribution_distance",
                "measured_value": measured,
                "reference_min": lower if lower is not None else "",
                "reference_max": upper if upper is not None else "",
                "distance": distance,
                "status": status,
                "allowed_claim_level": entry["allowed_claim_level"],
            }
        )
    return rows


def _is_kth_entry(entry: dict[str, Any]) -> bool:
    return (
        str(entry.get("dataset_id")) == KTH_DATASET_ID
        or str(entry.get("adapter_name", "")) == "kth_microdoppler_v1"
    )


def _kth_applicability_notes(distribution: dict[str, Any]) -> dict[str, Any]:
    shape_report = distribution.get("raw_shape_report", {})
    counts = {}
    if isinstance(shape_report, dict):
        counts = {
            "processed_class_family_counts": shape_report.get("class_family_counts", {}),
            "split_role_counts": shape_report.get("split_role_counts", {}),
            "edge_truncation_counts": shape_report.get("edge_truncation_counts", {}),
        }
    return {
        "dataset_scope": (
            "KTH is applied as a public measured anchor for distribution-distance "
            "hardening of drone/confuser micro-Doppler observables only."
        ),
        "observable_applicability": KTH_APPLICABILITY_BY_OBSERVABLE,
        "class_applicability": KTH_CLASS_APPLICABILITY,
        "family_mapping": KTH_SIMULATOR_FAMILY_TARGETS,
        "counts": counts,
        "tune": [
            "Class-conditional micro-Doppler rank and overlap shape",
            "Bird hard-negative micro-Doppler realism",
            "Scan-gap and edge-truncation robustness",
        ],
        "compare_only": [
            "Short-range KTH range_m",
            "return_power_db and calibration-reflector sanity checks",
            "Human rows for false-alarm and ground-confuser reporting",
        ],
        "exclude": [
            "Exact measured truth for any Iranian platform",
            "Raw measured trace reuse",
            "Direct platform equivalence or proprietary-equivalent sensor behavior",
        ],
        "micro_doppler_energy_policy": (
            "Raw KTH energy magnitude is never used as a simulator micro_amp "
            "value. Only within-class rank/quantile shape is mapped into existing "
            "synthetic public-proxy micro_amp bounds."
        ),
        "current_family_prior_knobs": [
            "micro_bw",
            "micro_amp",
            "dropout",
        ],
        "positive_class_policy": (
            "KTH drone rows do not alter public_proxy_fixed_wing priors; they may "
            "inform drone-like hard negatives such as rc_fixed_wing."
        ),
    }


def _kth_tunable_stats(observable: str, class_family: str, stats: dict[str, Any]) -> bool:
    applicability = KTH_APPLICABILITY_BY_OBSERVABLE.get(observable)
    class_applicability = KTH_CLASS_APPLICABILITY.get(class_family, "exclude")
    return (
        applicability in {"tune", "tune_rank_shape_only"}
        and class_applicability == "tune"
        and float(stats.get("count", 0.0)) > 0.0
    )


def _rank_mapped_prior(
    *,
    observable: str,
    knob: str,
    class_family: str,
    simulator_family: str,
    stats: dict[str, Any],
) -> dict[str, Any]:
    bounds = KTH_FAMILY_PRIOR_BOUNDS[simulator_family][knob]
    lower, upper = float(bounds[0]), float(bounds[1])
    span = upper - lower
    measured_min = float(stats["min"])
    measured_max = float(stats["max"])

    def quantile_rank(name: str) -> float:
        value = float(stats[name])
        if measured_max <= measured_min:
            return 0.50
        return min(0.95, max(0.05, (value - measured_min) / (measured_max - measured_min)))

    def map_quantile(name: str) -> float:
        return lower + quantile_rank(name) * span

    reference_center = (lower + upper) / 2.0
    target_center = min(upper, max(lower, map_quantile("q50")))
    shrinkage_cap = KTH_PRIOR_SHRINKAGE_BY_FAMILY[simulator_family]
    if observable == "micro_doppler_energy":
        rank_width = max(0.0, quantile_rank("q90") - quantile_rank("q10"))
        target_width = max(rank_width * span, span * 0.45)
        target_center = reference_center
        target_q10 = max(lower, reference_center - target_width / 2.0)
        target_q90 = min(upper, reference_center + target_width / 2.0)
        shrinkage_cap = min(shrinkage_cap, 0.10)
    else:
        target_q10 = map_quantile("q10")
        target_q90 = map_quantile("q90")
        minimum_width = max(span * 0.12, 1e-6)
        if target_q90 - target_q10 < minimum_width:
            midpoint = (target_q10 + target_q90) / 2.0
            target_q10 = max(lower, midpoint - minimum_width / 2.0)
            target_q90 = min(upper, midpoint + minimum_width / 2.0)
    if target_q90 <= target_q10:
        midpoint = (target_q10 + target_q90) / 2.0
        minimum_width = max(span * 0.12, 1e-6)
        target_q10 = max(lower, midpoint - minimum_width / 2.0)
        target_q90 = min(upper, midpoint + minimum_width / 2.0)
    count = float(stats.get("count", 0.0))
    sample_shrinkage = count / (count + 20.0)
    shrinkage = min(shrinkage_cap, sample_shrinkage)
    prior = {
        "source_observable": observable,
        "source_class_family": class_family,
        "applicability": KTH_APPLICABILITY_BY_OBSERVABLE.get(observable, "exclude"),
        "count": int(count),
        "reference_center": reference_center,
        "target_center": target_center,
        "target_q10": target_q10,
        "target_q90": target_q90,
        "offset": (target_center - reference_center) * shrinkage,
        "shrinkage": shrinkage,
        "mapping": "within_class_rank_quantile_to_existing_synthetic_bounds",
        "synthetic_bounds": [lower, upper],
        "bootstrap_median_ci": [
            float(stats["bootstrap_median_ci_low"]),
            float(stats["bootstrap_median_ci_high"]),
        ],
    }
    if observable == "micro_doppler_energy":
        prior["raw_magnitude_policy"] = "rank_shape_only_not_raw_energy"
    return prior


def _fit_kth_family_priors(distribution: dict[str, Any]) -> dict[str, dict[str, Any]]:
    family_priors: dict[str, dict[str, Any]] = {}
    by_class = distribution.get("by_class_family", {})
    if not isinstance(by_class, dict):
        return family_priors
    for class_family, observable_stats in by_class.items():
        simulator_families = KTH_SIMULATOR_FAMILY_TARGETS.get(str(class_family), [])
        if not simulator_families or not isinstance(observable_stats, dict):
            continue
        for observable, stats in observable_stats.items():
            knob = SIMULATOR_OBSERVABLE_MAP.get(str(observable))
            if not knob:
                continue
            if not _kth_tunable_stats(str(observable), str(class_family), stats):
                continue
            for simulator_family in simulator_families:
                if knob not in KTH_FAMILY_PRIOR_BOUNDS.get(simulator_family, {}):
                    continue
                family_priors.setdefault(simulator_family, {})[knob] = _rank_mapped_prior(
                    observable=str(observable),
                    knob=knob,
                    class_family=str(class_family),
                    simulator_family=simulator_family,
                    stats=stats,
                )
    return family_priors


def _fit_calibration_coefficients(
    entry: dict[str, Any],
    distribution: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    if _is_kth_entry(entry):
        family_priors = _fit_kth_family_priors(distribution)
        return {
            "format": "echoforge.real_anchor_priors.v2",
            "dataset_id": entry["dataset_id"],
            "status": status,
            "allowed_claim_level": entry["allowed_claim_level"],
            "claim_boundary": entry["claim_boundary"],
            "simulator_priors": {},
            "simulator_priors_by_family": family_priors,
            "applicability_notes": _kth_applicability_notes(distribution),
            "policy": (
                "KTH measured anchors adjust class-conditional public-proxy hard-negative "
                "priors and gap analysis only. They do not establish exact Iranian-platform "
                "truth, proprietary-equivalent behavior, or classified fidelity."
            ),
        }
    simulator_priors: dict[str, dict[str, Any]] = {}
    for observable, stats in distribution.get("features", {}).items():
        knob = SIMULATOR_OBSERVABLE_MAP.get(observable)
        if not knob:
            continue
        count = float(stats.get("count", 0.0))
        shrinkage = count / (count + 20.0)
        reference_center = SYNTHETIC_PRIOR_CENTERS.get(knob, float(stats["q50"]))
        target_center = float(stats["q50"])
        simulator_priors[knob] = {
            "source_observable": observable,
            "count": int(count),
            "reference_center": reference_center,
            "target_center": target_center,
            "target_q10": float(stats["q10"]),
            "target_q90": float(stats["q90"]),
            "offset": (target_center - reference_center) * shrinkage,
            "shrinkage": shrinkage,
            "bootstrap_median_ci": [
                float(stats["bootstrap_median_ci_low"]),
                float(stats["bootstrap_median_ci_high"]),
            ],
        }
    return {
        "format": "echoforge.real_anchor_priors.v1",
        "dataset_id": entry["dataset_id"],
        "status": status,
        "allowed_claim_level": entry["allowed_claim_level"],
        "claim_boundary": entry["claim_boundary"],
        "simulator_priors": simulator_priors,
        "simulator_priors_by_family": {},
        "applicability_notes": {
            "dataset_scope": "Generic measured-anchor priors use legacy global simulator_priors.",
            "fallback_policy": "Generator uses family-specific priors when present, then these global priors.",
        },
        "policy": (
            "Measured anchors adjust public-proxy simulator priors and gap analysis only; "
            "they do not establish platform-specific behavior, proprietary-equivalent behavior, "
            "or classified fidelity."
        ),
    }


def _sim_real_gap_rows(
    entry: dict[str, Any], summary: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bounds = entry.get("reference_feature_bounds", {})
    for observable, stats in summary.items():
        feature_bounds = bounds.get(observable, {}) if isinstance(bounds, dict) else {}
        ref_min = _to_float(feature_bounds.get("mean_min"))
        ref_max = _to_float(feature_bounds.get("mean_max"))
        ref_center = None
        if ref_min is not None and ref_max is not None:
            ref_center = (ref_min + ref_max) / 2.0
            gap = abs(float(stats["mean"]) - ref_center)
            status = (
                "candidate_pass" if ref_min <= float(stats["mean"]) <= ref_max else "candidate_gap"
            )
        else:
            gap = ""
            status = "measured_anchor_no_reference_distribution"
        rows.append(
            {
                "dataset_id": entry["dataset_id"],
                "observable_name": observable,
                "metric": "wasserstein_1_proxy_mean_gap",
                "measured_value": stats["mean"],
                "synthetic_reference_value": ref_center if ref_center is not None else "",
                "gap_value": gap,
                "status": status,
                "allowed_claim_level": entry["allowed_claim_level"],
            }
        )
        q50 = float(stats["q50"])
        ratio = "" if ref_center in (None, 0.0) else q50 / float(ref_center)
        rows.append(
            {
                "dataset_id": entry["dataset_id"],
                "observable_name": observable,
                "metric": "quantile_ratio_q50_to_reference",
                "measured_value": q50,
                "synthetic_reference_value": ref_center if ref_center is not None else "",
                "gap_value": ratio,
                "status": status,
                "allowed_claim_level": entry["allowed_claim_level"],
            }
        )
    return rows


def _reference_gap_rows(entry: dict[str, Any], status: str) -> list[dict[str, Any]]:
    return [
        {
            "dataset_id": entry["dataset_id"],
            "observable_name": feature,
            "metric": "unavailable",
            "measured_value": "",
            "synthetic_reference_value": "",
            "gap_value": "",
            "status": status,
            "allowed_claim_level": entry["allowed_claim_level"],
        }
        for feature in entry.get("reference_features", ["unspecified"])
    ]


def _write_alignment_report(
    output_dir: Path,
    entry: dict[str, Any],
    status: str,
    normalized_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
) -> None:
    unsupported = [
        "Exact measured truth for any named object or platform",
        "Proprietary-equivalent radar behavior",
        "Classified fidelity",
    ]
    payload = {
        "dataset_id": entry["dataset_id"],
        "status": status,
        "observation_count": len(normalized_rows),
        "gap_rows": gap_rows,
        "claim_boundary": entry["claim_boundary"],
        "unsupported_claims": unsupported,
        "paper_ready_summary": (
            "Available measured anchors support public-proxy distribution hardening and "
            "gap reporting only. Unavailable or gated anchors are reported as unsupported."
        ),
    }
    _write_json(output_dir / "real_anchor_alignment_report.json", payload)

    lines = [
        "# Real Anchor Alignment Report",
        "",
        f"Dataset: `{entry['dataset_id']}`",
        f"Status: `{status}`",
        f"Normalized observation rows: {len(normalized_rows)}",
        "",
        "Measured anchors are used for public-proxy simulator hardening and gap analysis only.",
        "They do not validate platform-specific behavior, proprietary radar behavior, or classified fidelity.",
        "",
        "## Gap Metrics",
    ]
    if gap_rows:
        lines.extend(
            f"- {row['observable_name']} `{row['metric']}`: `{row['status']}`" for row in gap_rows
        )
    else:
        lines.append("- No measured-anchor rows were available.")
    lines.extend(["", "## Unsupported Claims"])
    lines.extend(f"- {item}" for item in unsupported)
    output_dir.joinpath("real_anchor_alignment_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_dataset_report(
    entry: dict[str, Any],
    *,
    raw_root: Path | None = None,
    out_root: Path = Path("outputs/real-data"),
    run_id: str = "reference-only",
) -> DatasetReport:
    root = default_raw_root() if raw_root is None else Path(raw_root)
    local_root = dataset_root(root, str(entry["dataset_id"]))
    output_dir = derived_output_dir(Path(out_root), str(entry["dataset_id"]), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_id": entry["dataset_id"],
        "display_name": entry["display_name"],
        "source_url": entry["source_url"],
        "adapter_name": entry["adapter_name"],
        "allowed_claim_level": entry["allowed_claim_level"],
        "claim_boundary": entry["claim_boundary"],
        "license_status": entry["license_status"],
        "local_root": local_root.as_posix(),
        "raw_data_tracked_in_git": False,
        "normalized_samples_tracked_in_git": False,
    }

    normalized_rows, candidate_status, shape_report = _extract_observations(entry, local_root)
    if not local_root.exists():
        status = "reference_only"
        message = "No local measured-anchor directory was found; emitted reference-only metadata."
    elif candidate_status == "license_gated_reference_only":
        status = candidate_status
        message = "Local files were not read because access/license verification is not present."
    elif not normalized_rows:
        status = "reference_only"
        message = "No supported local observation CSV was found; emitted reference-only metadata."
    else:
        status = "measured_anchor_candidate"
        message = (
            "Local measured-anchor observations were normalized; no measured traces were copied."
        )

    manifest.update(
        {
            "calibration_anchor_status": status,
            "local_status": "observations_present" if normalized_rows else "not_available",
            "message": message,
        }
    )
    _write_json(output_dir / "anchor_manifest.json", manifest)
    _write_json(
        output_dir / "raw_shape_report.json",
        shape_report
        or {
            "dataset_id": entry["dataset_id"],
            "status": status,
            "source_file": "",
            "message": "No supported local raw shape was available.",
        },
    )
    _write_csv(
        output_dir / "anchor_observations.csv",
        normalized_rows,
        NORMALIZED_OBSERVATION_FIELDS,
    )

    distribution = _distribution_payload(entry, normalized_rows, status)
    if shape_report:
        distribution["raw_shape_report"] = shape_report
    _write_json(output_dir / "feature_distributions.json", distribution)
    coefficients = _fit_calibration_coefficients(entry, distribution, status)
    _write_json(output_dir / "calibration_coefficients.json", coefficients)

    if normalized_rows:
        summary = distribution["features"]
        calibration_rows = _candidate_distance_rows(entry, summary)
        gap_rows = _sim_real_gap_rows(entry, summary)
    else:
        calibration_rows = _reference_only_rows(entry)
        gap_rows = _reference_gap_rows(entry, status)
    _write_csv(output_dir / "calibration_distance.csv", calibration_rows, CALIBRATION_FIELDS)
    _write_csv(output_dir / "sim_real_gap.csv", gap_rows, SIM_REAL_GAP_FIELDS)
    _write_alignment_report(output_dir, entry, status, normalized_rows, gap_rows)

    return DatasetReport(
        dataset_id=str(entry["dataset_id"]),
        status=status,
        output_dir=output_dir,
        message=message,
    )
