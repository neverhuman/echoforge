# boundary: python-ai-service-ml-slice
from __future__ import annotations

from dataclasses import dataclass
import cmath
import math
from typing import Iterable, List, Sequence


ComplexSample = complex


@dataclass(frozen=True)
class LfmChirp:
    sample_rate_hz: float
    pulse_width_s: float
    bandwidth_hz: float
    carrier_hz: float = 0.0
    initial_phase_rad: float = 0.0

    def sample_count(self) -> int:
        return int(round(self.sample_rate_hz * self.pulse_width_s))

    def samples(self) -> List[ComplexSample]:
        return lfm_chirp(self)


@dataclass(frozen=True)
class CfarParams:
    training_cells: int
    guard_cells: int
    pfa: float


def lfm_chirp(config: LfmChirp) -> List[ComplexSample]:
    sample_count = config.sample_count()
    if sample_count <= 0:
        return []
    dt = 1.0 / config.sample_rate_hz
    slope = config.bandwidth_hz / config.pulse_width_s
    center = config.pulse_width_s / 2.0
    samples: List[ComplexSample] = []
    for index in range(sample_count):
        t = index * dt
        centered_t = t - center
        phase = config.initial_phase_rad + 2.0 * math.pi * (
            config.carrier_hz * t + 0.5 * slope * centered_t * centered_t
        )
        samples.append(cmath.rect(1.0, phase))
    return samples


def matched_filter(received: Sequence[ComplexSample], reference: Sequence[ComplexSample]) -> List[ComplexSample]:
    if not received or not reference:
        return []
    output = [0j] * (len(received) + len(reference) - 1)
    ref_conj_rev = [sample.conjugate() for sample in reversed(reference)]
    for i, sample in enumerate(received):
        for j, ref_sample in enumerate(ref_conj_rev):
            output[i + j] += sample * ref_sample
    return output


def ca_cfar_scale(training_cells: int, pfa: float) -> float:
    if training_cells <= 0:
        return 0.0
    n = float(training_cells)
    return n * (pfa ** (-1.0 / n) - 1.0)


def ca_cfar_1d(power: Sequence[float], params: CfarParams) -> List[dict]:
    decisions: List[dict] = []
    alpha = ca_cfar_scale(params.training_cells, params.pfa)
    window = params.training_cells + params.guard_cells
    for index, value in enumerate(power):
        if index < window or index + window >= len(power):
            decisions.append(
                {
                    "index": index,
                    "evaluated": False,
                    "detected": False,
                    "statistic": float(value),
                    "threshold": math.inf,
                    "noise_estimate": 0.0,
                }
            )
            continue
        left_start = index - window
        left_end = index - params.guard_cells
        right_start = index + params.guard_cells + 1
        right_end = index + window + 1
        train_cells = list(power[left_start:left_end]) + list(power[right_start:right_end])
        noise_estimate = sum(train_cells) / len(train_cells) if train_cells else 0.0
        threshold = alpha * noise_estimate
        decisions.append(
            {
                "index": index,
                "evaluated": True,
                "detected": float(value) > threshold,
                "statistic": float(value),
                "threshold": threshold,
                "noise_estimate": noise_estimate,
            }
        )
    return decisions

