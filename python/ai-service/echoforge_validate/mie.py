# boundary: python-ai-service-ml-slice
"""PEC Mie sphere reference via scipy (oracle for the Rust implementation).

References (cited, not copied):
  * Bohren & Huffman, *Absorption and Scattering of Light by Small Particles*,
    Wiley 1983, eq. 4.56 / 4.83 (PEC limit and back-scatter sum).
  * Wiscombe (1980), Appl. Opt. 19, 1505.

Implementation uses scipy's spherical Bessel functions. scipy is imported
lazily so the rest of this package can load without it; functions raise
``ImportError`` if invoked when scipy is missing.
"""

from __future__ import annotations

from typing import List, Tuple

import math


def _require_scipy():
    try:
        from scipy.special import spherical_jn, spherical_yn  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "scipy is required for echoforge_validate.mie; install scipy or skip"
        ) from exc
    return spherical_jn, spherical_yn


def n_max(ka: float) -> int:
    """Wiscombe truncation: ceil(ka + 4·ka^(1/3) + 2), floor 5."""
    v = ka + 4.0 * ka ** (1.0 / 3.0) + 2.0
    return max(5, int(math.ceil(v)))


def mie_pec_coeffs(ka: float) -> Tuple[List[complex], List[complex]]:
    """Return (a_n, b_n) PEC scattering coefficients for n = 1..N_max.

    Uses the relations
        a_n = j_n(ka) / h_n^{(1)}(ka)
        b_n = [ka·j_n(ka)]' / [ka·h_n^{(1)}(ka)]'
    with h_n^{(1)} = j_n + i·y_n.
    """
    spherical_jn, spherical_yn = _require_scipy()
    nmax = n_max(ka)
    a_n: List[complex] = []
    b_n: List[complex] = []
    for n in range(1, nmax + 1):
        jn = spherical_jn(n, ka)
        yn = spherical_yn(n, ka)
        # ψ_n(x) = x·j_n(x), ξ_n(x) = x·h_n^{(1)}(x).
        psi = ka * jn
        xi = ka * complex(jn, yn)
        # Derivatives via the recurrence f_n' = f_{n-1} − (n/x)·f_n
        # implemented through scipy's derivative argument.
        djn = spherical_jn(n, ka, derivative=True)
        dyn = spherical_yn(n, ka, derivative=True)
        # d/dx [x·f_n(x)] = f_n(x) + x·f_n'(x).
        psi_d = jn + ka * djn
        xi_d = complex(jn + ka * djn, yn + ka * dyn)
        a_n.append(psi / xi)
        b_n.append(psi_d / xi_d)
    return a_n, b_n


def pec_sphere_sigma(radius_m: float, wavelength_m: float) -> float:
    """Backscatter RCS in m² for PEC sphere via Mie series."""
    ka = 2.0 * math.pi * radius_m / wavelength_m
    a_n, b_n = mie_pec_coeffs(ka)
    s = 0.0 + 0.0j
    for idx, (a, b) in enumerate(zip(a_n, b_n)):
        n = idx + 1
        sign = 1.0 if (n % 2 == 0) else -1.0
        s += sign * (n + 0.5) * (b - a)
    lam = wavelength_m
    return (lam * lam) / math.pi * abs(s) ** 2
