"""Cross-check the Rust-CSV sphere truth against the scipy Mie oracle.

Skipped when scipy is unavailable.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python", "ai-service"))

CSV_PATH = os.path.join(
    REPO_ROOT,
    "tests",
    "science",
    "fixtures",
    "sphere_truth.csv",
)


class TestMieCrossCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import scipy  # type: ignore  # noqa: F401
        except Exception:  # pragma: no cover
            raise unittest.SkipTest("scipy not available")
        from echoforge_validate import mie  # noqa: F401

    def test_csv_loads(self):
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertGreaterEqual(len(rows), 10)

    def test_optical_regime_agrees_within_loose_tol(self):
        from echoforge_validate import mie
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["regime"] != "optical":
                continue
            a = float(r["a_m"])
            ka = float(r["ka"])
            lam = 2.0 * math.pi * a / ka
            scipy_sigma = mie.pec_sphere_sigma(a, lam)
            csv_sigma = float(r["sigma_m2"])
            # CSV stores asymptotes; Mie may differ ±3 dB at finite ka.
            db_csv = 10.0 * math.log10(csv_sigma)
            db_scipy = 10.0 * math.log10(scipy_sigma)
            self.assertLess(abs(db_csv - db_scipy), 3.0)

    def test_rayleigh_regime_agrees_tightly(self):
        from echoforge_validate import mie
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["regime"] != "rayleigh":
                continue
            a = float(r["a_m"])
            ka = float(r["ka"])
            lam = 2.0 * math.pi * a / ka
            scipy_sigma = mie.pec_sphere_sigma(a, lam)
            csv_sigma = float(r["sigma_m2"])
            # Both should converge to 9π a² (ka)^4 to high precision.
            rel = abs(scipy_sigma - csv_sigma) / max(csv_sigma, 1e-30)
            self.assertLess(rel, 0.05, f"a={a} ka={ka} scipy={scipy_sigma} csv={csv_sigma}")


if __name__ == "__main__":
    unittest.main()
