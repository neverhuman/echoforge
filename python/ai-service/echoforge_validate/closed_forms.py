"""Closed-form analytic ground truth for canonical PEC scatterers.

References (cited, not copied):
  * Knott, Shaeffer, Tuley, *Radar Cross Section* 2nd ed.
  * Ruck et al., *Radar Cross Section Handbook* (1970).
"""

from __future__ import annotations

import math


def _sinc(x: float) -> float:
    return 1.0 if abs(x) < 1e-12 else math.sin(x) / x


def flat_plate_sigma(width_m: float, height_m: float, wavelength_m: float,
                     theta_rad: float = 0.0, phi_rad: float = 0.0) -> float:
    area = width_m * height_m
    k = 2.0 * math.pi / wavelength_m
    prefactor = 4.0 * math.pi * area * area / (wavelength_m * wavelength_m)
    sw = _sinc(k * width_m * math.sin(theta_rad))
    sh = _sinc(k * height_m * math.sin(phi_rad))
    return (
        prefactor
        * math.cos(theta_rad) ** 2
        * math.cos(phi_rad) ** 2
        * sw ** 2
        * sh ** 2
    )


def dihedral_peak_sigma(width_m: float, height_m: float, wavelength_m: float) -> float:
    return (
        8.0
        * math.pi
        * width_m ** 2
        * height_m ** 2
        / (wavelength_m * wavelength_m)
    )


def trihedral_square_peak(edge_m: float, wavelength_m: float) -> float:
    return 12.0 * math.pi * edge_m ** 4 / (wavelength_m * wavelength_m)


def trihedral_triangular_peak(edge_m: float, wavelength_m: float) -> float:
    return 4.0 * math.pi * edge_m ** 4 / (3.0 * wavelength_m * wavelength_m)


def cylinder_broadside_sigma(radius_m: float, length_m: float, wavelength_m: float) -> float:
    return 2.0 * math.pi * radius_m * length_m ** 2 / wavelength_m


def cone_tip_sigma(half_angle_rad: float, wavelength_m: float) -> float:
    t = math.tan(half_angle_rad / 2.0)
    return (wavelength_m * wavelength_m) / (16.0 * math.pi) * t ** 4
