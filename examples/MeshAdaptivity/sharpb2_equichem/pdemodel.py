"""Sharp-B equilibrium-air model with guarded material-table coordinates."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import sympy as sp


_BASE_PATH = Path(__file__).resolve().parents[1] / "isoq2d_equichem" / "pdemodel.py"
_SPEC = importlib.util.spec_from_file_location("_sharpb2_equichem_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load equilibrium-air model from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

# Reuse the common axisymmetric equilibrium-air operators.
mass = _BASE.mass
flux = _BASE.flux
source = _BASE.source
avfield = _BASE.avfield
fbou = _BASE.fbou
ubou = _BASE.ubou
fbouhdg = _BASE.fbouhdg
initu = _BASE.initu
initv = _BASE.initv
initw = _BASE.initw
visscalars = _BASE.visscalars
visvectors = _BASE.visvectors


def materialstate(u, q, w, v, x, t, mu, eta):
    """Limit table coordinates strictly inside the high-Mach database bounds."""
    rho = u[0]
    velocity_z = u[1] / rho
    velocity_r = u[2] / rho
    internal_energy = u[3] / rho - 0.5 * (
        velocity_z * velocity_z + velocity_r * velocity_r
    )

    rho_min = 0.0001 / mu[0]
    rho_max = 20.0 / mu[0]
    limited_rho = _BASE._limiting(rho, rho_min, rho_max, 1.0e2, rho_min)

    energy_safety = 1000.0
    energy_min = -150000.0 + energy_safety
    energy_max = 20000000.0 - energy_safety
    limited_energy = _BASE._limiting(
        mu[3] * internal_energy,
        energy_min,
        energy_max,
        1.0e2,
        energy_min,
    )
    return np.array([sp.log(mu[0] * limited_rho), limited_energy])
