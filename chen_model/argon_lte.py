# Approximate LTE argon thermophysical / transport properties vs T.
# Open literature style (not ArcWeld proprietary tables).
# Suitable for demo MHD arcs; without metal vapour cooling arcs run hotter than MIG.

from __future__ import annotations

import numpy as np

# Sample temperatures [K]
_T = np.array(
    [
        300.0,
        1000.0,
        2000.0,
        3000.0,
        5000.0,
        7000.0,
        9000.0,
        11000.0,
        13000.0,
        15000.0,
        18000.0,
        20000.0,
        25000.0,
    ],
    dtype=np.float64,
)

# Electrical conductivity [S/m] — rises sharply after ionization
_SIGMA = np.array(
    [
        1.0e-8,
        1.0e-6,
        1.0e-2,
        10.0,
        2.0e2,
        1.5e3,
        4.0e3,
        7.0e3,
        9.5e3,
        1.1e4,
        1.3e4,
        1.4e4,
        1.5e4,
    ],
    dtype=np.float64,
)

# Thermal conductivity [W/(m·K)]
_K = np.array(
    [
        0.018,
        0.05,
        0.10,
        0.25,
        0.8,
        1.8,
        2.8,
        3.5,
        4.0,
        4.5,
        5.5,
        6.0,
        7.0,
    ],
    dtype=np.float64,
)

# Specific heat [J/(kg·K)] (effective, with ionization peaks flattened)
_CP = np.array(
    [
        520.0,
        520.0,
        550.0,
        800.0,
        2500.0,
        5000.0,
        4000.0,
        3500.0,
        3200.0,
        3000.0,
        2800.0,
        2700.0,
        2600.0,
    ],
    dtype=np.float64,
)

# Dynamic viscosity [Pa·s]
_MU = np.array(
    [
        2.2e-5,
        5.0e-5,
        8.0e-5,
        1.2e-4,
        2.0e-4,
        2.5e-4,
        2.8e-4,
        3.0e-4,
        3.2e-4,
        3.4e-4,
        3.6e-4,
        3.7e-4,
        3.9e-4,
    ],
    dtype=np.float64,
)

# Mass density [kg/m³] at ~1 atm (ideal-gas like drop)
_RHO = np.array(
    [
        1.6,
        0.48,
        0.24,
        0.16,
        0.095,
        0.068,
        0.053,
        0.043,
        0.037,
        0.032,
        0.027,
        0.024,
        0.019,
    ],
    dtype=np.float64,
)

# Net radiative emission coefficient U [W/m³] (order-of-magnitude Ar NEC)
_U = np.array(
    [
        0.0,
        0.0,
        1.0e2,
        1.0e4,
        5.0e6,
        5.0e7,
        2.0e8,
        5.0e8,
        1.0e9,
        1.5e9,
        2.5e9,
        3.0e9,
        4.0e9,
    ],
    dtype=np.float64,
)


def _interp(T: np.ndarray, table: np.ndarray) -> np.ndarray:
    Tc = np.clip(np.asarray(T, dtype=np.float64), _T[0], _T[-1])
    return np.interp(Tc, _T, table)


def sigma(T: np.ndarray) -> np.ndarray:
    return _interp(T, _SIGMA)


def k_th(T: np.ndarray) -> np.ndarray:
    return _interp(T, _K)


def cp(T: np.ndarray) -> np.ndarray:
    return _interp(T, _CP)


def mu(T: np.ndarray) -> np.ndarray:
    return _interp(T, _MU)


def rho(T: np.ndarray) -> np.ndarray:
    return _interp(T, _RHO)


def U_rad(T: np.ndarray) -> np.ndarray:
    return _interp(T, _U)


def props(T: np.ndarray) -> dict:
    """Return dict of all interpolated plasma properties at T."""
    return {
        "sigma": sigma(T),
        "k": k_th(T),
        "cp": cp(T),
        "mu": mu(T),
        "rho": rho(T),
        "U": U_rad(T),
    }
