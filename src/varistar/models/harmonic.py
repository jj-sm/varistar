"""
varistar.models.harmonic
========================
Fourier (harmonic) series model functions for light curve fitting.

All functions are pure numpy operations with no class dependencies,
making them independently testable and importable anywhere.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


def fourier_series(x: np.ndarray, *params) -> np.ndarray:
    """
    Evaluate a Fourier series at phase values x.

    Model:  m(x) = A0 + Σ_n [ A_n·cos(2πnx) + B_n·sin(2πnx) ]

    Parameters
    ----------
    x : np.ndarray
        Phase values in [0, 1].
    *params : float
        Flat parameter array: [offset, A1, B1, A2, B2, ..., An, Bn].
        Length must be 1 + 2·n_harmonics.

    Returns
    -------
    np.ndarray
        Model magnitudes at each phase value.
    """
    offset = params[0]
    result = np.full_like(x, offset, dtype=float)
    n_harmonics = (len(params) - 1) // 2
    for i in range(n_harmonics):
        a = params[2 * i + 1]
        b = params[2 * i + 2]
        result += a * np.cos(2.0 * np.pi * (i + 1) * x) + b * np.sin(
            2.0 * np.pi * (i + 1) * x
        )
    return result


# ---------------------------------------------------------------------------
# Fitting helper
# ---------------------------------------------------------------------------


def fit_fourier(
    phase: np.ndarray,
    mag: np.ndarray,
    n_harmonics: int = 4,
    maxfev: int = 5000,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """
    Fit a Fourier series to phase-folded photometry.

    Parameters
    ----------
    phase : np.ndarray
        Phase values in [0, 1].
    mag : np.ndarray
        Magnitude values.
    n_harmonics : int
        Number of harmonic terms to include.
    maxfev : int
        Maximum function evaluations passed to scipy curve_fit.

    Returns
    -------
    popt : np.ndarray | None
        Best-fit parameters; None if the fit failed.
    residuals : np.ndarray | None
        (mag - model) residuals; None if the fit failed.
    mea : float
        Mean Absolute Error of the fit (999.0 on failure).
    """
    p0 = [np.mean(mag)] + [0.0] * (2 * n_harmonics)
    try:
        popt, _ = curve_fit(fourier_series, phase, mag, p0=p0, maxfev=maxfev)
        y_pred = fourier_series(phase, *popt)
        residuals = mag - y_pred
        mea = float(np.mean(np.abs(residuals)))
        return popt, residuals, mea
    except Exception:
        return None, None, 999.0


# ---------------------------------------------------------------------------
# Fourier decomposition parameters
# ---------------------------------------------------------------------------


def amplitude_r21(popt: np.ndarray) -> float:
    """
    Compute the R21 Fourier amplitude ratio: A2 / A1.

    R21 is a key feature for variable-star classification (e.g. RR Lyrae
    subtypes, Cepheids).  Requires at least 2 harmonics in *popt*.

    Parameters
    ----------
    popt : np.ndarray
        Parameter array from `fit_fourier` (length ≥ 5).

    Returns
    -------
    float
        R21 = sqrt(A2² + B2²) / sqrt(A1² + B1²), or 0.0 if A1 ≈ 0.
    """
    if len(popt) < 5:
        return 0.0
    a1, b1 = popt[1], popt[2]
    a2, b2 = popt[3], popt[4]
    amp1 = np.hypot(a1, b1)
    amp2 = np.hypot(a2, b2)
    return float(amp2 / (amp1 + 1e-9))


def phase_phi21(popt: np.ndarray) -> float:
    """
    Compute the φ21 Fourier phase difference: φ2 - 2·φ1  (mod 2π).

    Parameters
    ----------
    popt : np.ndarray
        Parameter array from `fit_fourier` (length ≥ 5).

    Returns
    -------
    float
        φ21 in radians ∈ [0, 2π).
    """
    if len(popt) < 5:
        return 0.0
    phi1 = np.arctan2(popt[2], popt[1])
    phi2 = np.arctan2(popt[4], popt[3])
    return float((phi2 - 2.0 * phi1) % (2.0 * np.pi))
