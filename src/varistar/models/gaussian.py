"""
varistar.models.gaussian
========================
Gaussian and Super-Gaussian model functions for eclipsing binary light curve
fitting.

All models handle circular phase wrapping internally so they work correctly
near phase = 0 / 1 boundaries without any pre-processing by the caller.

Shape parameter guide for Super-Gaussian models
------------------------------------------------
shape ≈ 2.0  →  standard bell curve
shape > 2.0  →  flat-bottomed / boxy  (detached EB)
shape < 2.0  →  pointed / V-shaped   (contact EB)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Phase-wrapping helper (shared by all models)
# ---------------------------------------------------------------------------


def _phase_dist(x: np.ndarray, center: float) -> np.ndarray:
    """Circular distance on the unit interval: handles 0/1 wrap-around."""
    delta = np.abs(x - center)
    return np.minimum(delta, 1.0 - delta)


# ---------------------------------------------------------------------------
# Single-dip models
# ---------------------------------------------------------------------------


def gaussian_model(
    x: np.ndarray,
    baseline: float,
    amp: float,
    center: float,
    width: float,
) -> np.ndarray:
    """
    Inverted Gaussian dip for a single-eclipse light curve.

    m(x) = baseline - amp · exp(-0.5 · (Δ/width)²)
    where Δ is the circular phase distance to *center*.
    """
    delta = _phase_dist(x, center)
    return baseline - amp * np.exp(-0.5 * (delta / width) ** 2)


def super_gaussian_model(
    x: np.ndarray,
    baseline: float,
    amp: float,
    center: float,
    width: float,
    shape: float,
) -> np.ndarray:
    """
    Generalised Normal (Super-Gaussian) dip for a single eclipse.

    m(x) = baseline - amp · exp(-0.5 · |Δ/width|^shape)

    A shape parameter != 2 allows fitting boxy (detached) or pointed (contact)
    eclipse profiles more accurately than a standard Gaussian.
    """
    delta = _phase_dist(x, center)
    safe_width = np.maximum(width, 1e-4)
    return baseline - amp * np.exp(-0.5 * np.abs(delta / safe_width) ** shape)


# ---------------------------------------------------------------------------
# Double-dip models  (primary + secondary eclipse)
# ---------------------------------------------------------------------------


def double_gaussian_model(
    x: np.ndarray,
    baseline: float,
    amp1: float,
    center1: float,
    width1: float,
    amp2: float,
    center2: float,
    width2: float,
) -> np.ndarray:
    """Double inverted Gaussian for primary + secondary eclipse."""
    g1 = amp1 * np.exp(-0.5 * (_phase_dist(x, center1) / width1) ** 2)
    g2 = amp2 * np.exp(-0.5 * (_phase_dist(x, center2) / width2) ** 2)
    return baseline - g1 - g2


def double_super_gaussian_model(
    x: np.ndarray,
    baseline: float,
    amp1: float,
    cent1: float,
    wid1: float,
    shape1: float,
    amp2: float,
    cent2: float,
    wid2: float,
    shape2: float,
) -> np.ndarray:
    """Double Super-Gaussian for primary + secondary eclipse."""
    d1 = _phase_dist(x, cent1)
    d2 = _phase_dist(x, cent2)
    g1 = amp1 * np.exp(-0.5 * np.abs(d1 / np.maximum(wid1, 1e-4)) ** shape1)
    g2 = amp2 * np.exp(-0.5 * np.abs(d2 / np.maximum(wid2, 1e-4)) ** shape2)
    return baseline - g1 - g2


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------


def fit_double_super_gaussian(
    phase: np.ndarray,
    mag: np.ndarray,
    maxfev: int = 5000,
) -> tuple[np.ndarray | None, float]:
    """
    Fit a Double Super-Gaussian to a phase-folded eclipsing binary light curve.

    Uses physically motivated initial guesses and bounded optimisation to
    prevent the shape parameter from diverging.

    Parameters
    ----------
    phase : np.ndarray
        Phase values in [0, 1].
    mag : np.ndarray
        Magnitude values (inverted: deeper = larger value).
    maxfev : int
        Maximum function evaluations for scipy curve_fit.

    Returns
    -------
    popt : np.ndarray | None
        Best-fit parameters; None on failure.
    mea : float
        Mean Absolute Error (999.0 on failure).
    """
    baseline = float(np.percentile(mag, 10))
    amp = float(np.ptp(mag))
    center = float(phase[np.argmax(mag)])

    # [base, amp1, cen1, wid1, shape1, amp2, cen2, wid2, shape2]
    p0 = [baseline, amp, center, 0.05, 2.0, amp * 0.5, (center + 0.5) % 1.0, 0.05, 2.0]
    lower = [-np.inf, 0, 0, 0.001, 0.5, 0, 0, 0.001, 0.5]
    upper = [np.inf, np.inf, 1, 0.5, 10.0, np.inf, 1, 0.5, 10.0]

    try:
        popt, _ = curve_fit(
            double_super_gaussian_model,
            phase,
            mag,
            p0=p0,
            bounds=(lower, upper),
            maxfev=maxfev,
        )
        y_pred = double_super_gaussian_model(phase, *popt)
        mea = float(np.mean(np.abs(mag - y_pred)))
        return popt, mea
    except Exception:
        return None, 999.0
