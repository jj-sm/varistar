"""
varistar.classify.eb_detector
==============================
Eclipsing Binary (EB) detection and morphology classification.

The primary entry point is ``score_eb()``, which combines a Fourier fit with
a density-in-dip heuristic to flag light curves that look like eclipsing
binaries.

All functions accept raw numpy arrays or a ``LightCurve`` instance, so they
can be called from anywhere in the package without circular imports.
"""

from __future__ import annotations

import numpy as np

from varistar.models.harmonic import fourier_series, fit_fourier


# ---------------------------------------------------------------------------
# Core EB scorer
# ---------------------------------------------------------------------------


def score_eb(
    lc,
    period: float | None = None,
    mag_col: str | None = None,
    mea_range: tuple[float, float] = (0.03, 0.8),
) -> tuple[bool, float, np.ndarray | None]:
    """
    Determine whether a light curve is consistent with an eclipsing binary.

    Algorithm
    ---------
    1. Phase-fold at *period*.
    2. Fit a Fourier series and compute MEA (Mean Absolute Error).
    3. If MEA is within *mea_range*, proceed to the density check.
    4. **Density check**: the Fourier model defines a "dip" region (phases
       where the model exceeds mean + 1σ).  If the density of *well-fitted*
       observations *inside* the dip is lower than the global well-fitted
       density, the shape is consistent with a sharp eclipse that Fourier
       cannot model accurately → likely EB.

    Parameters
    ----------
    lc : LightCurve
        The light curve to test.
    period : float | None
        Test period.  Defaults to ``lc.periods[0]`` if available.
    mag_col : str | None
        Magnitude column override.
    mea_range : tuple[float, float]
        Acceptable MEA window.  Curves outside this range are not EBs.

    Returns
    -------
    (is_eb, mea, popt)
        ``is_eb``  — True if classified as an EB.
        ``mea``    — MEA of the Fourier fit (999.0 on failure).
        ``popt``   — Fourier best-fit parameters (None on failure).
    """
    # Resolve period
    if period is None:
        if not lc.periods:
            lc.run_ls()
        period = lc.periods[0] if lc.periods else None

    if period is None or period <= 0.0:
        return False, 999.0, None

    # Extract data
    mag_col = mag_col or lc.timeseries.mag_col
    ts = lc.timeseries
    t = ts.timeseries_df[ts.time_col].to_numpy()
    y = ts.timeseries_df[mag_col].to_numpy()
    phase = ((t - t.min()) / period) % 1.0

    # --- A. Fourier fit and MEA (4 harmonics) ---
    popt, residuals, mea = fit_fourier(phase, y, n_harmonics=4)
    if popt is None:
        return False, 999.0, None

    # Gate on MEA range
    if not (mea_range[0] <= mea <= mea_range[1]):
        return False, mea, popt

    # --- B. Density-in-dip check ---
    std_resid = float(np.std(residuals))
    y_pred = fourier_series(phase, *popt)

    # Define the "dip": model values above mean + 1σ (deep eclipse region)
    threshold = float(np.mean(y_pred) + np.std(y_pred))
    in_dip = y_pred > threshold
    if not np.any(in_dip):
        return False, mea, popt

    dip_width = float(np.sum(in_dip)) / len(y_pred)
    if dip_width == 0.0:
        return False, mea, popt

    # Well-fitted = within ±1σ of the Fourier residuals
    well_fitted = np.abs(residuals) < std_resid
    in_dip_and_well = in_dip & well_fitted

    density_dip = float(np.sum(in_dip_and_well)) / dip_width
    density_global = float(np.sum(well_fitted))

    if density_dip <= density_global + std_resid:
        print(
            f"[score_eb] Possible EB: {ts.timeseries_id} | "
            f"ρ_dip={density_dip:.2f}  ρ_global={density_global:.2f}"
        )
        return True, mea, popt

    return False, mea, popt


# ---------------------------------------------------------------------------
# Morphology classification
# ---------------------------------------------------------------------------


def classify_eb_type(popt: np.ndarray) -> str:
    """
    Classify an EB's morphology from a fitted Double Super-Gaussian.

    The shape parameter of the deeper (primary) dip determines the class:

    * ``shape > 2.5``  →  ``'detached'``   (flat-bottomed, U-shaped eclipse)
    * ``shape < 1.5``  →  ``'contact'``    (pointed, V-shaped eclipse)
    * otherwise        →  ``'semi-detached'``

    Parameters
    ----------
    popt : np.ndarray
        Best-fit parameters from ``fit_double_super_gaussian()``.
        Format: [baseline, amp1, cent1, wid1, **shape1**, amp2, cent2, wid2, shape2].

    Returns
    -------
    str
        One of ``'detached'``, ``'semi-detached'``, ``'contact'``.
    """
    if popt is None or len(popt) < 5:
        return "unknown"
    # Primary eclipse shape is parameter index 4
    shape = float(popt[4])
    if shape > 2.5:
        return "detached"
    if shape < 1.5:
        return "contact"
    return "semi-detached"


def detect_secondary_eclipse(
    phase: np.ndarray,
    mag: np.ndarray,
    primary_center: float = 0.0,
    search_window: float = 0.15,
) -> dict:
    """
    Search for a secondary eclipse dip near phase 0.5.

    Parameters
    ----------
    phase : np.ndarray
        Phase values in [0, 1].
    mag : np.ndarray
        Magnitude values.
    primary_center : float
        Phase of the primary eclipse (used to anchor the 0.5 search).
    search_window : float
        Half-width of the search window around phase 0.5.

    Returns
    -------
    dict with keys:
        ``found``    — True if a secondary dip was detected.
        ``center``   — Phase of the secondary dip (None if not found).
        ``depth``    — Magnitude depth of the secondary dip.
    """
    secondary_center = (primary_center + 0.5) % 1.0
    lo = (secondary_center - search_window) % 1.0
    hi = (secondary_center + search_window) % 1.0

    if lo < hi:
        mask = (phase >= lo) & (phase <= hi)
    else:
        # Wraps around phase 0/1 boundary
        mask = (phase >= lo) | (phase <= hi)

    if not np.any(mask):
        return {"found": False, "center": None, "depth": 0.0}

    sub_mag = mag[mask]
    sub_phase = phase[mask]
    dip_idx = np.argmax(sub_mag)  # Faintest point (highest mag value)
    dip_depth = float(sub_mag[dip_idx] - np.median(mag))

    # Only flag as a secondary if the dip is more than 0.5σ above the median
    threshold = 0.5 * float(np.std(mag))
    found = dip_depth > threshold

    return {
        "found": found,
        "center": float(sub_phase[dip_idx]) if found else None,
        "depth": dip_depth if found else 0.0,
    }
