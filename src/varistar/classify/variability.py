"""
varistar.classify.variability
==============================
Scalar variability indices computed from a raw magnitude array.

These indices are survey-independent, require no period information, and
are the standard first-pass features used to separate variable stars from
constant sources in large photometric surveys.

All functions are pure numpy and accept a 1-D magnitude array directly,
so they can be called from ``varistar.ml.features`` or standalone.

References
----------
Stetson (1996) — PASP 108, 851  (J and K indices)
Von Neumann (1941) — Ann. Math. Stat. 12, 367  (η index)
Welch & Stetson (1993) — AJ 105, 1813  (IQR index, beyond1std)
Kim et al. (2011) — A&A 529, A28  (combined index suite for OGLE)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Individual index functions
# ---------------------------------------------------------------------------


def stetson_j(mag: np.ndarray, err: np.ndarray | None = None) -> float:
    """
    Stetson J index.

    Measures correlated brightness changes between *consecutive* observation
    pairs.  Robust variability indicator for well-sampled light curves.

    J = Σ sgn(δᵢ · δᵢ₊₁) · √|δᵢ · δᵢ₊₁| / (N - 1)

    where δᵢ = (mᵢ - <m>) / σ_total.

    Parameters
    ----------
    mag : np.ndarray
        Magnitude array (time-ordered).
    err : np.ndarray | None
        Per-observation errors.  If provided, σ_total = quadrature mean of
        errors; otherwise the sample std is used.

    Returns
    -------
    float
        Stetson J value.  Constant stars cluster near 0; variables > 0.5.
    """
    n = len(mag)
    if n < 2:
        return 0.0
    sigma = float(np.sqrt(np.mean(err**2))) if err is not None else float(np.std(mag))
    sigma = max(sigma, 1e-9)
    delta = (mag - np.mean(mag)) / sigma
    pairs = delta[:-1] * delta[1:]
    return float(np.sum(np.sign(pairs) * np.sqrt(np.abs(pairs))) / (n - 1))


def stetson_k(mag: np.ndarray, err: np.ndarray | None = None) -> float:
    """
    Stetson K index.

    Kurtosis-like measure of the residual distribution shape.  A Gaussian
    noise distribution gives K ≈ 0.798; variable stars tend to have K > 0.9.

    K = (1/N · Σ |δᵢ|) / √(1/N · Σ δᵢ²)

    Parameters
    ----------
    mag, err : np.ndarray
        Magnitudes and optional per-observation errors.

    Returns
    -------
    float
    """
    n = len(mag)
    if n < 2:
        return 0.0
    sigma = float(np.sqrt(np.mean(err**2))) if err is not None else float(np.std(mag))
    sigma = max(sigma, 1e-9)
    delta = (mag - np.mean(mag)) / sigma
    mean_abs = float(np.mean(np.abs(delta)))
    mean_sq = float(np.mean(delta**2))
    return mean_abs / (np.sqrt(mean_sq) + 1e-9)


def eta_index(mag: np.ndarray) -> float:
    """
    Von Neumann η index.

    η = Σ(mᵢ - mᵢ₋₁)² / Σ(mᵢ - <m>)²

    Low η values indicate smooth, correlated variability (e.g. Cepheids,
    Miras).  Random noise gives η ≈ 2.

    Parameters
    ----------
    mag : np.ndarray
        Magnitude array (time-ordered).

    Returns
    -------
    float
    """
    if len(mag) < 2:
        return 2.0
    num = float(np.sum(np.diff(mag) ** 2))
    denom = float(np.sum((mag - np.mean(mag)) ** 2)) + 1e-12
    return num / denom


def iqr_index(mag: np.ndarray) -> float:
    """
    Interquartile Range (IQR) of the magnitude distribution.

    A simple, outlier-resistant amplitude proxy.

    Parameters
    ----------
    mag : np.ndarray

    Returns
    -------
    float
        Q75 - Q25.
    """
    q75, q25 = float(np.percentile(mag, 75)), float(np.percentile(mag, 25))
    return q75 - q25


def amplitude(
    mag: np.ndarray, percentile_range: tuple[float, float] = (5.0, 95.0)
) -> float:
    """
    Robust amplitude estimate: P95 − P5 of the magnitude distribution.

    Uses percentiles rather than min/max to suppress outlier contamination.

    Parameters
    ----------
    mag : np.ndarray
    percentile_range : tuple[float, float]
        Lower and upper percentiles.  Default (5, 95) is standard in the
        OGLE classification literature.

    Returns
    -------
    float
    """
    lo, hi = (
        float(np.percentile(mag, percentile_range[0])),
        float(np.percentile(mag, percentile_range[1])),
    )
    return hi - lo


def excess_variance(mag: np.ndarray, err: np.ndarray) -> float:
    """
    Normalised Excess Variance (NEV).

    NEV = (σ² - <σ_err²>) / <m>²

    Positive values indicate genuine intrinsic variability beyond what is
    expected from photon noise.

    Parameters
    ----------
    mag, err : np.ndarray

    Returns
    -------
    float
    """
    n = len(mag)
    if n < 2:
        return 0.0
    mean_mag = float(np.mean(mag))
    var_obs = float(np.var(mag, ddof=1))
    mean_err2 = float(np.mean(err**2))
    return (var_obs - mean_err2) / (mean_mag**2 + 1e-12)


def welch_stetson_i(
    mag1: np.ndarray, mag2: np.ndarray, err1: np.ndarray, err2: np.ndarray
) -> float:
    """
    Welch-Stetson I index for *two-band* simultaneous observations.

    Detects correlated variations between two photometric bands.
    Requires paired observations (same epoch in both bands).

    I = Σ δ₁ᵢ · δ₂ᵢ / (N·(N-1))

    Parameters
    ----------
    mag1, mag2 : np.ndarray
        Magnitudes in band 1 and band 2.
    err1, err2 : np.ndarray
        Corresponding photometric errors.

    Returns
    -------
    float
    """
    n = len(mag1)
    if n < 2 or len(mag2) != n:
        return 0.0

    def _delta(m, e):
        sigma = max(float(np.sqrt(np.mean(e**2))), 1e-9)
        return (m - np.mean(m)) / sigma

    d1 = _delta(mag1, err1)
    d2 = _delta(mag2, err2)
    return float(np.sum(d1 * d2) / (n * (n - 1)))


# ---------------------------------------------------------------------------
# Composite: compute all indices at once
# ---------------------------------------------------------------------------


def compute_all_indices(ts) -> dict:
    """
    Compute all variability indices for a ``TimeSeries`` object.

    Parameters
    ----------
    ts : TimeSeries
        Must have a non-empty ``timeseries_df`` with ``mag_col`` and
        ``err_col`` attributes.

    Returns
    -------
    dict
        Keys: ``timeseries_id``, ``stetson_j``, ``stetson_k``, ``eta``,
        ``iqr``, ``amplitude``, ``excess_variance``.
        Returns a dict of zeros if the DataFrame is empty.
    """
    empty = {
        "timeseries_id": getattr(ts, "timeseries_id", "unknown"),
        "stetson_j": 0.0,
        "stetson_k": 0.0,
        "eta": 2.0,
        "iqr": 0.0,
        "amplitude": 0.0,
        "excess_variance": 0.0,
    }

    if ts.timeseries_df.is_empty():
        return empty

    mag = ts.timeseries_df[ts.mag_col].to_numpy()
    err = ts.timeseries_df[ts.err_col].to_numpy()

    return {
        "timeseries_id": ts.timeseries_id,
        "stetson_j": stetson_j(mag, err),
        "stetson_k": stetson_k(mag, err),
        "eta": eta_index(mag),
        "iqr": iqr_index(mag),
        "amplitude": amplitude(mag),
        "excess_variance": excess_variance(mag, err),
    }
