"""
varistar.period.lomb_scargle
============================
Lomb-Scargle periodogram and Spectrum Resampling period-finding functions.

Both functions are pure numpy/astropy operations: they accept raw arrays and
return raw arrays, with no dependency on TimeSeries or LightCurve objects.
"""

from __future__ import annotations

import numpy as np
from astropy.timeseries import LombScargle
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Lomb-Scargle
# ---------------------------------------------------------------------------

def compute_ls(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray | None = None,
    min_freq: float = 0.001,
    max_freq: float = 10.0,
    samples_per_peak: int = 10,
    n_top: int = 20,
) -> dict:
    """
    Compute a Lomb-Scargle power spectrum and return the top candidate periods.

    Invalid (NaN, non-positive error) observations are removed automatically.
    If *all* errors are invalid, uniform weighting is used as a fallback.

    Parameters
    ----------
    t, y, dy : np.ndarray
        Time (HJD), magnitude, and uncertainty arrays.
    min_freq, max_freq : float
        Frequency search bounds in cycles/day.
    samples_per_peak : int
        Oversampling factor for the frequency grid.
    n_top : int
        Number of top-power periods to return.

    Returns
    -------
    dict with keys:
        ``frequency``  — np.ndarray of sampled frequencies
        ``period``     — 1 / frequency
        ``power``      — Lomb-Scargle power at each frequency
        ``periods``    — list of top-*n_top* periods sorted by descending power
        ``periods_map``— {``'p1'``: strongest, ``'p2'``: second, …}
    """
    # --- Clean invalid values ---
    if dy is not None:
        valid = (dy > 1e-6) & (~np.isnan(dy)) & (~np.isnan(y))
        if not np.any(valid):
            dy = None  # Fall back to uniform weights
        else:
            t, y, dy = t[valid], y[valid], dy[valid]

    ls = LombScargle(t, y, dy, normalization="standard")
    frequency, power = ls.autopower(
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        samples_per_peak=samples_per_peak,
    )

    # Sort by power (strongest first) — NOT by period length
    top_idx = np.argsort(power)[::-1]
    top_periods = (1.0 / frequency[top_idx])[:n_top].tolist()
    periods_map = {f"p{i + 1}": p for i, p in enumerate(top_periods)}

    return {
        "frequency": frequency,
        "period": 1.0 / frequency,
        "power": power,
        "periods": top_periods,
        "periods_map": periods_map,
    }


def false_alarm_levels(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray | None = None,
    fap_levels: tuple[float, ...] = (0.1, 0.01, 0.001),
    min_freq: float = 0.001,
    max_freq: float = 10.0,
) -> dict[float, float]:
    """
    Return the Lomb-Scargle power thresholds at given False Alarm Probability
    levels using astropy's bootstrap FAP method.

    Parameters
    ----------
    fap_levels : tuple[float, ...]
        FAP levels to evaluate (e.g. 0.1 = 10 %, 0.01 = 1 %).

    Returns
    -------
    dict mapping each FAP level to its corresponding power threshold.
    """
    if dy is not None:
        valid = (dy > 1e-6) & (~np.isnan(dy)) & (~np.isnan(y))
        if np.any(valid):
            t, y, dy = t[valid], y[valid], dy[valid]
        else:
            dy = None

    ls = LombScargle(t, y, dy, normalization="standard")
    return {
        level: float(ls.false_alarm_level(level, method="bootstrap",
                                           minimum_frequency=min_freq,
                                           maximum_frequency=max_freq))
        for level in fap_levels
    }


# ---------------------------------------------------------------------------
# Spectrum Resampling
# ---------------------------------------------------------------------------

def compute_sr(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray | None = None,
    min_freq: float = 0.001,
    max_freq: float = 10.0,
    samples_per_peak: int = 10,
    smoothing_sigma: float = 2.0,
    n_bootstrap: int = 1000,
    verbose: bool = True,
) -> dict:
    """
    Estimate the dominant period using Spectrum Resampling (SR).

    SR bootstraps the residuals of a smoothed power spectrum to quantify
    period uncertainty.  It is more robust than a single peak-pick when the
    spectrum has multiple comparable peaks.

    Parameters
    ----------
    t, y, dy : np.ndarray
        Time, magnitude, and uncertainty arrays.
    smoothing_sigma : float
        Gaussian kernel σ (in frequency-grid steps) for spectrum smoothing.
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    dict with keys:
        ``best_period``  — 1 / mean bootstrap frequency
        ``uncertainty``  — asymmetric 1-σ uncertainty on the period
        ``mean_freq``    — mean of the bootstrap frequency distribution
        ``std_freq``     — std of the bootstrap frequency distribution
        ``all_freqs``    — list of all bootstrap peak frequencies
    """
    # --- Clean ---
    if dy is not None:
        valid = (dy > 1e-6) & (~np.isnan(dy)) & (~np.isnan(y))
        dy = dy[valid] if np.any(valid) else None
        if dy is not None:
            t, y = t[valid], y[valid]

    ls = LombScargle(t, y, dy, normalization="standard")
    freq_grid, power_raw = ls.autopower(
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        samples_per_peak=samples_per_peak,
    )

    # Smooth once to get the "base spectrum"
    power_smooth = gaussian_filter1d(power_raw, sigma=smoothing_sigma)
    noise = power_raw - power_smooth

    if verbose:
        print(f"Spectrum Resampling: running {n_bootstrap} bootstraps...")

    bootstrap_freqs: list[float] = []
    for _ in range(n_bootstrap):
        shuffled = np.random.choice(noise, size=len(noise), replace=True)
        sample = gaussian_filter1d(power_smooth + shuffled, sigma=smoothing_sigma)
        bootstrap_freqs.append(float(freq_grid[np.argmax(sample)]))

    mean_freq = float(np.mean(bootstrap_freqs))
    std_freq = float(np.std(bootstrap_freqs))
    best_period = 1.0 / mean_freq
    # Asymmetric uncertainty: how much the period shifts at ±1σ of frequency
    uncertainty = float(abs(1.0 / (mean_freq - std_freq) - best_period))

    if verbose:
        print(f"SR result: P = {best_period:.5f} ± {uncertainty:.5f} d")

    return {
        "best_period": best_period,
        "uncertainty": uncertainty,
        "mean_freq": mean_freq,
        "std_freq": std_freq,
        "all_freqs": bootstrap_freqs,
    }