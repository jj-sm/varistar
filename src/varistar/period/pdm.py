"""
varistar.period.pdm
===================
Phase Dispersion Minimization (PDM) period-finding functions.

Two variants are provided:

``compute_pdm``   — Stellingwerf (1978) binned PDM.  Fast and well-understood;
                    best for regular, moderate-cadence surveys.
``compute_pdm2``  — Binless PDM (Plavchan et al. 2008 variant).  Uses piecewise-
                    linear interpolation between bin means rather than discrete
                    bins, giving smoother θ curves for sparse data.

Both functions are pure numpy operations with no class dependencies.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# PDM (Stellingwerf 1978)
# ---------------------------------------------------------------------------

def compute_pdm(
    t: np.ndarray,
    y: np.ndarray,
    min_freq: float = 0.001,
    max_freq: float = 10.0,
    n_freq: int = 20_000,
    n_bins: int = 10,
    n_top: int = 100,
) -> dict:
    """
    Compute a PDM periodogram using the Stellingwerf (1978) θ statistic.

    Lower θ → better phase coherence → better period candidate.

    Parameters
    ----------
    t, y : np.ndarray
        Time (HJD) and magnitude arrays.
    min_freq, max_freq : float
        Frequency search bounds in cycles/day.
    n_freq : int
        Number of trial frequencies on the linear grid.
    n_bins : int
        Number of phase bins.
    n_top : int
        Number of best candidate periods to return.

    Returns
    -------
    dict with keys:
        ``frequencies``  — np.ndarray of trial frequencies
        ``theta``        — θ statistic at each frequency (lower = better)
        ``periods``      — list of top-*n_top* periods sorted by ascending θ
        ``best_period``  — period with the lowest θ
    """
    frequencies = np.linspace(min_freq, max_freq, n_freq)
    sigma_total = np.var(y)
    N = len(y)
    denom = N - n_bins  # denominator for s²

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    theta = np.ones(n_freq)  # Default: worst score

    for k, freq in enumerate(frequencies):
        period = 1.0 / freq
        phase = (t / period) % 1.0
        indices = np.digitize(phase, bin_edges) - 1  # 0-indexed

        total_bin_var = 0.0
        for i in range(n_bins):
            in_bin = indices == i
            n_pts = int(np.sum(in_bin))
            if n_pts > 1:
                total_bin_var += np.var(y[in_bin]) * (n_pts - 1)

        if total_bin_var > 0.0 and denom > 0:
            s_sq = total_bin_var / denom
            theta[k] = s_sq / (sigma_total + 1e-12)

    sorted_idx = np.argsort(theta)
    top_periods = (1.0 / frequencies[sorted_idx[:n_top]]).tolist()

    return {
        "frequencies": frequencies,
        "theta": theta,
        "periods": top_periods,
        "best_period": top_periods[0],
    }


# ---------------------------------------------------------------------------
# PDM2 / Binless PDM (Plavchan et al. 2008 variant)
# ---------------------------------------------------------------------------

def compute_pdm2(
    t: np.ndarray,
    y: np.ndarray,
    min_freq: float = 0.001,
    max_freq: float = 10.0,
    samples_per_peak: int = 10,
    phase_bins: int = 50,
    n_top: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Compute a binless PDM2 periodogram.

    Instead of assigning observations to hard bins, this method builds a
    piecewise-linear template from phase-bin means and measures the variance
    of the residuals.  This gives smoother θ curves and handles sparse
    data better than the original binned PDM.

    Parameters
    ----------
    t, y : np.ndarray
        Time and magnitude arrays.
    samples_per_peak : int
        Frequency grid step = 1 / (baseline · samples_per_peak).
    phase_bins : int
        Number of phase bins for the piecewise template.
    n_top : int
        Number of best candidate periods to return.

    Returns
    -------
    dict with keys:
        ``frequencies``  — np.ndarray of trial frequencies
        ``theta``        — θ statistic at each frequency
        ``periods``      — top-*n_top* periods sorted by ascending θ
        ``best_period``  — period with the lowest θ
    """
    y_centered = y - np.median(y)
    total_variance = np.sum(y_centered ** 2) + 1e-12

    freq_step = 1.0 / (np.ptp(t) * samples_per_peak)
    frequencies = np.arange(min_freq, max_freq, freq_step)
    theta = np.zeros(len(frequencies))

    bin_edges = np.linspace(0.0, 1.0, phase_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    x_bins = np.arange(phase_bins) + 0.5  # For interpolation of empty bins

    if verbose:
        print(f"PDM2: {len(frequencies)} trial frequencies...")

    for k, freq in enumerate(frequencies):
        period = 1.0 / freq
        phi = (t / period) % 1.0

        sort_idx = np.argsort(phi)
        phi_s = phi[sort_idx]
        y_s = y_centered[sort_idx]

        bin_idx = np.digitize(phi_s, bin_edges) - 1
        bin_means = np.zeros(phase_bins)
        bin_counts = np.zeros(phase_bins)

        for b in range(phase_bins):
            mask = bin_idx == b
            if np.any(mask):
                bin_means[b] = np.mean(y_s[mask])
                bin_counts[b] = np.sum(mask)

        # Fill empty bins by linear interpolation from populated neighbours
        empty = bin_counts == 0
        if np.any(empty) and np.sum(~empty) > 1:
            bin_means[empty] = np.interp(
                x_bins[empty], x_bins[~empty], bin_means[~empty],
                period=phase_bins,
            )

        # Piecewise-linear model: interpolate template at each observation's phase
        model_y = np.interp(phi_s, bin_centers, bin_means, period=1.0)
        residuals = y_s - model_y
        theta[k] = np.sum(residuals ** 2) / total_variance

    sorted_idx = np.argsort(theta)
    top_periods = (1.0 / frequencies[sorted_idx[:n_top]]).tolist()

    return {
        "frequencies": frequencies,
        "theta": theta,
        "periods": top_periods,
        "best_period": top_periods[0],
    }