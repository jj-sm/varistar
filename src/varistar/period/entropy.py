"""
varistar.period.entropy
=======================
Information-theoretic and Analysis-of-Variance period-finding methods.

Two algorithms are implemented:

``compute_ce``   — Conditional Entropy (Graham et al. 2013 / Cincotta et al. 1999).
                   Minimises the Shannon entropy of the phase-folded magnitude
                   distribution.  Effective for sparse, non-sinusoidal light curves
                   where Lomb-Scargle power is diluted.

``compute_aov``  — Analysis of Variance (Schwarzenberg-Czerny 1989).
                   A variance-ratio statistic similar to PDM but uses the F-ratio
                   between inter-bin and intra-bin variance, giving it better
                   statistical properties and a well-defined significance level.

Both functions are pure numpy and accept raw arrays, consistent with the
rest of the ``varistar.period`` sub-package.

References
----------
Graham et al. (2013), MNRAS 434, 3423.
Cincotta et al. (1999), A&AS 137, 21.
Schwarzenberg-Czerny (1989), MNRAS 241, 153.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Conditional Entropy
# ---------------------------------------------------------------------------

def compute_ce(
    t: np.ndarray,
    y: np.ndarray,
    min_freq: float = 0.001,
    max_freq: float = 10.0,
    samples_per_peak: int = 10,
    n_phase_bins: int = 10,
    n_mag_bins: int = 10,
    n_top: int = 100,
    verbose: bool = False,
) -> dict:
    """
    Compute the Conditional Entropy (CE) periodogram.

    For each trial frequency the data are phase-folded and the joint
    phase-magnitude distribution is binned into a 2-D grid.  The
    Conditional Entropy of magnitude given phase is:

        CE(f) = -Σᵢ Σⱼ p(φᵢ, mⱼ) · log( p(φᵢ, mⱼ) / p(φᵢ) )

    Lower CE → more structured (periodic) light curve at that frequency.

    Parameters
    ----------
    t, y : np.ndarray
        Observation times and magnitudes.
    min_freq, max_freq : float
        Frequency search bounds in cycles/day.
    samples_per_peak : int
        Frequency grid density relative to the Nyquist sampling.
    n_phase_bins, n_mag_bins : int
        Grid resolution for the 2-D histogram.  Increasing these gives
        finer resolution at the cost of more computation.
    n_top : int
        Number of best candidate periods to return.
    verbose : bool
        Print progress updates.

    Returns
    -------
    dict with keys:
        ``frequencies``  — np.ndarray of trial frequencies
        ``ce``           — CE statistic (lower = better)
        ``periods``      — top-*n_top* periods sorted by ascending CE
        ``best_period``  — period with the lowest CE
    """
    T_baseline = float(np.ptp(t))
    if T_baseline == 0.0:
        raise ValueError("All observations have the same time stamp.")

    freq_step  = 1.0 / (T_baseline * samples_per_peak)
    frequencies = np.arange(min_freq, max_freq, freq_step)
    n_freqs     = len(frequencies)
    ce_stats    = np.zeros(n_freqs)

    if verbose:
        print(f"CE: {n_freqs} trial frequencies, "
              f"{n_phase_bins}×{n_mag_bins} grid...")

    # Normalise magnitudes to [0, 1] once (outside the loop)
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min + 1e-12)

    phase_edges = np.linspace(0.0, 1.0, n_phase_bins + 1)
    mag_edges   = np.linspace(0.0, 1.0, n_mag_bins   + 1)

    for k, freq in enumerate(frequencies):
        period = 1.0 / freq
        phase  = (t / period) % 1.0

        # 2-D histogram: p(phase_bin, mag_bin)
        counts, _, _ = np.histogram2d(phase, y_norm,
                                       bins=[phase_edges, mag_edges])
        p_joint = counts / (counts.sum() + 1e-12)

        # Marginal over magnitude bins → p(phase)
        p_phase = p_joint.sum(axis=1, keepdims=True)  # shape (n_phase, 1)

        # Conditional probability p(mag | phase) = p(phase, mag) / p(phase)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_cond = np.where(
                p_joint > 0,
                np.log(p_joint / (p_phase + 1e-12)),
                0.0,
            )

        ce_stats[k] = -float(np.sum(p_joint * log_cond))

    sorted_idx = np.argsort(ce_stats)
    top_periods = (1.0 / frequencies[sorted_idx[:n_top]]).tolist()

    return {
        "frequencies": frequencies,
        "ce":          ce_stats,
        "periods":     top_periods,
        "best_period": top_periods[0],
    }


# ---------------------------------------------------------------------------
# Analysis of Variance (AOV)
# ---------------------------------------------------------------------------

def compute_aov(
    t: np.ndarray,
    y: np.ndarray,
    min_freq: float = 0.001,
    max_freq: float = 10.0,
    samples_per_peak: int = 10,
    n_bins: int = 10,
    n_top: int = 100,
    verbose: bool = False,
) -> dict:
    """
    Compute an Analysis of Variance (AOV) periodogram.

    The AOV statistic is an F-ratio:

        AOV(f) = (N - n_bins) / (n_bins - 1)
                 · Σⱼ nⱼ (ȳⱼ - ȳ)² / Σᵢ (yᵢ - ȳⱼ)²

    where nⱼ and ȳⱼ are the count and mean of the j-th phase bin.
    Higher AOV → better phase coherence → better period candidate.

    This implementation inverts the statistic for consistent storage
    (lower = better, matching CE and PDM conventions), stored as ``1/aov``.

    Parameters
    ----------
    t, y : np.ndarray
        Observation times and magnitudes.
    min_freq, max_freq : float
        Frequency search bounds.
    samples_per_peak : int
        Frequency grid density.
    n_bins : int
        Number of phase bins.
    n_top : int
        Number of best candidate periods to return.

    Returns
    -------
    dict with keys:
        ``frequencies``  — np.ndarray of trial frequencies
        ``aov``          — raw AOV F-ratio at each frequency (higher = better)
        ``periods``      — top-*n_top* periods sorted by descending AOV
        ``best_period``  — period with the highest AOV
    """
    T_baseline = float(np.ptp(t))
    if T_baseline == 0.0:
        raise ValueError("All observations have the same time stamp.")

    freq_step   = 1.0 / (T_baseline * samples_per_peak)
    frequencies = np.arange(min_freq, max_freq, freq_step)
    n_freqs     = len(frequencies)
    aov_stats   = np.zeros(n_freqs)

    N          = len(y)
    y_mean     = float(np.mean(y))
    ss_total   = float(np.sum((y - y_mean) ** 2))

    if verbose:
        print(f"AOV: {n_freqs} trial frequencies, {n_bins} bins...")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for k, freq in enumerate(frequencies):
        period = 1.0 / freq
        phase  = (t / period) % 1.0
        idx    = np.digitize(phase, bin_edges) - 1

        ss_between = 0.0
        ss_within  = 0.0

        for b in range(n_bins):
            mask = idx == b
            nb   = int(np.sum(mask))
            if nb < 1:
                continue
            yb          = y[mask]
            bin_mean    = float(np.mean(yb))
            ss_between += nb * (bin_mean - y_mean) ** 2
            ss_within  += float(np.sum((yb - bin_mean) ** 2))

        # Avoid division by zero for degenerate cases
        denom = ss_within * (n_bins - 1)
        if denom > 0.0 and (N - n_bins) > 0:
            aov_stats[k] = ((N - n_bins) * ss_between) / denom
        else:
            aov_stats[k] = 0.0

    # Highest AOV = best period
    sorted_idx  = np.argsort(aov_stats)[::-1]
    top_periods = (1.0 / frequencies[sorted_idx[:n_top]]).tolist()

    return {
        "frequencies": frequencies,
        "aov":         aov_stats,
        "periods":     top_periods,
        "best_period": top_periods[0],
    }