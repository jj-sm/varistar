"""
varistar.period.vs_period
=========================
High-level period selection utilities for variable star light curves.

These functions are designed to be called by ``LightCurve`` methods but are
also available as standalone functions, making them unit-testable and reusable
from outside the class hierarchy.

Key responsibilities
--------------------
* Detect harmonic relationships among candidate periods.
* Assess phase-coverage quality for a given period and dataset.
* Pick the best physically-meaningful period from a ranked candidate list,
  accounting for harmonics and phase-coverage gaps.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Harmonic detection
# ---------------------------------------------------------------------------

def check_harmonics(
    periods: list[float],
    tolerance: float = 0.02,
) -> dict:
    """
    Determine whether candidate periods are harmonics of the dominant period.

    Two relationships are checked for each candidate *P*:
    * **Integer harmonic**: P_dom / P ≈ integer n > 1  →  P = P_dom / n
    * **Fraction harmonic**: P / P_dom ≈ integer n > 1 →  P = n · P_dom

    Parameters
    ----------
    periods : list[float]
        Candidate periods sorted by descending significance (strongest first).
    tolerance : float
        Fractional tolerance for ratio matching (e.g. 0.02 = 2 %).

    Returns
    -------
    dict with keys:
        ``is_harmonic``      — True if any harmonic relationships were found.
        ``dominant_period``  — periods[0], the strongest candidate.
        ``harmonic_periods`` — list of periods identified as harmonics.
        ``relationships``    — matching ratio strings, e.g. ``'1/2'`` or ``'2/1'``.
    """
    if not periods:
        return {"is_harmonic": False, "msg": "No periods provided."}
    if len(periods) < 2:
        return {
            "is_harmonic": False,
            "dominant_period": periods[0],
            "harmonic_periods": [],
            "relationships": [],
        }

    p_dom = periods[0]
    # Deduplicate candidates (round to 3 dp to merge near-duplicates)
    unique = sorted(set(round(p, 3) for p in periods), reverse=True)

    harmonics: list[float] = []
    ratios: list[str] = []

    for p_cand in unique:
        if abs(p_cand - p_dom) < 1e-9:
            continue  # Skip the dominant itself

        ratio = p_dom / p_cand
        inv_ratio = p_cand / p_dom
        n_int = round(ratio)
        n_inv = round(inv_ratio)

        if n_int > 1 and abs(ratio - n_int) < tolerance:
            harmonics.append(p_cand)
            ratios.append(f"1/{n_int}")
        elif n_inv > 1 and abs(inv_ratio - n_inv) < tolerance:
            harmonics.append(p_cand)
            ratios.append(f"{n_inv}/1")

    return {
        "is_harmonic": len(harmonics) > 0,
        "dominant_period": p_dom,
        "harmonic_periods": harmonics,
        "relationships": ratios,
    }


# ---------------------------------------------------------------------------
# Phase coverage
# ---------------------------------------------------------------------------

def get_phase_coverage(
    t: np.ndarray,
    period: float,
    max_gap_threshold: float = 0.10,
) -> dict:
    """
    Measure the largest gap in phase coverage for a given trial period.

    A large gap (> *max_gap_threshold*) indicates that the folded light curve
    has a significant phase region without any observations — the period may
    be aliased or the data may be too sparse for reliable folding.

    Parameters
    ----------
    t : np.ndarray
        Observation times (HJD).
    period : float
        Trial period in days.
    max_gap_threshold : float
        Maximum acceptable gap in phase units [0, 1].

    Returns
    -------
    dict with keys:
        ``max_gap``    — size of the largest gap in phase (0–1).
        ``is_gap_safe``— True if max_gap < max_gap_threshold.
    """
    if period <= 0.0:
        return {"max_gap": 1.0, "is_gap_safe": False}

    phase = np.sort((t / period) % 1.0)
    gaps = np.diff(phase)
    wrap_gap = 1.0 - phase[-1] + phase[0]
    max_gap = float(np.max(np.append(gaps, wrap_gap)))

    return {
        "max_gap": max_gap,
        "is_gap_safe": max_gap < max_gap_threshold,
    }


# ---------------------------------------------------------------------------
# Best-period selection
# ---------------------------------------------------------------------------

def select_best_period(
    t: np.ndarray,
    periods: list[float],
    harmonic_tolerance: float = 0.02,
    gap_threshold: float = 0.10,
) -> dict:
    """
    Select the best physically-meaningful period from a ranked candidate list.

    Decision logic
    --------------
    1. Run harmonic detection on all candidates.
    2. If no harmonics are found:
       * Check phase coverage for the dominant period.
       * If coverage is poor, walk down the candidate list until a well-covered
         period is found.
    3. If harmonics are found:
       * Pool the dominant + harmonic periods and pick the longest one that
         still has acceptable phase coverage.
    4. If no candidate passes the coverage check, flag the result as
       ``not_periodic`` and fall back to the raw dominant period.

    Parameters
    ----------
    t : np.ndarray
        Observation times.
    periods : list[float]
        Candidate periods sorted by descending significance.
    harmonic_tolerance : float
        Tolerance passed to ``check_harmonics``.
    gap_threshold : float
        Acceptable phase-gap threshold passed to ``get_phase_coverage``.

    Returns
    -------
    dict with keys:
        ``best_period``  — selected period.
        ``is_harmonic``  — whether harmonic relationships were detected.
        ``not_periodic`` — True if no candidate had acceptable phase coverage.
        ``not_dominant`` — True if the dominant period was replaced due to gaps.
    """
    if not periods:
        return {
            "best_period": None,
            "is_harmonic": False,
            "not_periodic": True,
            "not_dominant": True,
        }

    harmonic_info = check_harmonics(periods, tolerance=harmonic_tolerance)
    p_dominant = harmonic_info["dominant_period"]
    is_harmonic = harmonic_info["is_harmonic"]

    not_periodic = False
    not_dominant = False

    if not is_harmonic:
        # Walk the full candidate list for acceptable coverage
        if get_phase_coverage(t, p_dominant, gap_threshold)["is_gap_safe"]:
            best_period = p_dominant
        else:
            unique = sorted(set(round(p, 4) for p in periods), reverse=True)
            best_period = None
            for p in unique:
                if get_phase_coverage(t, p, gap_threshold)["is_gap_safe"]:
                    best_period = p
                    not_dominant = True
                    break
            if best_period is None:
                best_period = p_dominant
                not_periodic = True
                not_dominant = True

    else:
        # Try dominant + harmonics from longest to shortest
        pool = sorted(
            [p_dominant] + harmonic_info["harmonic_periods"],
            reverse=True,
        )
        best_period = None
        for p in pool:
            if get_phase_coverage(t, p, gap_threshold)["is_gap_safe"]:
                best_period = p
                break
        if best_period is None:
            best_period = p_dominant
            not_periodic = True

    return {
        "best_period": best_period,
        "is_harmonic": is_harmonic,
        "not_periodic": not_periodic,
        "not_dominant": not_dominant,
    }