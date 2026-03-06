"""
varistar.ml.features
====================
Statistical and astrophysical feature extraction for variable star light curves.

Feature index reference
-----------------------
 0  median          — Median magnitude (Q0.5)
 1  mad             — Median Absolute Deviation
 2  octile_skew     — Octile Skewness (OS)
 3  low             — Left Octile ratio (LOW)
 4  row             — Right Octile ratio (ROW)
 5  mav             — Modified Abbe Value (Huber-loss variant)
 6  skew            — Standard skewness
 7  stetson_j       — Stetson J variability index
 8  flux_perc_ratio — Flux Percentile Ratio (F5–95 / F40–60)
 9  log_freq        — log10 of the dominant frequency
10  log_amp         — log10 of the peak-to-peak amplitude
11  r21             — Fourier amplitude ratio R21
12  ph21_cos        — cos(φ21) Fourier phase difference
13  ph21_sin        — sin(φ21) Fourier phase difference
14  freq_ratio      — Ratio of second-to-first frequency
15  von_neumann     — Von Neumann η index
16  kurtosis        — Excess kurtosis
17  beyond1std      — Fraction of points beyond 1 standard deviation
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------


def f0_median(y: np.ndarray) -> float:
    """Median magnitude."""
    return float(np.median(y))


def f1_mad(y: np.ndarray) -> float:
    """Median Absolute Deviation."""
    return float(np.median(np.abs(y - np.median(y))))


def f2_octile_skewness(y: np.ndarray) -> float:
    """
    Octile Skewness (OS).
    Robust skewness estimate using the 12.5th, 50th, and 87.5th percentiles.
    """
    q = np.percentile(y, [12.5, 50.0, 87.5])
    return float((q[2] + q[0] - 2.0 * q[1]) / (q[2] - q[0] + 1e-6))


def f3_low(y: np.ndarray) -> float:
    """
    Left Octile ratio (LOW).
    Measures asymmetry on the faint side of the distribution.
    """
    q = np.percentile(y, [12.5, 25.0, 50.0])
    return float((q[1] - q[0]) / (q[2] - q[0] + 1e-6))


def f4_row(y: np.ndarray) -> float:
    """
    Right Octile ratio (ROW).
    Measures asymmetry on the bright side of the distribution.
    """
    q = np.percentile(y, [50.0, 75.0, 87.5])
    return float((q[2] - q[1]) / (q[2] - q[0] + 1e-6))


def f5_mav(y: np.ndarray) -> float:
    """
    Modified Abbe Value (MAV).
    Abbe value computed with Huber-robust loss to suppress outlier influence.
    k=1.345 corresponds to 95 % asymptotic efficiency under Gaussian noise.
    (Pérez et.al, 2017)
    """
    median = np.median(y)
    mad = np.median(np.abs(y - median)) + 1e-6
    diffs = np.diff(y) / (np.sqrt(2.0) * mad)

    k = 1.345
    abs_d = np.abs(diffs)
    loss = np.where(abs_d <= k, 0.5 * diffs**2, k * (abs_d - 0.5 * k))
    return float(np.mean(loss))


def f6_skew(y: np.ndarray) -> float:
    """Standard (Fisher–Pearson) skewness."""
    return float(skew(y))


def f7_stetson_j(y: np.ndarray) -> float:
    """
    Stetson J index.
    Detects correlated variability between consecutive pairs of observations.
    Normalised by (N-1) to be comparable across different dataset sizes.
    """
    std = np.std(y) + 1e-6
    delta = (y - np.mean(y)) / std
    pairs = delta[:-1] * delta[1:]
    return float(np.sum(np.sign(pairs) * np.sqrt(np.abs(pairs))) / (len(y) - 1))


def f8_flux_perc_ratio(y: np.ndarray) -> float:
    """
    Flux Percentile Ratio.
    Ratio of the 5–95th percentile range to the 40–60th percentile range.
    A large value indicates a peaked, outlier-rich distribution.
    """
    q = np.percentile(y, [5.0, 40.0, 60.0, 95.0])
    return float((q[3] - q[0]) / (q[2] - q[1] + 1e-6))


def f9_log_freq(f1: float) -> float:
    """
    log10 of the dominant frequency (cycles per day).
    Returns -5 as a sentinel when no valid frequency is available.
    """
    return float(np.log10(f1)) if f1 > 0.0 else -5.0


def f10_log_amp(y: np.ndarray) -> float:
    """
    log10 of the peak-to-peak magnitude amplitude.
    Returns -5 as a sentinel for flat light curves.
    """
    amp = float(np.max(y) - np.min(y))
    return float(np.log10(amp)) if amp > 0.0 else -5.0


def f11_12_13_fourier(
    t: np.ndarray,
    y: np.ndarray,
    p1: float | None,
) -> tuple[float, float, float]:
    """
    Fourier decomposition features: R21, cos(φ21), sin(φ21).

    Fits the two-harmonic model:
        m(φ) = A0 + A1·sin(2πφ + φ1) + A2·sin(4πφ + φ2)

    Returns
    -------
    r21      : amplitude ratio A2 / A1
    ph21_cos : cos(φ2 - 2·φ1)
    ph21_sin : sin(φ2 - 2·φ1)
    """
    if p1 is None or p1 <= 0.0:
        return 0.0, 0.0, 0.0

    phase = (t / p1) % 1.0
    std = np.std(y)

    def _model(x, off, a1, ph1, a2, ph2):
        return (
            off
            + a1 * np.sin(2.0 * np.pi * x + ph1)
            + a2 * np.sin(4.0 * np.pi * x + ph2)
        )

    try:
        p0 = [np.mean(y), std, 0.0, 0.5 * std, 0.0]
        popt, _ = curve_fit(_model, phase, y, p0=p0, maxfev=500)
        a1, ph1 = popt[1], popt[2]
        a2, ph2 = popt[3], popt[4]
        r21 = abs(a2) / (abs(a1) + 1e-6)
        phi21 = (ph2 - 2.0 * ph1) % (2.0 * np.pi)
        return float(r21), float(np.cos(phi21)), float(np.sin(phi21))
    except Exception:
        return 0.0, 0.0, 0.0


def f14_freq_ratio(periods: list[float], f1: float) -> float:
    """
    Ratio of the second-strongest frequency to the dominant frequency.
    Returns 0 when fewer than two candidate periods are available.
    """
    if f1 <= 0.0 or len(periods) < 2:
        return 0.0
    return float((1.0 / periods[1]) / f1)


def f15_von_neumann(y: np.ndarray) -> float:
    """
    Von Neumann η index.
    Ratio of mean-squared successive differences to the sample variance.
    Low values indicate correlated, smooth variability.
    """
    diff_sq = np.diff(y) ** 2
    var = np.sum((y - np.mean(y)) ** 2) + 1e-6
    return float(np.sum(diff_sq) / var)


def f16_kurtosis(y: np.ndarray) -> float:
    """Excess kurtosis (Fisher definition; normal distribution → 0)."""
    return float(kurtosis(y))


def f17_beyond1std(y: np.ndarray) -> float:
    """
    Fraction of data points that lie more than one standard deviation
    from the mean magnitude.
    """
    std = np.std(y)
    mean = np.mean(y)
    return float(np.sum(np.abs(y - mean) > std) / len(y))


# ---------------------------------------------------------------------------
# Feature registry
# Maps index → (column_name, callable_or_sentinel)
# Callables that need extra arguments (t, p1, lc_periods) are handled
# explicitly inside FeatureExtractor.extract().
# ---------------------------------------------------------------------------

_FEATURE_NAMES: dict[int, str] = {
    0: "median",
    1: "mad",
    2: "octile_skew",
    3: "low",
    4: "row",
    5: "mav",
    6: "skew",
    7: "stetson_j",
    8: "flux_perc_ratio",
    9: "log_freq",
    10: "log_amp",
    11: "r21",
    12: "ph21_cos",
    13: "ph21_sin",
    14: "freq_ratio",
    15: "von_neumann",
    16: "kurtosis",
    17: "beyond1std",
}

ALL_FEATURE_INDICES: list[int] = list(_FEATURE_NAMES.keys())


def feature_name(index: int) -> str:
    """Return the column name for a feature index."""
    return _FEATURE_NAMES[index]


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """
    Stateless extractor that converts a TimeSeries / LightCurve pair into a
    flat feature dictionary suitable for DataFrame construction.

    Usage
    -----
    >>> result = FeatureExtractor.extract(ts_obj=ts, lc_obj=lc)
    >>> result = FeatureExtractor.extract(ts_obj=ts, lc_obj=lc, selected_indices=[0, 1, 7, 15])
    """

    @staticmethod
    def extract(
        ts_obj=None,
        lc_obj=None,
        selected_indices: list[int] | None = None,
    ) -> dict:
        """
        Extract features from a TimeSeries and/or LightCurve object.

        Parameters
        ----------
        ts_obj : TimeSeries, optional
            Preferred source of raw time/magnitude data.
        lc_obj : LightCurve, optional
            Used for period-dependent features (indices 9–14).
            If omitted, all period-dependent features default to 0.
        selected_indices : list[int] | None
            Subset of feature indices (0–17) to compute.
            Defaults to all 18 features.

        Returns
        -------
        dict
            {"id": str, feature_name: value, ...}
        """
        # Data Source
        source = ts_obj if ts_obj is not None else lc_obj
        if source is None:
            raise ValueError("Either ts_obj or lc_obj must be provided.")

        df = source.timeseries_df
        y: np.ndarray = df[source.colnames[1]].to_numpy()
        t: np.ndarray = df["hjd"].to_numpy()
        ts_id: str = getattr(source, "timeseries_id", "unknown")

        # Period
        p1: float | None = None
        f1: float = 0.0
        lc_periods: list[float] = []

        if lc_obj is not None:
            if not lc_obj.periods:
                lc_obj.find_best_period()
            if lc_obj.periods:
                p1 = float(lc_obj.periods[0])
                f1 = 1.0 / p1
                lc_periods = lc_obj.periods

        # Selected Indexes
        indices = (
            ALL_FEATURE_INDICES
            if (selected_indices is None or len(selected_indices) == 0)
            else selected_indices
        )

        # Fourier Calc
        fourier_needed = any(i in indices for i in (11, 12, 13))
        r21, ph21_cos, ph21_sin = (0.0, 0.0, 0.0)
        if fourier_needed:
            r21, ph21_cos, ph21_sin = f11_12_13_fourier(t, y, p1)

        # Feature Map
        feature_map: dict[int, tuple[str, float]] = {
            0: ("median", f0_median(y)),
            1: ("mad", f1_mad(y)),
            2: ("octile_skew", f2_octile_skewness(y)),
            3: ("low", f3_low(y)),
            4: ("row", f4_row(y)),
            5: ("mav", f5_mav(y)),
            6: ("skew", f6_skew(y)),
            7: ("stetson_j", f7_stetson_j(y)),
            8: ("flux_perc_ratio", f8_flux_perc_ratio(y)),
            9: ("log_freq", f9_log_freq(f1)),
            10: ("log_amp", f10_log_amp(y)),
            11: ("r21", r21),
            12: ("ph21_cos", ph21_cos),
            13: ("ph21_sin", ph21_sin),
            14: ("freq_ratio", f14_freq_ratio(lc_periods, f1)),
            15: ("von_neumann", f15_von_neumann(y)),
            16: ("kurtosis", f16_kurtosis(y)),
            17: ("beyond1std", f17_beyond1std(y)),
        }

        # Clean & Return
        output: dict = {"id": ts_id}
        for idx in indices:
            if idx in feature_map:
                name, value = feature_map[idx]
                output[name] = value

        return output
