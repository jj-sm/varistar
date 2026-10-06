"""Synthetic photometry generators used by the varistar docs examples.

Not part of the public varistar API -- just reproducible sample data so
every notebook in this docs instance can execute standalone, without
needing real survey files on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sinusoidal_lightcurve(
    n: int = 400,
    baseline_days: float = 120.0,
    period: float = 3.2453,
    mean_mag: float = 15.0,
    amplitude: float = 0.25,
    noise: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a clean sinusoidal variable (RR Lyrae-like) with Gaussian noise.

    Columns match varistar's default TimeSeries colnames:
    ``["hjd", "mag_i", "m_error"]``.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, baseline_days, n))
    mag = (
        mean_mag + amplitude * np.sin(2 * np.pi * t / period) + rng.normal(0, noise, n)
    )
    err = np.full(n, noise)
    return pd.DataFrame({"hjd": t, "mag_i": mag, "m_error": err})


def make_eclipsing_lightcurve(
    n: int = 600,
    baseline_days: float = 150.0,
    period: float = 5.7,
    mean_mag: float = 14.5,
    depth: float = 0.6,
    width: float = 0.04,
    noise: float = 0.015,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a detached eclipsing-binary-like light curve with two dips per cycle."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, baseline_days, n))
    phase = (t / period) % 1.0

    def dip(center: float, d: float) -> np.ndarray:
        dist = np.minimum(np.abs(phase - center), 1 - np.abs(phase - center))
        return d * np.exp(-0.5 * (dist / width) ** 2)

    mag = mean_mag + dip(0.0, depth) + dip(0.5, depth * 0.4) + rng.normal(0, noise, n)
    err = np.full(n, noise)
    return pd.DataFrame({"hjd": t, "mag_i": mag, "m_error": err})
