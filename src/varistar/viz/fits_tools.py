"""FITS image normalization and convex-hull helpers for the FITS viewer."""

import numpy as np
from astropy.visualization import ZScaleInterval


def clean_channel(img: np.ndarray) -> np.ndarray:
    """Normalize a FITS image channel to [0, 1] using zscale limits.

    Non-finite pixels are excluded when computing the zscale limits
    and are mapped to 0 in the output.

    Parameters
    ----------
    img : np.ndarray
        2D array of FITS image data.

    Returns
    -------
    np.ndarray
        2D array of the same shape as `img`, normalized to [0, 1].
        Returns an all-zero array if `img` has no finite values or
        the zscale limits collapse to a non-positive range.
    """
    img = np.asarray(img, dtype=float)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img, dtype=float)

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(img[finite])
    scale = vmax - vmin
    if not np.isfinite(scale) or scale <= 0:
        scaled = np.zeros_like(img, dtype=float)
    else:
        scaled = (img - vmin) / scale

    scaled = np.where(finite, scaled, 0.0)
    return np.clip(scaled, 0.0, 1.0)

def convex_hull(points: np.ndarray) -> np.ndarray:
    """Compute the convex hull of a set of 2D points.

    Uses Andrew's monotone chain algorithm, O(n log n).

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 2) with 2D point coordinates.

    Returns
    -------
    np.ndarray
        Array of shape (m, 2) with the hull vertices in
        counter-clockwise order, starting from the lowest point.
    """
    if points.shape[0] <= 1:
        return points
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])