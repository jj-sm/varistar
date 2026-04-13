import numpy as np
from astropy.visualization import ZScaleInterval

# Clean fits
def clean_channel(img: np.ndarray) -> np.ndarray:
    """
    Cleans a FITS image by applying a zscale normalization. 
    This function takes a 2D array representing the image data, 
    identifies the finite values, and applies a zscale normalization 
    to scale the pixel values to the range [0, 1]. 
    The function handles cases where there are no finite values 
    or where the scale is not valid, ensuring that the output is
    a properly normalized image suitable for visualization.
    Parameters:
        img (np.ndarray): A 2D array representing the FITS image data.
    Returns:
        np.ndarray: A 2D array of the same shape as the input.
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
    """
    Computes the convex hull of a set of 2D points using the monotone
    chain algorithm.
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