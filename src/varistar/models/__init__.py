"""varistar.models — Mathematical fit functions."""
from varistar.models.harmonic import fourier_series, fit_fourier, amplitude_r21, phase_phi21
from varistar.models.gaussian import (
    gaussian_model, double_gaussian_model,
    super_gaussian_model, double_super_gaussian_model,
    fit_double_super_gaussian,
)
__all__ = [
    "fourier_series", "fit_fourier", "amplitude_r21", "phase_phi21",
    "gaussian_model", "double_gaussian_model",
    "super_gaussian_model", "double_super_gaussian_model",
    "fit_double_super_gaussian",
]