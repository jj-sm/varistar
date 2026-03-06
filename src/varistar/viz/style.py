"""
varistar.viz.style
==================
Shared matplotlib styling for all varistar plots.

Two themes are provided:

``apply_science_style()``   — publication-quality: monospace, inward ticks,
                              minor ticks, no top/right spines clutter.
``apply_poster_style()``    — larger fonts and thicker lines for talks / posters.

A curated colour palette ``VARISTAR_COLORS`` is defined so all modules use
the same colours for equivalent concepts (fit lines, error bars, residuals, etc.).

Usage
-----
>>> from varistar.viz.style import apply_science_style, VARISTAR_COLORS
>>> apply_science_style()
>>> # … all subsequent matplotlib calls use the science theme
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

VARISTAR_COLORS: dict[str, str] = {
    # Data points
    "data": "#2C3E50",  # Dark slate
    "data_alpha": "#7F8C8D",  # Muted grey  (error bars)
    # Model fits
    "fourier": "#E74C3C",  # Crimson
    "gaussian": "#E67E22",  # Amber
    "reference": "#2980B9",  # Steel blue  (mean / median lines)
    # Status indicators
    "good": "#27AE60",  # Emerald
    "bad": "#C0392B",  # Alizarin
    "harmonic": "#2980B9",  # Blue dot
    "not_periodic": "#E74C3C",  # Red dot
    "not_dominant": "#F39C12",  # Orange dot
    "eb_confirmed": "#2C3E50",  # Black dot
    # Residuals
    "residual": "#E74C3C",  # Crimson
    "sigma_band": "#F39C12",  # Orange
    # Periodogram
    "pgram_line": "#2C3E50",
    "fap_10": "#BDC3C7",
    "fap_1": "#95A5A6",
    "fap_0p1": "#7F8C8D",
}


# ---------------------------------------------------------------------------
# Science theme
# ---------------------------------------------------------------------------

_SCIENCE_RC: dict = {
    # Font
    "font.family": "monospace",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    # Lines
    "lines.linewidth": 0.9,
    "axes.linewidth": 0.8,
    # Grid
    "axes.grid": False,
    # Saving
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    # Figure
    "figure.autolayout": True,
}

_POSTER_RC: dict = {
    **_SCIENCE_RC,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.5,
}


def apply_science_style() -> None:
    """Apply the varistar publication-quality matplotlib theme globally."""
    plt.rcParams.update(_SCIENCE_RC)


def apply_poster_style() -> None:
    """Apply the varistar large-font poster / talk matplotlib theme globally."""
    plt.rcParams.update(_POSTER_RC)


def reset_style() -> None:
    """Restore matplotlib's default rcParams."""
    mpl.rcdefaults()


# ---------------------------------------------------------------------------
# Axis helpers used across multiple plot functions
# ---------------------------------------------------------------------------


def science_ticks(ax: plt.Axes, own_figure: bool = True) -> None:
    """
    Apply consistent tick styling to a single axis.

    Inward major + minor ticks on all four sides; rotated x-labels for
    standalone figures only (rotation inside mosaics is done per-axis by
    matplotlib).
    """
    ax.tick_params(
        axis="both", which="both", direction="in", top=True, right=True, labelsize=8
    )
    ax.minorticks_on()
    if own_figure:
        try:
            ax.ticklabel_format(useOffset=False, style="plain", axis="x")
        except Exception:
            pass
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)


def add_status_badge(
    ax: plt.Axes,
    label: str,
    color: str,
    x: float = 0.95,
    y: float = 0.95,
) -> None:
    """
    Add a coloured text badge (e.g. GOOD / BAD) to an axis corner.

    Parameters
    ----------
    ax : plt.Axes
    label : str
        Badge text.
    color : str
        Text and border colour.
    x, y : float
        Axes-fraction coordinates (default: top-right).
    """
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        color=color,
        fontweight="bold",
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor=color, linewidth=0.8),
    )


def add_status_dots(
    ax: plt.Axes,
    dots: list[str],
    x0: float = 0.05,
    y0: float = 0.95,
    dx: float = 0.08,
) -> None:
    """
    Draw small coloured ellipses as status indicators in the top-left corner.

    Convention (from LightCurve):
    * Blue   → harmonic period detected
    * Red    → no valid phase coverage (not periodic)
    * Orange → dominant period was replaced by a better-covered alternative
    * Black  → confirmed eclipsing binary (at 2× period)

    Parameters
    ----------
    ax : plt.Axes
    dots : list[str]
        Colour strings in left-to-right order.
    """
    from matplotlib.patches import Ellipse

    for i, colour in enumerate(dots):
        ax.add_patch(
            Ellipse(
                (x0 + i * dx, y0),
                0.022,
                0.030,
                transform=ax.transAxes,
                color=colour,
                zorder=10,
            )
        )


def fap_lines(
    ax: plt.Axes,
    fap_levels: dict[float, float],
    orientation: str = "horizontal",
) -> None:
    """
    Draw False Alarm Probability reference lines on a periodogram axis.

    Parameters
    ----------
    ax : plt.Axes
    fap_levels : dict[float, float]
        Mapping of ``{fap_probability: power_threshold}`` as returned by
        ``varistar.period.lomb_scargle.false_alarm_levels()``.
    orientation : str
        ``'horizontal'`` (power on y-axis, default) or ``'vertical'``.
    """
    styles = {
        0.1: ("--", VARISTAR_COLORS["fap_10"]),
        0.01: ("-.", VARISTAR_COLORS["fap_1"]),
        0.001: (":", VARISTAR_COLORS["fap_0p1"]),
    }
    for fap, power in fap_levels.items():
        ls, color = styles.get(fap, (":", "gray"))
        label = f"FAP {fap * 100:.1f}%"
        if orientation == "horizontal":
            ax.axhline(power, ls=ls, color=color, lw=0.8, alpha=0.8, label=label)
        else:
            ax.axvline(power, ls=ls, color=color, lw=0.8, alpha=0.8, label=label)
