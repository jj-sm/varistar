"""
varistar.viz
============
Plotting engine for varistar.

Static plots (matplotlib) are in ``varistar.viz.style``.
Interactive plots (plotly) are in ``varistar.viz.interactive``.

Apply the science theme globally::

    from varistar.viz.style import apply_science_style
    apply_science_style()
"""
from varistar.viz.style import (
    apply_science_style,
    apply_poster_style,
    reset_style,
    VARISTAR_COLORS,
    science_ticks,
    add_status_badge,
    add_status_dots,
    fap_lines,
)
__all__ = [
    "apply_science_style", "apply_poster_style", "reset_style",
    "VARISTAR_COLORS",
    "science_ticks", "add_status_badge", "add_status_dots", "fap_lines",
]