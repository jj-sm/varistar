"""
varistar.viz.interactive
========================
Optional interactive plots powered by Plotly.

All functions in this module require ``plotly`` to be installed::

    pip install varistar[viz]
    # or
    pip install plotly

Functions return ``plotly.graph_objects.Figure`` objects so the caller can
further customise them, save them to HTML, or embed them in notebooks/dashboards.

Quick start
-----------
>>> from varistar.viz.interactive import plot_timeseries, plot_phased, plot_periodogram
>>> fig = plot_timeseries(ts)
>>> fig.show()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for type hints — avoids hard dependency at import time
    from varistar.timeseries import TimeSeries
    from varistar.lightcurve import LightCurve
    from varistar.groups import TestGroup


def _require_plotly():
    """Raise a clear ImportError if plotly is not installed."""
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive plots. "
            "Install it with: pip install plotly"
        ) from exc


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


def plot_timeseries(
    ts: "TimeSeries",
    mag_col: str | None = None,
    err_col: str | None = None,
    band_name: str = "I",
    title: str | None = None,
    height: int = 400,
    width: int = 900,
) -> "plotly.graph_objects.Figure":
    """
    Interactive time series plot with hover labels and range-selector.

    Parameters
    ----------
    ts : TimeSeries
    mag_col, err_col : str | None
        Column overrides.
    band_name : str
        Photometric band label (used in y-axis title).
    title : str | None
        Plot title.  Defaults to ``ts.timeseries_id``.
    height, width : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()
    import plotly.graph_objects as go

    mag_col = mag_col or ts.mag_col
    err_col = err_col or ts.err_col

    df = ts.timeseries_df.to_pandas()
    t, y, dy = df[ts.time_col], df[mag_col], df[err_col]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            error_y=dict(
                type="data",
                array=dy,
                visible=True,
                color="rgba(100,100,100,0.4)",
                thickness=0.8,
            ),
            mode="markers",
            marker=dict(size=4, color="#2C3E50", opacity=0.75),
            name="Data",
            hovertemplate=(
                f"HJD: %{{x:.4f}}<br>"
                f"{band_name} mag: %{{y:.4f}}<br>"
                f"±%{{error_y.array:.4f}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or ts.timeseries_id,
        xaxis_title=ts.time_scale,
        yaxis_title=f"{band_name} mag",
        yaxis_autorange="reversed",
        height=height,
        width=width,
        template="plotly_white",
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
        ),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Phase-folded light curve
# ---------------------------------------------------------------------------


def plot_phased(
    lc: "LightCurve",
    period: float | None = None,
    mag_col: str | None = None,
    err_col: str | None = None,
    band_name: str = "I",
    show_fit: bool = True,
    n_harmonics: int = 4,
    title: str | None = None,
    height: int = 450,
    width: int = 900,
) -> "plotly.graph_objects.Figure":
    """
    Interactive phase-folded light curve with optional Fourier fit overlay.

    Parameters
    ----------
    lc : LightCurve
    period : float | None
        Override period.
    show_fit : bool
        Overlay a Fourier fit line.
    n_harmonics : int
        Number of harmonics for the Fourier fit.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()
    import plotly.graph_objects as go
    import numpy as np
    from varistar.models.harmonic import fourier_series, fit_fourier

    ts = lc.timeseries
    mag_col = mag_col or ts.mag_col
    err_col = err_col or ts.err_col

    if not lc.periods and period is None:
        lc.run_ls()

    best_p = float(
        period if period is not None else (lc.periods[0] if lc.periods else 1.0)
    )

    t = ts.timeseries_df[ts.time_col].to_numpy()
    y = ts.timeseries_df[mag_col].to_numpy()
    dy = ts.timeseries_df[err_col].to_numpy()
    phase = ((t - t.min()) / best_p) % 1.0

    # Two cycles
    ph2 = np.concatenate([phase, phase + 1.0])
    y2 = np.concatenate([y, y])
    dy2 = np.concatenate([dy, dy])

    fig = go.Figure()

    # Data scatter
    fig.add_trace(
        go.Scatter(
            x=ph2,
            y=y2,
            error_y=dict(
                type="data",
                array=dy2,
                visible=True,
                color="rgba(100,100,100,0.3)",
                thickness=0.6,
            ),
            mode="markers",
            marker=dict(size=3, color="#2C3E50", opacity=0.5),
            name="Data",
            hovertemplate="Phase: %{x:.4f}<br>Mag: %{y:.4f}<extra></extra>",
        )
    )

    # Fourier fit
    if show_fit:
        popt, _, mea = fit_fourier(phase, y, n_harmonics=n_harmonics)
        if popt is not None:
            x_fit = np.linspace(0.0, 2.0, 500)
            y_fit = fourier_series(x_fit % 1.0, *popt)
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode="lines",
                    line=dict(color="#E74C3C", width=2),
                    name=f"Fourier ({n_harmonics}H)  MEA={mea:.4f}",
                )
            )

    fig.update_layout(
        title=title or f"{ts.timeseries_id}  |  P = {best_p:.5f} d",
        xaxis_title="Phase",
        yaxis_title=f"{band_name} mag",
        yaxis_autorange="reversed",
        height=height,
        width=width,
        template="plotly_white",
        xaxis=dict(range=[-0.02, 2.02]),
    )
    return fig


# ---------------------------------------------------------------------------
# Periodogram
# ---------------------------------------------------------------------------


def plot_periodogram(
    lc: "LightCurve",
    use_frequency: bool = False,
    log_period: bool = True,
    fap_levels: dict[float, float] | None = None,
    title: str | None = None,
    height: int = 400,
    width: int = 900,
) -> "plotly.graph_objects.Figure":
    """
    Interactive Lomb-Scargle power spectrum with clickable peak annotation.

    Clicking a point shows the period / frequency and power in the hover label.
    Top-5 period candidates are annotated automatically.

    Parameters
    ----------
    lc : LightCurve
    use_frequency : bool
        Plot frequency on x-axis (default: period).
    log_period : bool
        Use log scale on the period axis (ignored when use_frequency=True).
    fap_levels : dict[float, float] | None
        FAP reference lines from ``varistar.period.lomb_scargle.false_alarm_levels()``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()
    import plotly.graph_objects as go
    import numpy as np

    if not lc.power_spectra:
        lc.run_ls()

    freq = lc.power_spectra["frequency"]
    power = lc.power_spectra["power"]

    x_data = freq if use_frequency else 1.0 / freq
    xlabel = "Frequency (1/d)" if use_frequency else "Period (days)"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=power,
            mode="lines",
            line=dict(color="#2C3E50", width=0.8),
            name="LS Power",
            hovertemplate=(
                f"{xlabel[:-7] if use_frequency else 'Period'}: %{{x:.5f}}<br>"
                f"Power: %{{y:.5f}}<extra></extra>"
            ),
        )
    )

    # Annotate top-5 peaks
    colors = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#3498DB"]
    for i, p in enumerate(lc.periods[:5]):
        f_peak = 1.0 / p
        x_peak = f_peak if use_frequency else p
        idx = np.argmin(np.abs(freq - f_peak))
        pwr = float(power[idx])
        fig.add_trace(
            go.Scatter(
                x=[x_peak],
                y=[pwr],
                mode="markers+text",
                marker=dict(size=8, color=colors[i], symbol="triangle-up"),
                text=[f"P{i + 1}={p:.4f}d"],
                textposition="top center",
                textfont=dict(size=9),
                name=f"P{i + 1} = {p:.5f} d",
                showlegend=True,
            )
        )

    # FAP reference lines
    if fap_levels:
        fap_styles = {0.1: "dash", 0.01: "dashdot", 0.001: "dot"}
        for fap, power_level in fap_levels.items():
            fig.add_hline(
                y=power_level,
                line_dash=fap_styles.get(fap, "dot"),
                line_color="rgba(128,128,128,0.7)",
                line_width=1.0,
                annotation_text=f"FAP {fap * 100:.1f}%",
                annotation_position="right",
            )

    fig.update_layout(
        title=title or f"Periodogram: {lc.timeseries.timeseries_id}",
        xaxis_title=xlabel,
        yaxis_title="LS Power",
        height=height,
        width=width,
        template="plotly_white",
        xaxis_type="log" if (log_period and not use_frequency) else "linear",
    )
    return fig


# ---------------------------------------------------------------------------
# Group / mosaic (HTML export via subplot grid)
# ---------------------------------------------------------------------------


def mosaic_phased(
    group: "TestGroup",
    period_attr: str = "periods",
    max_stars: int = 16,
    n_cols: int = 4,
    height_per_row: int = 280,
    width: int = 1100,
    title: str | None = None,
) -> "plotly.graph_objects.Figure":
    """
    Build a Plotly subplot grid of phase-folded light curves.

    Intended as a browser-renderable equivalent of ``TestGroup.plot_mosaic``.
    Each panel is a mini phase-folded scatter with no fit line (for speed).

    Parameters
    ----------
    group : TestGroup
    period_attr : str
        Attribute on each object from which to read the best period.
    max_stars : int
        Maximum number of panels.
    n_cols : int
        Number of columns in the grid.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    items = group._all_objects()[:max_stars]
    n_plots = len(items)
    if n_plots == 0:
        raise ValueError("Group is empty — nothing to plot.")

    n_rows = (n_plots + n_cols - 1) // n_cols
    subplot_titles = [
        getattr(obj, "timeseries_id", f"Star {i}") for i, (obj, _) in enumerate(items)
    ]
    # Pad to fill the grid
    subplot_titles += [""] * (n_rows * n_cols - n_plots)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    status_colors = {"GOOD": "#27AE60", "BAD": "#C0392B"}

    for idx, (obj, label) in enumerate(items):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Try to get period and TS
        ts = getattr(obj, "timeseries", obj)  # LightCurve has .timeseries; TS is itself
        periods = getattr(obj, period_attr, [])
        if not periods:
            continue
        best_p = float(periods[0])

        t = ts.timeseries_df[ts.time_col].to_numpy()
        y = ts.timeseries_df[ts.mag_col].to_numpy()
        phase = ((t - t.min()) / best_p) % 1.0
        ph2 = np.concatenate([phase, phase + 1.0])
        y2 = np.concatenate([y, y])

        colour = status_colors.get(label, "#2980B9")

        fig.add_trace(
            go.Scatter(
                x=ph2,
                y=y2,
                mode="markers",
                marker=dict(size=2, color=colour, opacity=0.5),
                showlegend=False,
                hovertemplate=f"Phase: %{{x:.3f}}<br>Mag: %{{y:.3f}}<extra>{label}</extra>",
            ),
            row=row,
            col=col,
        )
        # Invert y-axis per panel
        fig.update_yaxes(autorange="reversed", row=row, col=col)

    fig.update_layout(
        title=title or f"{group.name} — Phased LC mosaic",
        height=height_per_row * n_rows,
        width=width,
        template="plotly_white",
    )
    return fig
