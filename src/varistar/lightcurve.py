"""
varistar.lightcurve
===================
LightCurve class: period finding, phase folding, model fitting, and plotting.

This class composes a ``TimeSeries`` object and extends it with period-analysis
capabilities.  Heavy-lifting computations are delegated to the sub-modules
``varistar.period`` and ``varistar.models`` so this file stays readable and
testable.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

from varistar.timeseries import TimeSeries
from varistar.models.harmonic import fourier_series, fit_fourier
from varistar.models.gaussian import (
    double_super_gaussian_model,
    fit_double_super_gaussian,
)
from varistar.period.lomb_scargle import compute_ls, compute_sr
from varistar.period.pdm import compute_pdm, compute_pdm2
from varistar.period.vs_period import select_best_period


class LightCurve:
    """
    Phase-folded light curve analysis for a single variable star.

    Attributes
    ----------
    timeseries : TimeSeries
        The underlying photometric data.
    periods : list[float]
        Candidate periods sorted by descending significance (strongest first).
    periods_map : dict[str, float]
        Named map: ``{'p1': strongest, 'p2': second, …}``.
    power_spectra : dict
        Raw output from the most recently run periodogram.
    is_periodic : bool
        False after ``find_best_period()`` determines that no valid coverage
        exists for any candidate period.
    is_harmonic : bool
        True if the dominant period and others form a harmonic series.
    sr_stats : dict
        Statistics from the last Spectrum Resampling run.
    """

    def __init__(self, timeseries: TimeSeries) -> None:
        self.timeseries = timeseries
        self.periods: list[float] = []
        self.periods_map: dict[str, float] = {}
        self.power_spectra: dict = {}
        self.is_periodic: bool = True
        self.is_harmonic: bool = False
        self.sr_stats: dict = {}

    def __repr__(self) -> str:
        p = f"{self.periods[0]:.5f} d" if self.periods else "—"
        return (
            f"LightCurve(id='{self.timeseries.timeseries_id}', "
            f"n_periods={len(self.periods)}, best_p={p})"
        )

    # ------------------------------------------------------------------
    # Period finding  (thin wrappers that store results on self)
    # ------------------------------------------------------------------

    def run_ls(
        self,
        min_freq: float = 0.001,
        max_freq: float = 10.0,
        samples_per_peak: int = 10,
        n_top: int = 20,
    ) -> dict:
        """
        Run the Lomb-Scargle periodogram and store candidate periods.

        Returns the raw power-spectrum dict (frequency, period, power,
        periods, periods_map).
        """
        if self.timeseries.timeseries_df.is_empty():
            return {}

        t, y, dy = self._get_tyd()
        result = compute_ls(
            t,
            y,
            dy,
            min_freq=min_freq,
            max_freq=max_freq,
            samples_per_peak=samples_per_peak,
            n_top=n_top,
        )
        self.power_spectra = result
        self.periods = result["periods"]
        self.periods_map = result["periods_map"]
        return result

    # Keep old name as alias
    def get_power_spectra(
        self,
        mag_col: str | None = None,
        err_col: str | None = None,
        min_freq: float = 0.001,
        max_freq: float = 10.0,
        samples_per_peak: int = 10,
    ) -> dict:
        return self.run_ls(
            min_freq=min_freq,
            max_freq=max_freq,
            samples_per_peak=samples_per_peak,
        )

    def run_pdm(
        self,
        min_freq: float = 0.001,
        max_freq: float = 10.0,
        n_freq: int = 20_000,
        n_bins: int = 10,
    ) -> float:
        """Run Stellingwerf PDM; returns the best period."""
        if self.timeseries.timeseries_df.is_empty():
            return 0.0
        t, y, _ = self._get_tyd()
        result = compute_pdm(
            t, y, min_freq=min_freq, max_freq=max_freq, n_freq=n_freq, n_bins=n_bins
        )
        self.periods = result["periods"]
        self.periods_map = {f"p{i + 1}": p for i, p in enumerate(self.periods)}
        return result["best_period"]

    # Old name alias
    get_period_pdm = run_pdm

    def run_pdm2(
        self,
        min_freq: float = 0.001,
        max_freq: float = 10.0,
        samples_per_peak: int = 10,
        phase_bins: int = 50,
    ) -> float:
        """Run binless PDM2; returns the best period."""
        if self.timeseries.timeseries_df.is_empty():
            return 0.0
        t, y, _ = self._get_tyd()
        result = compute_pdm2(
            t,
            y,
            min_freq=min_freq,
            max_freq=max_freq,
            samples_per_peak=samples_per_peak,
            phase_bins=phase_bins,
        )
        self.periods = result["periods"]
        self.periods_map = {f"p{i + 1}": p for i, p in enumerate(self.periods)}
        return result["best_period"]

    # Old name alias
    get_period_pdm2 = run_pdm2

    def run_sr(
        self,
        min_freq: float = 0.001,
        max_freq: float = 10.0,
        samples_per_peak: int = 10,
        smoothing_sigma: float = 2.0,
        n_bootstrap: int = 1_000,
    ) -> float:
        """Run Spectrum Resampling; returns the best period."""
        if self.timeseries.timeseries_df.is_empty():
            return 0.0
        t, y, dy = self._get_tyd()
        result = compute_sr(
            t,
            y,
            dy,
            min_freq=min_freq,
            max_freq=max_freq,
            samples_per_peak=samples_per_peak,
            smoothing_sigma=smoothing_sigma,
            n_bootstrap=n_bootstrap,
        )
        self.sr_stats = result
        self.periods = [result["best_period"]]
        self.periods_map = {"p1": result["best_period"]}
        return result["best_period"]

    # Old name alias
    get_period_spectrum_resampling = run_sr

    def get_periods(self) -> list[float]:
        """Return current candidate periods (sorted by significance)."""
        return self.periods

    # ------------------------------------------------------------------
    # Best-period selection
    # ------------------------------------------------------------------

    def find_best_period(self) -> dict:
        """
        Select the best physically-meaningful period from the candidate list.

        Runs ``run_ls()`` automatically if no candidates exist yet.

        Returns
        -------
        dict with keys:
            ``best_period``  — selected period (float).
            ``is_harmonic``  — whether harmonic structure was detected.
            ``not_periodic`` — True if no candidate had acceptable phase coverage.
            ``not_dominant`` — True if the LS-dominant period was replaced.
        """
        if not self.periods:
            self.run_ls()

        if not self.periods:
            return {
                "best_period": None,
                "is_harmonic": False,
                "not_periodic": True,
                "not_dominant": True,
            }

        t = self.timeseries.timeseries_df[self.timeseries.time_col].to_numpy()
        result = select_best_period(t, self.periods)

        self.is_harmonic = result["is_harmonic"]
        self.is_periodic = not result["not_periodic"]
        return result

    # ------------------------------------------------------------------
    # Plotting — periodograms
    # ------------------------------------------------------------------

    def plot_periodogram(
        self,
        use_frequency: bool = False,
        fig_size: tuple = (8, 4),
        save_path: str | None = None,
        ax: plt.Axes | None = None,
    ) -> None:
        """
        Plot the Lomb-Scargle power spectrum.

        Parameters
        ----------
        use_frequency : bool
            Plot frequency (x-axis) instead of period.
        """
        if not self.power_spectra:
            self.run_ls()

        freq = self.power_spectra.get("frequency")
        power = self.power_spectra.get("power")

        ax, own = self._ax_or_figure(ax, fig_size)

        x_data = freq if use_frequency else 1.0 / freq
        xlabel = "Frequency (1/d)" if use_frequency else "Period (days)"

        ax.plot(x_data, power, color="black", linewidth=0.8)
        if not use_frequency:
            ax.set_xscale("log")

        tf, lf = (14, 12) if own else (10, 9)
        ax.set_title(
            self.timeseries.timeseries_id
            if use_frequency
            else f"Periodogram: {self.timeseries.timeseries_id}",
            fontsize=tf,
        )
        ax.set_ylabel("Power", fontsize=lf)
        ax.set_xlabel(xlabel, fontsize=lf)
        ax.tick_params(
            axis="both", which="both", direction="in", top=True, right=True, labelsize=8
        )
        ax.minorticks_on()
        ax.grid(True, which="major", linestyle="--", alpha=0.3)
        self._finalise(ax, own, save_path)

    def plot_frequency_series(
        self,
        fig_size: tuple = (8, 4),
        save_path: str | None = None,
        ax: plt.Axes | None = None,
    ) -> None:
        """Plot the power spectrum on a frequency axis."""
        self.plot_periodogram(
            use_frequency=True, fig_size=fig_size, save_path=save_path, ax=ax
        )

    # ------------------------------------------------------------------
    # Plotting — phase-folded light curve  (merged from 3 original methods)
    # ------------------------------------------------------------------

    def plot_phased(
        self,
        period: float | None = None,
        mag_col: str | None = None,
        err_col: str | None = None,
        band_name: str = "I",
        n_harmonics: int = 4,
        fit_model: str = "fourier",  # 'fourier' | 'gaussian' | None
        show_residuals: bool = False,
        fig_size: tuple = (8, 4),
        save_path: str | Path | None = None,
        ax: plt.Axes | None = None,
        dots: list[str] | None = None,
    ) -> None:
        """
        Plot the phase-folded light curve with an optional model fit and
        residuals panel.

        This method replaces ``plot_light_curve``, ``plot_light_curve_temp``,
        and ``plot_light_curve_v2``.

        Parameters
        ----------
        period : float | None
            Override period.  Defaults to ``self.periods[0]`` (or runs LS).
        fit_model : str | None
            ``'fourier'`` — n-harmonic Fourier series.
            ``'gaussian'`` — Double Super-Gaussian (good for EBs).
            ``None`` — no model fit.
        show_residuals : bool
            Attach a residuals panel below the main plot.
        dots : list[str] | None
            List of colour strings for status indicator dots in the top-left
            corner (e.g. ``['blue', 'red']``).
        """
        ts = self.timeseries
        if ts.timeseries_df.is_empty():
            if ax is not None:
                ax.text(0.5, 0.5, "No Data", ha="center", transform=ax.transAxes)
            else:
                print("[plot_phased] No data.")
            return

        mag_col = mag_col or ts.mag_col
        err_col = err_col or ts.err_col

        # --- Resolve period ---
        best_p = self._resolve_period(period, ax)
        if best_p is None:
            return

        # --- Data ---
        t = ts.timeseries_df[ts.time_col].to_numpy()
        y = ts.timeseries_df[mag_col].to_numpy()
        dy = ts.timeseries_df[err_col].to_numpy()
        phase = ((t - t.min()) / best_p) % 1.0

        # --- Axes ---
        ax, own = self._ax_or_figure(ax, fig_size)
        ax_res = None
        if show_residuals:
            divider = make_axes_locatable(ax)
            ax_res = divider.append_axes("bottom", size="30%", pad=0.1, sharex=ax)

        # --- Status dots ---
        if dots:
            for i, colour in enumerate(dots):
                ax.add_patch(
                    Ellipse(
                        (0.05 + i * 0.08, 0.95),
                        0.02,
                        0.028,
                        transform=ax.transAxes,
                        color=colour,
                        zorder=10,
                    )
                )

        # --- Model fit ---
        popt = None
        fit_func = None

        if fit_model == "fourier":
            popt, _, _ = fit_fourier(phase, y, n_harmonics=n_harmonics)
            if popt is not None:
                x_fit = np.linspace(0, 2, 400)
                ax.plot(
                    x_fit,
                    fourier_series(x_fit % 1.0, *popt),
                    color="crimson",
                    lw=2,
                    zorder=3,
                    label="Fourier Fit",
                )
                fit_func = lambda x, *p: fourier_series(x, *p)  # noqa: E731

        elif fit_model == "gaussian":
            popt, _ = fit_double_super_gaussian(phase, y)
            if popt is not None:
                x_fit = np.linspace(0, 2, 400)
                ax.plot(
                    x_fit,
                    double_super_gaussian_model(x_fit % 1.0, *popt),
                    color="darkorange",
                    lw=2,
                    zorder=3,
                    label="Super-Gaussian Fit",
                )
                fit_func = lambda x, *p: double_super_gaussian_model(x, *p)  # noqa: E731

        # --- Data (2 cycles) ---
        ph2 = np.concatenate([phase, phase + 1.0])
        y2 = np.concatenate([y, y])
        dy2 = np.concatenate([dy, dy])

        ax.errorbar(
            ph2,
            y2,
            yerr=dy2,
            fmt="o",
            markersize=2,
            color="black",
            ecolor="lightgray",
            alpha=0.3,
            capsize=0,
            label="Data",
        )

        # Reference lines
        ax.axhline(
            np.median(y), color="royalblue", ls="--", lw=1, alpha=0.6, label="Median"
        )
        ax.axhline(np.mean(y), color="seagreen", ls="--", lw=1, alpha=0.6, label="Mean")

        # --- Residuals panel ---
        if ax_res is not None:
            if popt is not None and fit_func is not None:
                res = y - fit_func(phase, *popt)
                res2 = np.concatenate([res, res])
                ax_res.errorbar(
                    ph2,
                    res2,
                    yerr=dy2,
                    fmt="o",
                    markersize=2,
                    color="crimson",
                    ecolor="lightgray",
                    alpha=0.4,
                )
                ax_res.axhline(0.0, color="black", lw=1, alpha=0.5)
                r_med = np.median(res)
                r_mad = np.median(np.abs(res - r_med)) * 1.4826  # Robust σ
                ax_res.axhline(
                    r_med + r_mad, color="darkorange", ls=":", lw=1, alpha=0.7
                )
                ax_res.axhline(
                    r_med - r_mad, color="darkorange", ls=":", lw=1, alpha=0.7
                )
                ax_res.invert_yaxis()
            else:
                ax_res.text(
                    0.5,
                    0.5,
                    "No Fit",
                    ha="center",
                    va="center",
                    transform=ax_res.transAxes,
                )

            ax_res.set_ylabel("Res", fontsize=8)
            ax_res.set_xlabel("Phase", fontsize=9)
            ax_res.grid(True, alpha=0.1)
            ax_res.tick_params(
                axis="both",
                which="both",
                direction="in",
                top=True,
                right=True,
                labelsize=8,
            )
            ax_res.set_xlim(-0.02, 2.02)
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Phase", fontsize=9 if not own else 10)

        # --- Formatting ---
        tf, lf = (11, 10) if own else (10, 8)
        ax.set_title(f"{ts.timeseries_id} | P={best_p:.5f} d", fontsize=tf)
        ax.set_ylabel(f"{band_name} mag", fontsize=lf)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.1)
        ax.tick_params(
            axis="both", which="both", direction="in", top=True, right=True, labelsize=8
        )
        ax.minorticks_on()
        ax.set_xlim(-0.02, 2.02)

        if own:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(loc="upper right", fontsize=7, frameon=True, ncol=2)

        self._finalise(ax, own, save_path)

    # Old name aliases kept for backward compatibility
    def plot_light_curve(self, **kwargs) -> None:
        """Alias for ``plot_phased(show_residuals=False)``."""
        kwargs.setdefault("show_residuals", False)
        self.plot_phased(**kwargs)

    def plot_light_curve_temp(self, **kwargs) -> None:
        """Alias for ``plot_phased(show_residuals=True, fit_model='fourier')``."""
        kwargs.setdefault("show_residuals", True)
        kwargs.setdefault("fit_model", "fourier")
        self.plot_phased(**kwargs)

    def plot_light_curve_v2(self, **kwargs) -> None:
        """Alias for ``plot_phased(show_residuals=True)``."""
        kwargs.setdefault("show_residuals", True)
        self.plot_phased(**kwargs)

    # ------------------------------------------------------------------
    # Plotting — best-period convenience wrappers
    # ------------------------------------------------------------------

    def plot_best(
        self,
        mag_col: str | None = None,
        fit_model: str = "fourier",
        mea_range: tuple[float, float] = (0.01, 0.5),
        show_residuals: bool = True,
        force_fourier_harmonics: int | None = None,
        **kwargs,
    ) -> None:
        """
        Find the best period, check for eclipsing binary morphology, and plot.

        This method replaces ``plot_best_period``, ``plot_best_period_temp``,
        and ``plot_best_period_v2``.

        Parameters
        ----------
        fit_model : str
            Starting fit model.  If EB is detected and *force_fourier_harmonics*
            is not set, this is overridden to ``'gaussian'``.
        mea_range : tuple[float, float]
            (min, max) MEA bounds for the EB classifier.
        show_residuals : bool
            Show the residuals panel.
        force_fourier_harmonics : int | None
            If set, always use Fourier with this many harmonics (skips EB check).
        """
        # Import here to avoid circular dependency
        from varistar.classify.eb_detector import score_eb

        mag_col = mag_col or self.timeseries.mag_col

        period_data = self.find_best_period()
        best_p = float(period_data["best_period"])

        dots: list[str] = []
        if period_data.get("is_harmonic"):
            dots.append("blue")
        if period_data.get("not_periodic"):
            dots.append("red")
        if period_data.get("not_dominant"):
            dots.append("orange")

        final_period = best_p
        final_fit = fit_model

        if force_fourier_harmonics is not None:
            # User explicitly requested forced Fourier — skip EB logic
            final_fit = "fourier"
            kwargs["n_harmonics"] = force_fourier_harmonics
        else:
            # EB check
            is_eb, mea_1x, _ = score_eb(
                self, period=best_p, mag_col=mag_col, mea_range=mea_range
            )
            if is_eb:
                p_2x = best_p * 2.0
                _, mea_2x = fit_double_super_gaussian(*self._phase_mag(p_2x, mag_col))
                if mea_2x < mea_1x:
                    final_period = p_2x
                    dots.append("black")  # Black dot = confirmed EB at 2×P
                final_fit = "gaussian"

        self.plot_phased(
            period=final_period,
            mag_col=mag_col,
            fit_model=final_fit,
            show_residuals=show_residuals,
            dots=dots,
            **kwargs,
        )

    # Old name aliases
    def plot_best_period(self, **kwargs) -> None:
        kwargs.setdefault("show_residuals", False)
        self.plot_best(**kwargs)

    def plot_best_period_temp(self, **kwargs) -> None:
        kwargs.setdefault("show_residuals", True)
        self.plot_best(**kwargs)

    def plot_best_period_v2(self, **kwargs) -> None:
        kwargs.setdefault("show_residuals", True)
        self.plot_best(**kwargs)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "timeseries_id": self.timeseries.timeseries_id,
            "periods": [float(round(p, 7)) for p in self.periods],
            "is_periodic": self.is_periodic,
            "is_harmonic": self.is_harmonic,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_tyd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract time, magnitude, and error as numpy arrays."""
        df = self.timeseries.timeseries_df
        tc, mc, ec = (
            self.timeseries.time_col,
            self.timeseries.mag_col,
            self.timeseries.err_col,
        )
        return df[tc].to_numpy(), df[mc].to_numpy(), df[ec].to_numpy()

    def _phase_mag(
        self,
        period: float,
        mag_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (phase, mag) folded at *period*."""
        t, y, _ = self._get_tyd()
        phase = ((t - t.min()) / period) % 1.0
        return phase, y

    def _resolve_period(
        self,
        period: float | None,
        ax: plt.Axes | None,
    ) -> float | None:
        """
        Return a valid period, running LS if needed.
        Posts an error message to *ax* (or prints) and returns None on failure.
        """
        if period is not None:
            return float(period)
        if self.periods:
            return float(self.periods[0])
        self.run_ls()
        if self.periods:
            return float(self.periods[0])
        msg = "No Period Found"
        if ax is not None:
            ax.text(0.5, 0.5, msg, ha="center", transform=ax.transAxes)
        else:
            print(f"[LightCurve] {msg}")
        return None

    @staticmethod
    def _ax_or_figure(
        ax: plt.Axes | None,
        fig_size: tuple,
    ) -> tuple[plt.Axes, bool]:
        if ax is None:
            plt.rcParams.update({"font.family": "monospace"})
            _, ax = plt.subplots(figsize=fig_size)
            return ax, True
        return ax, False

    @staticmethod
    def _finalise(
        ax: plt.Axes,
        own_figure: bool,
        save_path: str | Path | None,
    ) -> None:
        if not own_figure:
            return
        plt.tight_layout()
        if save_path:
            sp = Path(save_path)
            sp.parent.mkdir(parents=True, exist_ok=True)
            ax.get_figure().savefig(str(sp), bbox_inches="tight", dpi=200)
            plt.close(ax.get_figure())
        else:
            plt.show()
            plt.close(ax.get_figure())
