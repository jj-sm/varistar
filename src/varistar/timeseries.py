"""
varistar.timeseries
===================
Core TimeSeries class: data I/O, cleaning, statistics, and basic visualisation.

This module is survey-agnostic.  Data loading from specific survey formats
(OGLE, ASAS-SN, TESS) is handled by ``varistar.catalog``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats
from pathlib import Path


class TimeSeries:
    """
    Container for a single photometric time series.

    Attributes
    ----------
    timeseries_id : str
        Unique identifier (usually the filename stem).
    timeseries_df : pl.DataFrame
        Active (possibly cleaned) data.
    timeseries_df_orig : pd.DataFrame
        Snapshot of the original data, used by ``reset()``.
    colnames : list[str]
        Column names: ``[time_col, mag_col, err_col]``.
    magnitude : str
        Human-readable magnitude label (e.g. ``'mag I'``).
    time_scale : str
        Human-readable time label (e.g. ``'HJD'``).
    """

    def __init__(
        self,
        magnitude: str,
        time_scale: str,
        colnames: list[str] | None = None,
    ) -> None:
        self.colnames: list[str] = colnames if colnames is not None else ["hjd", "mag_i", "m_error"]
        self.timeseries_df: pl.DataFrame = pl.DataFrame()
        self.timeseries_df_orig: pd.DataFrame = pd.DataFrame()
        self.timeseries_id: str = ""
        self.magnitude = magnitude
        self.time_scale = time_scale

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def time_col(self) -> str:
        return self.colnames[0]

    @property
    def mag_col(self) -> str:
        return self.colnames[1]

    @property
    def err_col(self) -> str:
        return self.colnames[2]

    def __len__(self) -> int:
        return len(self.timeseries_df)

    def __repr__(self) -> str:
        return (
            f"TimeSeries(id='{self.timeseries_id}', "
            f"n={len(self)}, "
            f"mag='{self.magnitude}', "
            f"time='{self.time_scale}')"
        )

    # ------------------------------------------------------------------
    # Internal snapshot
    # ------------------------------------------------------------------

    def _copy_original_db(self) -> None:
        """Store a copy of the current DataFrame as the reset checkpoint."""
        self.timeseries_df_orig = self.timeseries_df.to_pandas()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Restore ``timeseries_df`` to its original state (before any cleaning).
        No-op if no original snapshot has been saved.
        """
        if self.timeseries_df_orig is None or self.timeseries_df_orig.empty:
            self.timeseries_df = pl.DataFrame()
            return
        self.timeseries_df = pl.from_pandas(self.timeseries_df_orig.copy())

    # Keep the old name as an alias for backward compatibility with groups.py
    reset_ts = reset

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def apply_mask(self, mask: pl.Series) -> None:
        """
        Remove rows where *mask* is True.

        Parameters
        ----------
        mask : pl.Series[bool]
            Boolean Series aligned with ``timeseries_df``; True = remove.
        """
        self.timeseries_df = self.timeseries_df.filter(~mask)

    # Old name alias
    apply_mask_to_df = apply_mask

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data_from_file(self, filepath: str | Path) -> None:
        """
        Load photometry from a survey data file.

        The loader is selected automatically based on file extension:
        * ``.dat`` → OGLE-style whitespace-delimited file via ``varistar.catalog.ogle``.
        * Other formats → generic CSV/TSV loader via ``varistar.catalog.generic``.

        Parameters
        ----------
        filepath : str | Path
            Path to the photometry file.
        """
        path = Path(filepath)
        if not path.exists():
            print(f"[TimeSeries] File not found: {path}")
            return

        self.timeseries_id = path.stem

        try:
            if path.suffix.lower() == ".dat":
                from varistar.catalog.ogle import load_dat
                self.timeseries_df = load_dat(str(path), col_names=self.colnames)
            else:
                from varistar.catalog.generic import load_csv
                self.timeseries_df = load_csv(
                    str(path),
                    time_col=self.time_col,
                    mag_col=self.mag_col,
                    err_col=self.err_col,
                )
            self._copy_original_db()
        except Exception as exc:
            print(f"[TimeSeries] Error reading {path}: {exc}")

    def load_data_from_df(
        self,
        df: pd.DataFrame | pl.DataFrame,
        data_id: int | str,
    ) -> None:
        """
        Load photometry from an existing DataFrame.

        Parameters
        ----------
        df : pd.DataFrame | pl.DataFrame
            Source data.  Must contain columns matching ``self.colnames``.
        data_id : int | str
            Identifier assigned to ``timeseries_id``.
        """
        self.timeseries_id = str(data_id)
        self.timeseries_df = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
        self._copy_original_db()

    # ------------------------------------------------------------------
    # Cleaning helpers
    # ------------------------------------------------------------------

    def clean_by_error(self, max_error: float) -> None:
        """
        Remove observations whose photometric error exceeds *max_error*.

        Parameters
        ----------
        max_error : float
            Upper threshold for the error column.
        """
        if self.timeseries_df.is_empty():
            return
        before = len(self.timeseries_df)
        self.timeseries_df = self.timeseries_df.filter(
            pl.col(self.err_col) <= max_error
        )
        print(f"[clean_by_error] Removed {before - len(self.timeseries_df)} points "
              f"(err > {max_error}).")

    def mask_iqr_outliers(
        self,
        column: str | None = None,
        error_column: str | None = None,
        k: float = 1.5,
    ) -> pl.Series:
        """
        Build an IQR-based outlier mask on the fractional photometric error.

        Does **not** modify ``timeseries_df`` — pass the returned mask to
        ``apply_mask()`` to actually remove the flagged rows.

        Parameters
        ----------
        column : str | None
            Magnitude column (defaults to ``self.mag_col``).
        error_column : str | None
            Error column (defaults to ``self.err_col``).
        k : float
            IQR multiplier.  Standard box-plot uses 1.5; conservative = 3.0.

        Returns
        -------
        pl.Series[bool]
            True where the fractional error is an outlier.
        """
        if self.timeseries_df.is_empty():
            return pl.Series([], dtype=pl.Boolean)

        col = column or self.mag_col
        err = error_column or self.err_col

        temp = self.timeseries_df.with_columns(
            (pl.col(err) / pl.col(col)).alias("_frac_err")
        )
        q = temp.select([
            pl.col("_frac_err").quantile(0.25).alias("q1"),
            pl.col("_frac_err").quantile(0.75).alias("q3"),
        ])
        q1, q3 = q[0, "q1"], q[0, "q3"]
        iqr = q3 - q1
        mask = (temp["_frac_err"] < q1 - k * iqr) | (temp["_frac_err"] > q3 + k * iqr)
        return mask

    # Old name alias kept for backward compatibility
    def stats_outlier_clipping(
        self,
        column: str = "mag_i",
        error_column: str = "m_error",
        k: float = 1.5,
    ) -> pl.Series:
        return self.mask_iqr_outliers(column=column, error_column=error_column, k=k)

    def mask_sigma_clip(
        self,
        column: str,
        n_sigma: float,
        max_iter: int = 5,
    ) -> pl.DataFrame:
        """
        Iterative sigma-clipping on a chosen column.

        Returns the clipped DataFrame (does **not** modify in-place).
        Useful for getting a clean copy without permanently altering the object.

        Parameters
        ----------
        column : str
            Column to clip.
        n_sigma : float
            Clipping threshold in units of the sample standard deviation.
        max_iter : int
            Maximum iterations.

        Returns
        -------
        pl.DataFrame
            The cleaned subset of ``timeseries_df``.
        """
        if self.timeseries_df.is_empty():
            return self.timeseries_df.clone()

        df = self.timeseries_df.clone()
        for _ in range(max_iter):
            s = df.select([
                pl.col(column).median().alias("med"),
                pl.col(column).std().alias("std"),
            ])
            med, std = s[0, "med"], s[0, "std"]
            lo, hi = med - n_sigma * std, med + n_sigma * std
            new_df = df.filter((pl.col(column) >= lo) & (pl.col(column) <= hi))
            if len(new_df) == len(df):
                break
            df = new_df
        return df

    # Old name alias
    stats_sigma_clipping = mask_sigma_clip

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict | None:
        """
        Return a nested dict of descriptive statistics for the magnitude and
        error columns.

        Returns
        -------
        dict | None
            Nested dict ``{col_name: {count, mean, std, min, max, median}}``;
            None if the DataFrame is empty.
        """
        if self.timeseries_df.is_empty():
            print("[TimeSeries] Warning: no data to analyse.")
            return None

        def _col_stats(col: str) -> dict:
            s = self.timeseries_df[col]
            return {
                "count":  len(s),
                "mean":   s.mean(),
                "std":    s.std(),
                "min":    s.min(),
                "max":    s.max(),
                "median": s.median(),
            }

        return {
            self.mag_col: _col_stats(self.mag_col),
            self.err_col: _col_stats(self.err_col),
        }

    def summary(self) -> None:
        """Print a concise one-line summary of the time series."""
        if self.timeseries_df.is_empty():
            print(f"[{self.timeseries_id}] — empty")
            return
        t = self.timeseries_df[self.time_col].to_numpy()
        m = self.timeseries_df[self.mag_col].to_numpy()
        print(
            f"[{self.timeseries_id}]  "
            f"n={len(self.timeseries_df)}  "
            f"baseline={np.ptp(t):.1f} d  "
            f"<mag>={np.mean(m):.3f}  "
            f"amp={np.ptp(m):.3f}"
        )

    def get_baseline(self) -> float:
        """Return the total time baseline (max - min HJD) in days."""
        if self.timeseries_df.is_empty():
            return 0.0
        t = self.timeseries_df[self.time_col].to_numpy()
        return float(np.ptp(t))

    def get_cadence(self) -> dict:
        """
        Return cadence statistics.

        Returns
        -------
        dict with keys:
            ``median_cadence`` — median time gap between consecutive observations (days).
            ``min_cadence``    — minimum gap.
            ``max_cadence``    — maximum gap.
        """
        if len(self.timeseries_df) < 2:
            return {"median_cadence": 0.0, "min_cadence": 0.0, "max_cadence": 0.0}
        t = np.sort(self.timeseries_df[self.time_col].to_numpy())
        diffs = np.diff(t)
        return {
            "median_cadence": float(np.median(diffs)),
            "min_cadence":    float(np.min(diffs)),
            "max_cadence":    float(np.max(diffs)),
        }

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------

    def bin_data(self, time_window: float = 0.1) -> None:
        """
        Average observations that fall within the same time bin.

        Errors are propagated correctly as sqrt(Σσ²) / N, reducing the
        effective noise by √N while preserving all columns.

        Parameters
        ----------
        time_window : float
            Bin width in days.
        """
        if self.timeseries_df.is_empty():
            return

        df = self.timeseries_df.with_columns(
            (pl.col(self.time_col) / time_window).round().cast(pl.Int64).alias("_bin_id")
        )
        binned = (
            df.group_by("_bin_id")
            .agg([
                pl.col(self.time_col).mean(),
                pl.col(self.mag_col).mean(),
                (
                    pl.col(self.err_col).pow(2).sum().sqrt()
                    / pl.col(self.err_col).count()
                ).alias(self.err_col),
            ])
            .sort(self.time_col)
        )
        n_before = len(self.timeseries_df)
        self.timeseries_df = binned.select(self.colnames)
        print(f"[bin_data] {n_before} → {len(self.timeseries_df)} points "
              f"(window={time_window} d).")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_pandas(self) -> pd.DataFrame:
        """Return the active data as a pandas DataFrame."""
        return self.timeseries_df.to_pandas()

    def to_polars(self) -> pl.DataFrame:
        """Return the active data as a polars DataFrame."""
        return self.timeseries_df.clone()

    def to_dict(self) -> dict:
        """Return a flat dict of key attributes for DataFrame export."""
        return {
            "timeseries_id": self.timeseries_id,
            "magnitude":     self.magnitude,
            "time_scale":    self.time_scale,
            "data_points":   len(self.timeseries_df),
            "baseline_days": self.get_baseline(),
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _ax_or_figure(
        self,
        ax: plt.Axes | None,
        fig_size: tuple,
    ) -> tuple[plt.Axes, bool]:
        """Return (ax, own_figure).  Creates a new figure when ax is None."""
        if ax is None:
            plt.rcParams.update({"font.family": "monospace"})
            _, ax = plt.subplots(figsize=fig_size)
            return ax, True
        return ax, False

    @staticmethod
    def _finalise(ax: plt.Axes, own_figure: bool, save_path: str | None) -> None:
        """tight_layout + save/show for standalone figures."""
        if not own_figure:
            return
        plt.tight_layout()
        if save_path:
            ax.get_figure().savefig(save_path, bbox_inches="tight", dpi=200)
            plt.close(ax.get_figure())
        else:
            plt.show()

    @staticmethod
    def _science_ticks(ax: plt.Axes, own_figure: bool) -> None:
        """Apply common tick styling."""
        ax.tick_params(axis="both", which="both", direction="in",
                       top=True, right=True, labelsize=8)
        ax.minorticks_on()
        if own_figure:
            ax.ticklabel_format(useOffset=False, style="plain", axis="x")
            plt.xticks(rotation=30, fontsize=9)
        else:
            ax.tick_params(axis="x", rotation=30)

    # --- plot_timeseries ---

    def plot_timeseries(
        self,
        df: pl.DataFrame | None = None,
        mag_col: str | None = None,
        err_col: str | None = None,
        band_name: str = "I",
        set_time_to_zero: bool = False,
        fig_size: tuple = (8, 4),
        save_path: str | None = None,
        ax: plt.Axes | None = None,
        **kwargs,
    ) -> None:
        """
        Plot the time series with error bars.

        Parameters
        ----------
        df : pl.DataFrame | None
            Override the internal DataFrame (useful for comparisons).
        mag_col, err_col : str | None
            Column overrides; default to ``self.mag_col`` / ``self.err_col``.
        band_name : str
            Photometric band label for the y-axis.
        set_time_to_zero : bool
            Shift the time axis so the first observation is at t = 0.
        ax : plt.Axes | None
            External axis for mosaic embedding.  A new figure is created if None.
        """
        plot_df = df if df is not None else self.timeseries_df
        mag_col = mag_col or self.mag_col
        err_col = err_col or self.err_col

        if plot_df.is_empty():
            if ax is not None:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                        transform=ax.transAxes)
            else:
                print("[plot_timeseries] Warning: no data to plot.")
            return

        data = plot_df.to_pandas()
        x = data[self.time_col]
        if set_time_to_zero:
            x = x - x.min()

        ax, own = self._ax_or_figure(ax, fig_size)

        ax.errorbar(
            x, data[mag_col], yerr=data[err_col],
            fmt="o", color="steelblue", ecolor="gray",
            markersize=3, capsize=2, alpha=0.7, elinewidth=0.8,
        )

        tf = 14 if own else 10
        lf = 12 if own else 9
        ax.set_title(self.timeseries_id, fontsize=tf)
        ax.set_ylabel(f"{band_name} mag", fontsize=lf)
        ax.set_xlabel("Time (Days)" if set_time_to_zero else self.time_scale, fontsize=lf)
        ax.invert_yaxis()
        self._science_ticks(ax, own)
        self._finalise(ax, own, save_path)

    # --- plot_cleaned ---

    def plot_cleaned(
        self,
        df_original: pl.DataFrame,
        *masks: pl.Series,
        labels: list[str] | None = None,
        fig_size: tuple = (8, 4),
        save_path: str | None = None,
        ax: plt.Axes | None = None,
    ) -> None:
        """
        Overlay kept observations and discarded outlier sets in different colours.

        Parameters
        ----------
        df_original : pl.DataFrame
            Unfiltered DataFrame to plot from.
        *masks : pl.Series[bool]
            One or more boolean masks (True = discarded).
        labels : list[str] | None
            Legend labels for each mask.
        """
        if df_original.is_empty():
            if ax is not None:
                ax.text(0.5, 0.5, "No Data", ha="center", transform=ax.transAxes)
            return

        ax, own = self._ax_or_figure(ax, fig_size)

        try:
            pdf = df_original.to_pandas()
            np_masks = [np.array(m).flatten() for m in masks]

            # Kept points
            combined = np.logical_or.reduce(np_masks) if np_masks else np.zeros(len(pdf), dtype=bool)
            kept = pdf[~combined]
            ax.errorbar(
                kept[self.time_col], kept[self.mag_col], yerr=kept[self.err_col],
                fmt="o", color="steelblue", ecolor="gray",
                markersize=3, capsize=2, alpha=0.4, elinewidth=0.8, label="Kept",
            )

            # Discarded subsets
            _colors  = ["crimson", "forestgreen", "darkorange", "purple", "black"]
            _markers = ["x", "s", "^", "d", "v"]
            for i, mask in enumerate(np_masks):
                sub = pdf[mask]
                if sub.empty:
                    continue
                lbl = labels[i] if (labels and i < len(labels)) else f"Method {i + 1}"
                ax.errorbar(
                    sub[self.time_col], sub[self.mag_col], yerr=sub[self.err_col],
                    fmt=_markers[i % len(_markers)],
                    markersize=4,
                    color=_colors[i % len(_colors)],
                    ecolor=_colors[i % len(_colors)],
                    elinewidth=1, alpha=0.9, capsize=2,
                    label=f"{lbl} (#{len(sub)})",
                )

            tf, lf = (14, 12) if own else (10, 9)
            ax.set_title(self.timeseries_id, fontsize=tf)
            ax.set_ylabel(self.magnitude, fontsize=lf)
            ax.set_xlabel(self.time_scale, fontsize=lf)
            ax.invert_yaxis()
            self._science_ticks(ax, own)
            ax.legend(loc="best", fontsize=8, frameon=True)

        except Exception as exc:
            print(f"[plot_cleaned] Error: {exc}")
            if own:
                plt.close()
            return

        self._finalise(ax, own, save_path)

    # Old alias
    plot_timeseries_cleaned = plot_cleaned

    # --- plot_mag_distribution ---

    def plot_mag_distribution(
        self,
        column: str | None = None,
        band_name: str = "I",
        fig_size: tuple = (8, 4),
        save_path: str | None = None,
        ax: plt.Axes | None = None,
    ) -> None:
        """
        Histogram of the magnitude distribution with a fitted normal curve overlay.
        """
        if self.timeseries_df.is_empty():
            if ax is not None:
                ax.text(0.5, 0.5, "No Data", ha="center", transform=ax.transAxes)
            return

        col = column or self.mag_col
        data = self.timeseries_df[col].to_numpy()
        mu, sigma = np.mean(data), np.std(data)

        ax, own = self._ax_or_figure(ax, fig_size)

        ax.hist(data, bins=30, density=True, alpha=0.6,
                color="royalblue", edgecolor="navy", label="Data")
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        ax.plot(x, scipy_stats.norm.pdf(x, mu, sigma),
                color="crimson", lw=2, label=f"Normal fit\n(μ={mu:.3f}, σ={sigma:.3f})")

        tf, lf = (14, 12) if own else (10, 9)
        ax.set_title(f"Distribution: {self.timeseries_id}", fontsize=tf)
        ax.set_xlabel(f"{band_name} Magnitude", fontsize=lf)
        ax.set_ylabel("Probability Density", fontsize=lf)
        ax.tick_params(axis="both", which="both", direction="in",
                       top=True, right=True, labelsize=8)
        ax.minorticks_on()
        ax.legend(loc="best", fontsize=8 if not own else 9, frameon=True)
        self._finalise(ax, own, save_path)

    # Old alias
    plot_normal_distribution = plot_mag_distribution