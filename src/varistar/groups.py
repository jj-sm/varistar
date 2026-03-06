"""
varistar.groups
===============
Batch processing container for collections of TimeSeries / LightCurve objects.

Objects are labelled as GOOD / BAD / ANY so that visual inspection and batch
operations can treat different subsets differently.
"""

from __future__ import annotations

import math
from typing import Callable

import matplotlib.pyplot as plt
import polars as pl


class TestGroup:
    """
    Container for labelled collections of TimeSeries / LightCurve objects.

    Parameters
    ----------
    good_ts : list
        Objects classified as "good" (e.g. clean light curves).
    bad_ts : list
        Objects classified as "bad" (e.g. problematic or rejected).
    any_ts : list | None
        Objects without a binary good/bad label (e.g. "unknown", "candidate").
    name : str
        Human-readable name for this group (used in export and summaries).
    status_str : str
        Label string shown for ``any_ts`` objects in mosaics and exports.
    """

    def __init__(
        self,
        good_ts: list,
        bad_ts: list,
        any_ts: list | None = None,
        name: str = "group",
        status_str: str = "LC",
    ) -> None:
        self.good_ts: list = good_ts
        self.bad_ts: list = bad_ts
        self.any_ts: list = any_ts if any_ts is not None else []
        self.name = name
        self.status_s = status_str

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.good_ts) + len(self.bad_ts) + len(self.any_ts)

    def __repr__(self) -> str:
        return (
            f"TestGroup(name='{self.name}', "
            f"good={len(self.good_ts)}, "
            f"bad={len(self.bad_ts)}, "
            f"any={len(self.any_ts)})"
        )

    def _all_objects(self) -> list[tuple]:
        """Yield (obj, label_str) for every registered object."""
        return (
            [(obj, "GOOD") for obj in self.good_ts]
            + [(obj, "BAD") for obj in self.bad_ts]
            + [(obj, self.status_s) for obj in self.any_ts]
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a brief summary of the group composition."""
        print(
            f"TestGroup '{self.name}'  |  "
            f"GOOD: {len(self.good_ts)}  "
            f"BAD: {len(self.bad_ts)}  "
            f"ANY/{self.status_s}: {len(self.any_ts)}  "
            f"(total: {len(self)})"
        )

    # ------------------------------------------------------------------
    # Mosaic plotting
    # ------------------------------------------------------------------

    def plot_mosaic(
        self,
        plot_method: Callable,
        n_cols: int = 4,
        max_plots: int = 16,
        save_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Create a grid of plots by calling *plot_method* on each object.

        Parameters
        ----------
        plot_method : Callable
            An **unbound** class method, e.g. ``TimeSeries.plot_timeseries``
            or ``LightCurve.plot_phased``.  It is called as
            ``plot_method(obj, ax=ax, **kwargs)``.
        n_cols : int
            Number of columns in the grid.
        max_plots : int
            Maximum total panels (sliced from GOOD first, then BAD, then ANY).
        save_path : str | None
            If given, save the figure to this path instead of showing it.
        **kwargs
            Forwarded verbatim to *plot_method*.

        Notes
        -----
        Each panel receives a coloured status label:
        ``GOOD`` → green, ``BAD`` → red, ``ANY`` / custom → blue.
        """
        items = self._all_objects()[:max_plots]
        n_plots = len(items)

        if n_plots == 0:
            print("[TestGroup] No objects to plot.")
            return

        n_rows = math.ceil(n_plots / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 4, n_rows * 3),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        _status_colors = {"GOOD": "green", "BAD": "red"}

        last_i = 0
        for i, (ax, (obj, label)) in enumerate(zip(axes_flat, items)):
            last_i = i
            try:
                plot_method(obj, ax=ax, **kwargs)
            except Exception as exc:
                print(f"[plot_mosaic] Error on object {i}: {exc}")
                ax.text(
                    0.5, 0.5, "Error", ha="center", va="center", transform=ax.transAxes
                )

            colour = _status_colors.get(label, "steelblue")
            ax.text(
                0.95,
                0.95,
                label,
                transform=ax.transAxes,
                color=colour,
                fontweight="bold",
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor=colour),
            )

        # Hide unused axes
        for j in range(last_i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def apply(self, method: Callable, **kwargs) -> None:
        """
        Call an unbound method on every registered object.

        Parameters
        ----------
        method : Callable
            An unbound class method, e.g. ``TimeSeries.bin_data``.
        **kwargs
            Forwarded to *method*.

        Example
        -------
        >>> group.apply(TimeSeries.clean_by_error, max_error=0.05)
        >>> group.apply(TimeSeries.bin_data, time_window=0.5)
        """
        for obj, _ in self._all_objects():
            try:
                method(obj, **kwargs)
            except Exception as exc:
                obj_id = getattr(obj, "timeseries_id", repr(obj))
                print(f"[apply] Error on '{obj_id}': {exc}")

    # Old name alias
    def apply_fun(self, method: Callable, **kwargs) -> None:
        self.apply(method, **kwargs)

    def reset_all(self) -> None:
        """Call ``reset()`` on every registered object."""
        for obj, _ in self._all_objects():
            obj.reset()

    # Old name alias
    def reset_data(self) -> None:
        self.reset_all()

    def run_periodicity(self, method: str = "ls") -> None:
        """
        Run period-finding on all registered objects.

        Parameters
        ----------
        method : str
            ``'ls'``   → Lomb-Scargle (``LightCurve.run_ls``).
            ``'pdm'``  → Stellingwerf PDM.
            ``'pdm2'`` → Binless PDM2.
        """
        _dispatch = {"ls": "run_ls", "pdm": "run_pdm", "pdm2": "run_pdm2"}
        attr = _dispatch.get(method, "run_ls")

        for obj, _ in self._all_objects():
            fn = getattr(obj, attr, None)
            if fn is None:
                print(f"[run_periodicity] Object has no method '{attr}'.")
                continue
            try:
                fn()
            except Exception as exc:
                obj_id = getattr(obj, "timeseries_id", repr(obj))
                print(f"[run_periodicity] Error on '{obj_id}': {exc}")

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by(
        self,
        attr: str,
        condition: Callable,
        target: str = "any",
    ) -> "TestGroup":
        """
        Return a new ``TestGroup`` containing only objects that satisfy
        ``condition(getattr(obj, attr))``.

        Parameters
        ----------
        attr : str
            Attribute name to inspect on each object.
        condition : Callable
            A predicate, e.g. ``lambda v: v > 0.5``.
        target : str
            Which sub-list to populate in the returned group: ``'good'``,
            ``'bad'``, or ``'any'`` (default).

        Returns
        -------
        TestGroup
            New group containing only the matching objects.
        """
        matching = [
            obj for obj, _ in self._all_objects() if condition(getattr(obj, attr, None))
        ]
        kwargs: dict = {"good_ts": [], "bad_ts": [], "any_ts": []}
        kwargs[f"{target}_ts"] = matching
        return TestGroup(
            name=f"{self.name}[filtered]",
            status_str=self.status_s,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_attributes(self) -> pl.DataFrame:
        """
        Build a Polars DataFrame of key attributes from all registered objects.

        Each object must implement ``to_dict()`` (both ``TimeSeries`` and
        ``LightCurve`` do).  A ``status`` column is appended automatically.

        Returns
        -------
        pl.DataFrame
            One row per object; columns depend on ``to_dict()`` output.
        """
        rows: list[dict] = []
        for obj, label in self._all_objects():
            attrs = obj.to_dict()
            attrs["status"] = label
            rows.append(attrs)
        return pl.DataFrame(rows)

    # Alias used in ml/data.py
    to_dataframe = export_attributes

    def export_periods(self) -> pl.DataFrame:
        """
        Convenience wrapper: export only ``timeseries_id`` and period columns.

        Objects that have no ``periods`` attribute are skipped silently.
        """
        rows: list[dict] = []
        for obj, label in self._all_objects():
            periods = getattr(obj, "periods", None)
            if periods is None:
                continue
            rows.append(
                {
                    "timeseries_id": getattr(obj, "timeseries_id", "unknown"),
                    "best_period": float(periods[0]) if periods else None,
                    "n_candidates": len(periods),
                    "status": label,
                }
            )
        return pl.DataFrame(rows)
