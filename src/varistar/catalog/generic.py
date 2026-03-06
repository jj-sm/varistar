"""
varistar.catalog.generic
=========================
Survey-agnostic loader for any delimited photometry file (CSV, TSV, whitespace).

Use this when your data does not come from a supported survey adapter
(OGLE, ASAS-SN, TESS) or when you want full control over column mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Primary loader
# ---------------------------------------------------------------------------

def load_csv(
    filepath: str | Path,
    time_col: str = "hjd",
    mag_col: str = "mag",
    err_col: str = "err",
    sep: str | None = None,
    comment_char: str = "#",
    skip_rows: int = 0,
    time_offset: float = 0.0,
    mag_offset: float = 0.0,
    quality_col: str | None = None,
    quality_values: list | None = None,
) -> pl.DataFrame:
    """
    Load a delimited photometry file into a Polars DataFrame.

    The loader auto-detects the delimiter unless *sep* is specified.
    Blank lines and comment lines are always skipped.

    Parameters
    ----------
    filepath : str | Path
        Path to the data file.
    time_col : str
        Name of the time column in the file header.
    mag_col : str
        Name of the magnitude column.
    err_col : str
        Name of the photometric error column.
    sep : str | None
        Column delimiter.  ``None`` → auto-detect (tries ``,``, ``\\t``,
        whitespace in that order).
    comment_char : str
        Lines starting with this character are skipped.
    skip_rows : int
        Number of non-comment header rows to skip before the column header.
    time_offset : float
        Added to every time value after loading.
    mag_offset : float
        Added to every magnitude value after loading (e.g. for zero-point shift).
    quality_col : str | None
        Optional column name to filter on.
    quality_values : list | None
        If *quality_col* is set, only rows whose *quality_col* value is in
        this list are kept (e.g. ``['A', 'B']`` for ASAS-SN grade filtering).

    Returns
    -------
    pl.DataFrame
        Columns: ``[time_col, mag_col, err_col]`` (plus quality col if used).

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    KeyError
        If any required column is missing from the parsed file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # --- Read raw lines ---
    with path.open("r") as fh:
        lines = fh.readlines()

    # Strip comment lines and blank lines
    data_lines = [
        ln for ln in lines
        if ln.strip() and not ln.strip().startswith(comment_char)
    ]

    if skip_rows:
        data_lines = data_lines[skip_rows:]

    if not data_lines:
        raise ValueError(f"No data lines found in: {path}")

    # --- Detect separator ---
    header_line = data_lines[0]
    if sep is None:
        sep = _detect_sep(header_line)

    # --- Parse header ---
    header = [c.strip() for c in header_line.split(sep)]
    body   = data_lines[1:]

    required = [time_col, mag_col, err_col]
    missing  = [c for c in required if c not in header]
    if missing:
        raise KeyError(
            f"Required columns {missing} not found in header. "
            f"Available: {header}"
        )

    # --- Build column arrays ---
    col_idx = {name: header.index(name) for name in header}
    n_cols  = len(header)

    arrays: dict[str, list[float | str]] = {h: [] for h in header}
    for ln in body:
        parts = ln.strip().split(sep)
        if len(parts) < n_cols:
            continue
        for i, h in enumerate(header):
            arrays[h].append(parts[i].strip())

    # --- Convert to Polars ---
    series: dict[str, pl.Series] = {}
    for col in header:
        raw = arrays[col]
        if col in (time_col, mag_col, err_col):
            series[col] = _to_float_series(col, raw)
        else:
            series[col] = pl.Series(col, raw)

    df = pl.DataFrame(series)

    # --- Offsets ---
    if time_offset:
        df = df.with_columns((pl.col(time_col) + time_offset).alias(time_col))
    if mag_offset:
        df = df.with_columns((pl.col(mag_col) + mag_offset).alias(mag_col))

    # --- Quality filter ---
    if quality_col and quality_values is not None and quality_col in df.columns:
        df = df.filter(pl.col(quality_col).is_in(quality_values))

    # --- Drop NaN rows in core columns ---
    df = df.drop_nulls(subset=[time_col, mag_col, err_col])

    return df.select([time_col, mag_col, err_col])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_sep(line: str) -> str:
    """Heuristic separator detection: comma > tab > whitespace."""
    if "," in line:
        return ","
    if "\t" in line:
        return "\t"
    return None  # None → str.split() splits on any whitespace run


def _to_float_series(name: str, raw: list[str]) -> pl.Series:
    """Parse a list of strings to a Float64 Polars Series, coercing errors to null."""
    vals: list[float | None] = []
    for v in raw:
        try:
            vals.append(float(v))
        except (ValueError, TypeError):
            vals.append(None)
    return pl.Series(name, vals, dtype=pl.Float64)


# ---------------------------------------------------------------------------
# Convenience: load from numpy / pandas arrays
# ---------------------------------------------------------------------------

def from_arrays(
    time: "np.ndarray",
    mag: "np.ndarray",
    err: "np.ndarray",
    col_names: list[str] | None = None,
) -> pl.DataFrame:
    """
    Wrap three numpy arrays into a Polars DataFrame suitable for
    ``TimeSeries.load_data_from_df()``.

    Parameters
    ----------
    time, mag, err : np.ndarray
        Observation times, magnitudes, and errors.
    col_names : list[str] | None
        Column names.  Defaults to ``['hjd', 'mag', 'err']``.

    Returns
    -------
    pl.DataFrame
    """
    import numpy as np

    col_names = col_names or ["hjd", "mag", "err"]
    if len(col_names) != 3:
        raise ValueError("col_names must have exactly 3 elements.")
    return pl.DataFrame({
        col_names[0]: np.asarray(time,  dtype=np.float64),
        col_names[1]: np.asarray(mag,   dtype=np.float64),
        col_names[2]: np.asarray(err,   dtype=np.float64),
    })