"""
varistar.catalog.ogle
=====================
Loaders for OGLE-II / III / IV photometry files.

OGLE ``.dat`` files are whitespace-delimited plain text with three columns:

    HJD   magnitude   error

Some variants prepend a short header (lines starting with ``#``).
All loaders return a **Polars DataFrame** with user-specified column names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Primary loader
# ---------------------------------------------------------------------------

def load_dat(
    filepath: str | Path,
    col_names: list[str] | None = None,
    skip_comment_char: str = "#",
    time_offset: float = 0.0,
) -> pl.DataFrame:
    """
    Load an OGLE ``.dat`` photometry file into a Polars DataFrame.

    Parameters
    ----------
    filepath : str | Path
        Path to the ``.dat`` file.
    col_names : list[str] | None
        Column names to assign.  Defaults to ``['hjd', 'mag_i', 'm_error']``.
        Must have exactly 3 elements.
    skip_comment_char : str
        Lines beginning with this character are treated as comments and ignored.
    time_offset : float
        Constant added to every time value after loading (e.g. 2_450_000 to
        convert truncated HJD to full HJD).  Default is 0 (no offset).

    Returns
    -------
    pl.DataFrame
        Three-column DataFrame in the order ``[time, mag, err]``.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file contains fewer than 3 numeric columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"OGLE .dat file not found: {path}")

    col_names = col_names or ["hjd", "mag_i", "m_error"]
    if len(col_names) != 3:
        raise ValueError(f"col_names must have exactly 3 elements, got {len(col_names)}.")

    rows: list[tuple[float, float, float]] = []
    with path.open("r") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith(skip_comment_char):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                t   = float(parts[0]) + time_offset
                mag = float(parts[1])
                err = float(parts[2])
            except ValueError:
                continue
            rows.append((t, mag, err))

    if not rows:
        raise ValueError(f"No valid data rows found in: {path}")

    t_arr   = np.array([r[0] for r in rows], dtype=np.float64)
    mag_arr = np.array([r[1] for r in rows], dtype=np.float64)
    err_arr = np.array([r[2] for r in rows], dtype=np.float64)

    return pl.DataFrame({
        col_names[0]: t_arr,
        col_names[1]: mag_arr,
        col_names[2]: err_arr,
    })


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def parse_ogle_id(filepath: str | Path) -> dict:
    """
    Extract the OGLE source identifier components from a file path.

    OGLE filenames typically follow the pattern::

        OGLE-LMC-RRLYR-01234.dat
        BLG501.05.123456.dat

    Parameters
    ----------
    filepath : str | Path

    Returns
    -------
    dict with keys:
        ``stem``    — filename without extension.
        ``field``   — survey field code (e.g. ``'LMC'``, ``'BLG501'``),
                      or ``''`` if the pattern is not recognised.
        ``sequence``— numeric sequence number as a string, or ``''``.
    """
    stem = Path(filepath).stem
    parts = stem.split("-")

    field    = parts[1] if len(parts) >= 3 else ""
    sequence = parts[-1] if len(parts) >= 2 else ""

    return {"stem": stem, "field": field, "sequence": sequence}


def load_dat_directory(
    directory: str | Path,
    col_names: list[str] | None = None,
    glob: str = "*.dat",
    max_files: int | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Load all ``.dat`` files in a directory.

    Parameters
    ----------
    directory : str | Path
        Directory to scan.
    col_names : list[str] | None
        Passed to ``load_dat``.
    glob : str
        Glob pattern (default ``'*.dat'``).
    max_files : int | None
        Cap on number of files loaded (useful for testing).

    Returns
    -------
    dict[str, pl.DataFrame]
        Mapping of ``stem → DataFrame``.  Files that fail to parse are
        skipped with a warning.
    """
    directory = Path(directory)
    files = sorted(directory.glob(glob))
    if max_files is not None:
        files = files[:max_files]

    results: dict[str, pl.DataFrame] = {}
    for fp in files:
        try:
            results[fp.stem] = load_dat(fp, col_names=col_names)
        except Exception as exc:
            print(f"[ogle.load_dat_directory] Skipping {fp.name}: {exc}")

    print(f"[ogle] Loaded {len(results)}/{len(files)} files from {directory}.")
    return results