"""
varistar.catalog.gaiadr3
========================
Loader for Gaia DR3 epoch photometry CSV files.

Gaia epoch photometry files are downloaded from the Gaia archive as CSV with the
filename pattern::

    EPOCH_PHOTOMETRY-Gaia DR3 <source_id>.csv

Each row represents a single transit.  Three photometric bands are present:
G (``g_transit_*``), BP (``bp_*``), and RP (``rp_*``).  BP and RP columns are
empty when only a G observation was recorded.

Times are in Barycentric JD(TCB) offset by the Gaia reference epoch::

    t_file = BJD(TCB) - 2455197.5        (days since 2010-01-01.0 TCB)

Pass ``time_offset=2455197.5`` to recover full BJD.

All loaders return a **Polars DataFrame** with three columns:
``[time_col, mag_col, err_col]``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pathlib import Path

# Gaia time zero-point: BJD(TCB) - 2455197.5
GAIA_TIME_OFFSET = 2_455_197.5

# 2.5 / ln(10) — converts SNR to magnitude uncertainty
_MAG_ERR_FACTOR = 2.5 / np.log(10)

# Band configuration: (time_col, flux_col, flux_err_col, mag_col, reject_flag_col)
_BAND_COLS: dict[str, tuple[str, str, str, str, str]] = {
    "G": (
        "g_transit_time",
        "g_transit_flux",
        "g_transit_flux_error",
        "g_transit_mag",
        "variability_flag_g_reject",
    ),
    "BP": (
        "bp_obs_time",
        "bp_flux",
        "bp_flux_error",
        "bp_mag",
        "variability_flag_bp_reject",
    ),
    "RP": (
        "rp_obs_time",
        "rp_flux",
        "rp_flux_error",
        "rp_mag",
        "variability_flag_rp_reject",
    ),
}


# ---------------------------------------------------------------------------
# Primary loader
# ---------------------------------------------------------------------------


def load_csv(
    filepath: str | Path,
    band: str = "G",
    col_names: list[str] | None = None,
    time_offset: float = 0.0,
    filter_rejected: bool = True,
) -> pl.DataFrame:
    """
    Load a Gaia DR3 epoch photometry CSV into a Polars DataFrame.

    Parameters
    ----------
    filepath : str | Path
        Path to the ``EPOCH_PHOTOMETRY-*.csv`` file.
    band : str
        Photometric band to extract: ``'G'``, ``'BP'``, or ``'RP'``.
    col_names : list[str] | None
        Output column names ``[time, mag, err]``.  Defaults to
        ``['bjd', 'mag_g', 'mag_err']`` (band letter substituted for mag/err).
    time_offset : float
        Added to every time value after loading.  Pass ``2455197.5`` to
        convert Gaia file times to full BJD(TCB).  Default is 0.
    filter_rejected : bool
        Drop rows where the band's ``variability_flag_*_reject`` column is
        ``true``.  Default is True.

    Returns
    -------
    pl.DataFrame
        Three-column DataFrame ``[time, mag, err]``, sorted by time.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If *band* is not ``'G'``, ``'BP'``, or ``'RP'``, or if no valid rows
        remain after filtering.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Gaia DR3 CSV not found: {path}")

    band = band.upper()
    if band not in _BAND_COLS:
        raise ValueError(f"band must be one of {list(_BAND_COLS)}, got {band!r}")

    t_col, flux_col, flux_err_col, mag_col, reject_col = _BAND_COLS[band]

    band_lower = band.lower()
    default_names = ["bjd", f"mag_{band_lower}", f"mag_{band_lower}_err"]
    col_names = col_names or default_names
    if len(col_names) != 3:
        raise ValueError(
            f"col_names must have exactly 3 elements, got {len(col_names)}."
        )

    df = pl.read_csv(path, infer_schema_length=500, null_values=["", "null"])

    if filter_rejected:
        if reject_col in df.columns:
            df = df.filter(pl.col(reject_col) != True)  # noqa: E712

    # Drop rows where band columns are null (BP/RP absent for some transits)
    df = df.filter(
        pl.col(t_col).is_not_null()
        & pl.col(flux_col).is_not_null()
        & pl.col(flux_err_col).is_not_null()
        & pl.col(mag_col).is_not_null()
    )

    if df.is_empty():
        raise ValueError(
            f"No valid {band}-band rows in {path.name} after filtering."
        )

    times = df[t_col].cast(pl.Float64) + time_offset
    mags = df[mag_col].cast(pl.Float64)
    # mag uncertainty from SNR: σ_mag = 2.5 / ln(10) * flux_err / flux
    fluxes = df[flux_col].cast(pl.Float64)
    flux_errs = df[flux_err_col].cast(pl.Float64)
    mag_errs = (_MAG_ERR_FACTOR * flux_errs / fluxes).abs()

    result = (
        pl.DataFrame(
            {
                col_names[0]: times,
                col_names[1]: mags,
                col_names[2]: mag_errs,
            }
        )
        .filter(pl.col(col_names[2]).is_finite() & pl.col(col_names[2]).gt(0))
        .sort(col_names[0])
    )

    if result.is_empty():
        raise ValueError(
            f"No valid {band}-band rows remain after error filtering in {path.name}."
        )

    return result


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def parse_gaia_id(filepath: str | Path) -> dict:
    """
    Extract Gaia DR3 source information from a file path.

    The filename convention is::

        EPOCH_PHOTOMETRY-Gaia DR3 <source_id>.csv

    Parameters
    ----------
    filepath : str | Path

    Returns
    -------
    dict with keys:
        ``stem``      — filename without extension.
        ``source_id`` — Gaia source ID string, or ``''`` if not found.
        ``release``   — catalogue release string (e.g. ``'Gaia DR3'``), or ``''``.
    """
    stem = Path(filepath).stem  # e.g. "EPOCH_PHOTOMETRY-Gaia DR3 4689115826279313408"
    prefix = "EPOCH_PHOTOMETRY-"
    if stem.startswith(prefix):
        remainder = stem[len(prefix):]  # "Gaia DR3 4689115826279313408"
        parts = remainder.rsplit(" ", maxsplit=1)
        if len(parts) == 2:
            return {"stem": stem, "release": parts[0], "source_id": parts[1]}

    return {"stem": stem, "release": "", "source_id": ""}


def load_csv_directory(
    directory: str | Path,
    band: str = "G",
    col_names: list[str] | None = None,
    time_offset: float = 0.0,
    filter_rejected: bool = True,
    glob: str = "EPOCH_PHOTOMETRY-*.csv",
    max_files: int | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Load all Gaia DR3 epoch photometry CSV files in a directory.

    Parameters
    ----------
    directory : str | Path
        Directory to scan.
    band : str
        Photometric band (``'G'``, ``'BP'``, or ``'RP'``).
    col_names : list[str] | None
        Passed to ``load_csv``.
    time_offset : float
        Passed to ``load_csv``.
    filter_rejected : bool
        Passed to ``load_csv``.
    glob : str
        Glob pattern (default ``'EPOCH_PHOTOMETRY-*.csv'``).
    max_files : int | None
        Cap on number of files loaded (useful for testing).

    Returns
    -------
    dict[str, pl.DataFrame]
        Mapping of ``source_id → DataFrame``.  Files that fail to parse are
        skipped with a warning.
    """
    directory = Path(directory)
    files = sorted(directory.glob(glob))
    if max_files is not None:
        files = files[:max_files]

    results: dict[str, pl.DataFrame] = {}
    for fp in files:
        meta = parse_gaia_id(fp)
        key = meta["source_id"] or fp.stem
        try:
            results[key] = load_csv(
                fp,
                band=band,
                col_names=col_names,
                time_offset=time_offset,
                filter_rejected=filter_rejected,
            )
        except Exception as exc:
            print(f"[gaiadr3.load_csv_directory] Skipping {fp.name}: {exc}")

    print(f"[gaiadr3] Loaded {len(results)}/{len(files)} files from {directory}.")
    return results
