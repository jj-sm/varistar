"""
varistar.catalog.tess
=====================
Loader for TESS photometry delivered via `lightkurve` or as FITS files.

Two loading paths are supported:

1. **FITS files** — standard TESS light curve (``.fits`` / ``.fit``) products
   from MAST, containing a ``LIGHTCURVE`` binary table extension with columns
   ``TIME``, ``PDCSAP_FLUX`` (or ``SAP_FLUX``), and ``PDCSAP_FLUX_ERR``.

2. **lightkurve** — if ``lightkurve`` is installed, ``load_from_tic()``
   queries MAST automatically and returns the same Polars DataFrame.

All loaders return a **Polars DataFrame** whose columns match the
``varistar.TimeSeries`` convention: ``[time_col, mag_col, err_col]``.

Notes on TESS flux → magnitude conversion
------------------------------------------
TESS delivers flux (e-/s), not magnitudes.  The helpers here convert using:

    mag = TESS_mag_zero - 2.5 * log10(flux)

where ``TESS_mag_zero = 20.44`` (based on Vega zero-point for the TESS band).
Set ``as_magnitude=False`` to keep flux units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

# TESS zero-point for flux → magnitude conversion
_TESS_MAG_ZERO: float = 20.44


# ---------------------------------------------------------------------------
# FITS file loader
# ---------------------------------------------------------------------------

def load_fits(
    filepath: str | Path,
    flux_column: str = "PDCSAP_FLUX",
    time_col: str = "hjd",
    mag_col: str = "mag_t",
    err_col: str = "m_error",
    as_magnitude: bool = True,
    quality_bitmask: int = 175,
    time_offset: float = 2_457_000.0,
) -> pl.DataFrame:
    """
    Load a TESS FITS light curve file.

    Parameters
    ----------
    filepath : str | Path
        Path to the ``.fits`` file.
    flux_column : str
        FITS column name for flux: ``'PDCSAP_FLUX'`` (detrended, default) or
        ``'SAP_FLUX'`` (simple aperture).
    time_col, mag_col, err_col : str
        Output column names in the returned DataFrame.
    as_magnitude : bool
        If True, convert flux to magnitude using the TESS zero-point.
        If False, flux values are kept as-is and column names still apply.
    quality_bitmask : int
        Observations where ``QUALITY & quality_bitmask != 0`` are removed.
        175 removes the most common systematics (default lightkurve value).
    time_offset : float
        Offset added to TESS BTJD (Barycentric TESS JD) to convert to HJD.
        TESS BTJD = BJD − 2 457 000; default converts to full BJD.

    Returns
    -------
    pl.DataFrame
        Columns: ``[time_col, mag_col, err_col]``.

    Raises
    ------
    ImportError
        If ``astropy`` is not installed.
    FileNotFoundError
        If *filepath* does not exist.
    """
    try:
        from astropy.io import fits as afits
    except ImportError as exc:
        raise ImportError(
            "astropy is required to load TESS FITS files. "
            "Install it with: pip install astropy"
        ) from exc

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"TESS FITS file not found: {path}")

    with afits.open(str(path)) as hdul:
        # TESS LC products store the light curve in extension 1
        lc_ext = hdul[1]
        data   = lc_ext.data

        time    = data["TIME"].astype(np.float64)
        flux    = data[flux_column].astype(np.float64)
        err_col_fits = flux_column + "_ERR"
        flux_err = (
            data[err_col_fits].astype(np.float64)
            if err_col_fits in data.names
            else np.full_like(flux, np.nan)
        )
        quality = data["QUALITY"].astype(np.int32) if "QUALITY" in data.names else np.zeros_like(time, dtype=np.int32)

    # --- Quality masking ---
    good = (quality & quality_bitmask) == 0
    # Also remove NaN flux / time
    good &= np.isfinite(time) & np.isfinite(flux)
    time, flux, flux_err = time[good], flux[good], flux_err[good]

    # --- Time conversion ---
    time = time + time_offset

    # --- Flux → magnitude ---
    if as_magnitude:
        valid_flux = flux > 0.0
        mag = np.where(
            valid_flux,
            _TESS_MAG_ZERO - 2.5 * np.log10(np.where(valid_flux, flux, 1.0)),
            np.nan,
        )
        # Error propagation:  σ_mag = 2.5 / ln(10) * σ_flux / flux
        mag_err = np.where(
            valid_flux & (flux_err > 0),
            (2.5 / np.log(10.0)) * np.abs(flux_err / np.where(valid_flux, flux, 1.0)),
            np.nan,
        )
    else:
        mag, mag_err = flux, flux_err

    # --- Build DataFrame ---
    df = pl.DataFrame({
        time_col: time,
        mag_col:  mag,
        err_col:  mag_err,
    })
    return df.drop_nulls(subset=[time_col, mag_col, err_col])


# ---------------------------------------------------------------------------
# lightkurve query loader
# ---------------------------------------------------------------------------

def load_from_tic(
    tic_id: int | str,
    sector: int | None = None,
    flux_column: str = "pdcsap_flux",
    time_col: str = "hjd",
    mag_col: str = "mag_t",
    err_col: str = "m_error",
    as_magnitude: bool = True,
    quality_bitmask: str = "default",
    author: str = "SPOC",
) -> pl.DataFrame:
    """
    Download and load a TESS light curve via ``lightkurve``.

    Parameters
    ----------
    tic_id : int | str
        TESS Input Catalog identifier (e.g. ``261136679`` or ``'TIC 261136679'``).
    sector : int | None
        TESS sector number.  ``None`` → use the first available sector.
    flux_column : str
        lightkurve flux type: ``'pdcsap_flux'`` or ``'sap_flux'``.
    as_magnitude : bool
        Convert flux to TESS magnitude (see module docstring).
    quality_bitmask : str | int
        Passed directly to ``lightkurve.search_lightcurve``.

    Returns
    -------
    pl.DataFrame
        Columns: ``[time_col, mag_col, err_col]``.

    Raises
    ------
    ImportError
        If ``lightkurve`` is not installed.
    """
    try:
        import lightkurve as lk
    except ImportError as exc:
        raise ImportError(
            "lightkurve is required for load_from_tic(). "
            "Install it with: pip install lightkurve"
        ) from exc

    tic_str = f"TIC {tic_id}" if not str(tic_id).startswith("TIC") else str(tic_id)
    search = lk.search_lightcurve(tic_str, sector=sector, author=author)
    if len(search) == 0:
        raise ValueError(f"No TESS light curves found for {tic_str} sector={sector}.")

    lc = search[0].download(quality_bitmask=quality_bitmask).normalize()
    lc = lc.remove_nans()

    time     = lc.time.value.astype(np.float64) + 2_457_000.0
    flux     = getattr(lc, flux_column).value.astype(np.float64)
    flux_err = getattr(lc, flux_column + "_err", lc.flux_err).value.astype(np.float64)

    if as_magnitude:
        valid = flux > 0.0
        mag = np.where(
            valid,
            _TESS_MAG_ZERO - 2.5 * np.log10(np.where(valid, flux, 1.0)),
            np.nan,
        )
        mag_err = np.where(
            valid & (flux_err > 0),
            (2.5 / np.log(10.0)) * np.abs(flux_err / np.where(valid, flux, 1.0)),
            np.nan,
        )
    else:
        mag, mag_err = flux, flux_err

    df = pl.DataFrame({
        time_col: time,
        mag_col:  mag,
        err_col:  mag_err,
    })
    return df.drop_nulls(subset=[time_col, mag_col, err_col])


# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------

def tess_fits_metadata(filepath: str | Path) -> dict:
    """
    Extract key header keywords from a TESS FITS file.

    Returns a dict with keys: ``tic_id``, ``sector``, ``camera``,
    ``ccd``, ``ra``, ``dec``, ``tess_mag``, ``exptime``.
    Unknown fields are returned as ``None``.
    """
    try:
        from astropy.io import fits as afits
    except ImportError:
        return {}

    path = Path(filepath)
    meta: dict = {}
    with afits.open(str(path)) as hdul:
        hdr = hdul[0].header
        meta["tic_id"]   = hdr.get("TICID",   None)
        meta["sector"]   = hdr.get("SECTOR",   None)
        meta["camera"]   = hdr.get("CAMERA",   None)
        meta["ccd"]      = hdr.get("CCD",      None)
        meta["ra"]       = hdr.get("RA_OBJ",   None)
        meta["dec"]      = hdr.get("DEC_OBJ",  None)
        meta["tess_mag"] = hdr.get("TESSMAG",  None)
        meta["exptime"]  = hdr.get("FRAMETIM", None)
    return meta