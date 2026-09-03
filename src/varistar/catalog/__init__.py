"""
varistar.catalog
================
Survey-specific I/O adapters.

Each module returns a ``pl.DataFrame`` with columns
``[time_col, mag_col, err_col]`` compatible with
``TimeSeries.load_data_from_df()``.

Quick reference
---------------
>>> from varistar.catalog.ogle import load_dat
>>> from varistar.catalog.generic import load_csv, from_arrays
>>> from varistar.catalog.tess import load_fits, load_from_tic
>>> from varistar.catalog.gaiadr3 import load_csv as gaiadr3_load_csv, GAIA_TIME_OFFSET
"""

from varistar.catalog.ogle import load_dat, load_dat_directory
from varistar.catalog.generic import load_csv, from_arrays
from varistar.catalog.tess import load_fits, load_from_tic
from varistar.catalog.gaia import load_gaia
from varistar.catalog.gaiadr3 import (
    load_csv as gaiadr3_load_csv,
    load_csv_directory as gaiadr3_load_csv_directory,
    parse_gaia_id,
    GAIA_TIME_OFFSET,
)

__all__ = [
    "load_dat",
    "load_dat_directory",
    "load_csv",
    "from_arrays",
    "load_fits",
    "load_from_tic",
    "gaiadr3_load_csv",
    "gaiadr3_load_csv_directory",
    "parse_gaia_id",
    "GAIA_TIME_OFFSET",
]
