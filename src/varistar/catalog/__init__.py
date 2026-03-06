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
"""

from varistar.catalog.ogle import load_dat, load_dat_directory
from varistar.catalog.generic import load_csv, from_arrays
from varistar.catalog.tess import load_fits, load_from_tic

__all__ = [
    "load_dat",
    "load_dat_directory",
    "load_csv",
    "from_arrays",
    "load_fits",
    "load_from_tic",
]
