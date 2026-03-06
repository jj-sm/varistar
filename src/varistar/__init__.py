"""
varistar
========
Variable star photometric analysis toolkit.

Quick start
-----------
>>> from varistar import TimeSeries, LightCurve
>>> ts = TimeSeries(magnitude="mag I", time_scale="HJD")
>>> ts.load_data_from_file("my_star.dat")
>>> lc = LightCurve(ts)
>>> lc.find_best_period()
>>> lc.plot_best()
"""

__version__ = "0.1.0"

from varistar.timeseries import TimeSeries
from varistar.lightcurve import LightCurve
from varistar.groups import TestGroup

__all__ = ["TimeSeries", "LightCurve", "TestGroup"]
