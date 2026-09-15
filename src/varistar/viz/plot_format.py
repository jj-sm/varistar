"""Matplotlib axis formatters for RA/Dec tick labels.

Both formatters accept a ``_position`` argument for compatibility with
`matplotlib.ticker.FuncFormatter`; it is unused.
"""

from astropy.coordinates import Angle
import astropy.units as u


def format_ra(value: float, _position: int | None = None) -> str:
    """Format a right ascension value for a matplotlib tick label.

    Parameters
    ----------
    value : float
        Right ascension in degrees.
    _position : int, optional
        Tick position, unused; accepted for `FuncFormatter` compatibility.

    Returns
    -------
    str
        RA formatted as hours and minutes, e.g. ``"20h 20m"``.
    """
    return Angle(value * u.deg).to_string(unit=u.hourangle, sep=('$^\\text{h}$', '$^\\text{m}$'), precision=0, pad=True, fields=2)  # type: ignore


def format_dec(value: float, _position: int | None = None) -> str:
    """Format a declination value for a matplotlib tick label.

    Parameters
    ----------
    value : float
        Declination in degrees.
    _position : int, optional
        Tick position, unused; accepted for `FuncFormatter` compatibility.

    Returns
    -------
    str
        Dec formatted as signed degrees and arcminutes, e.g. ``"+19° 20m"``.
    """
    return Angle(value * u.deg).to_string(unit=u.deg, sep=('$^\\text{°}$', '$^\\text{m}$'), precision=0, pad=True, alwayssign=True, fields=2)  # type: ignore
