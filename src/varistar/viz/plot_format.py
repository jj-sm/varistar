"""
Functions for formatting RA and DEC values for plotting on matplotlib axes. These functions convert RA and DEC values from degrees to a more human-readable format, suitable for astronomical plots. The RA is formatted in hours and minutes, while the DEC is formatted in degrees and arcminutes, with appropriate symbols and signs.
"""

from astropy.coordinates import Angle
import astropy.units as u

# RA and DEC formatting functions for matplotlib axes. The _position argument is required by the FuncFormatter but is not used here.
def format_ra(value: float, _position: int | None = None) -> str:
    """
    Formats the RA value in degrees to a string format suitable for plotting on a matplotlib axis. The output is in the form of hours and minutes, with appropriate symbols for hours and minutes.
    """
    # Formats to: 20h 20'
    return Angle(value * u.deg).to_string(unit=u.hourangle, sep=('$^\\text{h}$', '$^\\text{m}$'), precision=0, pad=True, fields=2) # type: ignore

def format_dec(value: float, _position: int | None = None) -> str:
    """
    Formats the DEC value in degrees to a string format suitable for plotting on a matplotlib axis. The output is in the form of degrees and arcminutes, with appropriate symbols for degrees and arcminutes, and includes a sign for positive and negative values.
    """
    # Formats to: +19° 20'
    return Angle(value * u.deg).to_string(unit=u.deg, sep=('$^\\text{°}$', '$^\\text{m}$'), precision=0, pad=True, alwayssign=True, fields=2) # type: ignore
