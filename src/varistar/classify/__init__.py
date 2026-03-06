"""varistar.classify — Star-type classification helpers."""

from varistar.classify.eb_detector import (
    score_eb,
    classify_eb_type,
    detect_secondary_eclipse,
)

__all__ = ["score_eb", "classify_eb_type", "detect_secondary_eclipse"]
