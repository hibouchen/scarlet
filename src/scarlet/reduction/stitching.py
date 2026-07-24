"""Public curve stitching API.

This module re-exports the maintained stitching implementation so callers can
use the correctly spelled ``scarlet.reduction.stitching`` path.
"""

from .stitchingbis import (
    CommonResolutionOverlap,
    DegradedCurve,
    MultiStitchResult,
    SASCurve,
    ScaleFit,
    StitchedCurve,
    StitchResult,
    choose_retained_overlap,
    common_resolution_overlap,
    degrade_to_resolution,
    fit_scale_factor,
    overlap_bounds,
    stitch_many,
    stitch_pair,
)

for _export in (
    CommonResolutionOverlap,
    DegradedCurve,
    MultiStitchResult,
    SASCurve,
    ScaleFit,
    StitchedCurve,
    StitchResult,
):
    _export.__module__ = __name__

for _export in (
    choose_retained_overlap,
    common_resolution_overlap,
    degrade_to_resolution,
    fit_scale_factor,
    overlap_bounds,
    stitch_many,
    stitch_pair,
):
    _export.__module__ = __name__

__all__ = [
    "CommonResolutionOverlap",
    "DegradedCurve",
    "MultiStitchResult",
    "SASCurve",
    "ScaleFit",
    "StitchedCurve",
    "StitchResult",
    "choose_retained_overlap",
    "common_resolution_overlap",
    "degrade_to_resolution",
    "fit_scale_factor",
    "overlap_bounds",
    "stitch_many",
    "stitch_pair",
]
