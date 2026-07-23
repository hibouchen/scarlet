"""SCARLET reduction package."""

from .correction import (
    correct_detector_dead_time,
    normalize_by_monitor,
    normalize_by_solid_angle,
    subtract_scattering_references,
)
from .geometry import (
    compute_chi_map,
    compute_q_norm_map,
    compute_qx_vector,
    compute_qy_vector,
    compute_theta_map,
)
from .integration import AzimuthalAverageResult, azimuthal_average

from .resolution import (
    compute_beam_divergence,
    compute_q_resolution_circular,
    compute_q_uncertainty_map,
    compute_qx_uncertainty_vector,
    compute_qy_uncertainty_vector,
)
from .transmission import (
    compute_transmission,
)
from .utils import apply_prefactor, concatenate_curves, crop_curve

__all__ = [
    "correct_detector_dead_time",
    "AzimuthalAverageResult",
    "apply_prefactor",
    "azimuthal_average",
    "concatenate_curves",
    "compute_chi_map",
    "compute_beam_divergence",
    "compute_q_norm_map",
    "normalize_by_monitor",
    "normalize_by_solid_angle",
    "CommonResolutionOverlap",
    "compute_q_resolution_circular",
    "compute_q_uncertainty_map",
    "compute_qx_vector",
    "compute_qx_uncertainty_vector",
    "compute_qy_vector",
    "compute_qy_uncertainty_vector",
    "compute_theta_map",
    "crop_curve",
    "choose_retained_overlap",
    "common_resolution_overlap",
    "DegradedCurve",
    "degrade_to_resolution",
    "fit_scale_factor",
    "MultiStitchResult",
    "overlap_bounds",
    "SASCurve",
    "ScaleFit",
    "StitchedCurve",
    "StitchResult",
    "stitch_many",
    "stitch_pair",
    "subtract_scattering_references",
    "compute_transmission",
]
