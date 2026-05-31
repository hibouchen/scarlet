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
from .resolution import compute_q_resolution_circular
from .transmission import (
    compute_reference_transmissions,
    compute_transmission,
)

__all__ = [
    "correct_detector_dead_time",
    "AzimuthalAverageResult",
    "azimuthal_average",
    "compute_chi_map",
    "compute_q_norm_map",
    "compute_q_resolution_circular",
    "compute_qx_vector",
    "compute_qy_vector",
    "compute_theta_map",
    "normalize_by_monitor",
    "normalize_by_solid_angle",
    "subtract_scattering_references",
    "compute_reference_transmissions",
    "compute_transmission",
]
