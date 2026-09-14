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
from .stitching import (
    CurveSegment,
    MultiStitchResult,
    PairFit,
    SASCurve,
    SelectedSegment,
    infer_ids_from_filename,
    load_segment_from_nexus,
    load_segments,
    rebase_result_to_reference,
    save_outputs,
    stitch_segments_greedy,
)

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
    "compute_q_resolution_circular",
    "compute_q_uncertainty_map",
    "compute_qx_vector",
    "compute_qx_uncertainty_vector",
    "compute_qy_vector",
    "compute_qy_uncertainty_vector",
    "compute_theta_map",
    "crop_curve",
    "CurveSegment",
    "MultiStitchResult",
    "PairFit",
    "SASCurve",
    "SelectedSegment",
    "infer_ids_from_filename",
    "load_segment_from_nexus",
    "load_segments",
    "rebase_result_to_reference",
    "save_outputs",
    "stitch_segments_greedy",
    "subtract_scattering_references",
    "compute_transmission",
]
