"""Public curve stitching API."""

from .stiching import (
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

for _export in (
    CurveSegment,
    MultiStitchResult,
    PairFit,
    SASCurve,
    SelectedSegment,
):
    _export.__module__ = __name__

for _export in (
    infer_ids_from_filename,
    load_segment_from_nexus,
    load_segments,
    rebase_result_to_reference,
    save_outputs,
    stitch_segments_greedy,
):
    _export.__module__ = __name__

__all__ = [
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
]
