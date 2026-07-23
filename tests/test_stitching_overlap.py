import numpy as np

from scarlet.reduction.stitchingbis import SASCurve, stitch_pair


def _make_curve(
    q: np.ndarray,
    intensity: np.ndarray,
    *,
    dq_scale: float,
    name: str,
) -> SASCurve:
    return SASCurve(
        q=q,
        i=intensity,
        di=np.full(q.shape, 0.02, dtype=float),
        dq=dq_scale * q,
        name=name,
    )


def test_blend_overlap_labels_overlap_points_as_blended() -> None:
    low_q = np.geomspace(0.01, 0.08, 24)
    high_q = np.geomspace(0.04, 0.2, 26)
    model = lambda q: 1.0 / (1.0 + 15.0 * q)

    low = _make_curve(low_q, model(low_q), dq_scale=0.10, name="low")
    high = _make_curve(high_q, model(high_q), dq_scale=0.05, name="high")

    result = stitch_pair(
        low,
        high,
        fit_range=(0.045, 0.075),
        keep_fraction=0.3,
        overlap_mode="blend",
    )

    in_overlap = (
        (result.curve.q >= result.retained_overlap[0])
        & (result.curve.q <= result.retained_overlap[1])
    )
    assert in_overlap.any()
    assert np.all(result.curve.source[in_overlap] == "blend:low|high")
