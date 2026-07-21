from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.special import erf


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SASCurve:
    """One reduced SAS configuration.

    Parameters
    ----------
    q, i, di, dq:
        One-dimensional arrays containing Q, intensity, intensity uncertainty,
        and Q resolution.
    name:
        Optional configuration name used for provenance in stitched outputs.

    Notes
    -----
    ``dq`` is assumed to be the 1-sigma width of a Gaussian Q-resolution
    function, expressed in the same unit as ``q``.
    """

    q: FloatArray
    i: FloatArray
    di: FloatArray
    dq: FloatArray
    name: str = ""
    config_id: str = ""

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=float)
        i = np.asarray(self.i, dtype=float)
        di = np.asarray(self.di, dtype=float)
        dq = np.asarray(self.dq, dtype=float)

        if not (q.ndim == i.ndim == di.ndim == dq.ndim == 1):
            raise ValueError("q, i, di and dq must be one-dimensional.")
        if not (q.size == i.size == di.size == dq.size):
            raise ValueError("q, i, di and dq must have identical lengths.")
        if q.size < 3:
            raise ValueError("A curve must contain at least three points.")
        if not np.all(np.isfinite(q)) or np.any(np.diff(q) <= 0):
            raise ValueError("q must be finite and strictly increasing.")
        if not np.all(np.isfinite(i)):
            raise ValueError("i must be finite.")
        if not np.all(np.isfinite(di)) or np.any(di <= 0):
            raise ValueError("di must be finite and strictly positive.")
        if not np.all(np.isfinite(dq)) or np.any(dq < 0):
            raise ValueError("dq must be finite and non-negative.")

        object.__setattr__(self, "q", q)
        object.__setattr__(self, "i", i)
        object.__setattr__(self, "di", di)
        object.__setattr__(self, "dq", dq)

    def __mul__(self, factor: float) -> "SASCurve":
        """Return a copy with I and dI multiplied by ``factor``."""
        return SASCurve(self.q, self.i * factor, self.di * factor, self.dq, self.name + "x%.1f" % factor, self.config_id)
    
    def __add__(self, factor: float) -> "SASCurve":
        return SASCurve(self.q, self.i + factor, self.di, self.dq, self.name + "+%.1f" % factor, self.config_id)
    

    @classmethod
    def from_array(cls, data: ArrayLike, *, name: str = "", config_id: str = "") -> "SASCurve":
        """Build a curve from a four-column array: Q, I, dI, dQ."""
        array = np.asarray(data, dtype=float)
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError("data must have shape (n, 4): Q, I, dI, dQ.")
        return cls(
            q=array[:, 0],
            i=array[:, 1],
            di=array[:, 2],
            dq=array[:, 3],
            name=name,
            config_id=config_id
        )

    def to_array(self) -> FloatArray:
        """Return a four-column array: Q, I, dI, dQ."""
        return np.column_stack((self.q, self.i, self.di, self.dq))

    def scaled(self, factor: float, *, name: str | None = None) -> "SASCurve":
        """Return a copy with I and dI multiplied by ``factor``."""
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("factor must be finite and strictly positive.")
        return SASCurve(
            q=self.q,
            i=self.i * factor,
            di=self.di * factor,
            dq=self.dq,
            name=self.name if name is None else name,
            config_id=self.config_id
        )


@dataclass(frozen=True)
class StitchedCurve:
    """Final stitched data with source labels."""

    q: FloatArray
    i: FloatArray
    di: FloatArray
    dq: FloatArray
    source: NDArray[np.object_]

    def to_array(self) -> FloatArray:
        """Return only the four numerical columns: Q, I, dI, dQ."""
        return np.column_stack((self.q, self.i, self.di, self.dq))


@dataclass(frozen=True)
class DegradedCurve:
    """Curve evaluated after additional Gaussian resolution smearing."""

    q: FloatArray
    i: FloatArray
    di: FloatArray
    dq: FloatArray
    valid: BoolArray
    coverage: FloatArray

    def __mul__(self, factor: float) -> "SASCurve":
        """Return a copy with I and dI multiplied by ``factor``."""
        return DegradedCurve(coverage=self.coverage, q=self.q, i=self.i * factor, di=self.di * factor, dq=self.dq, valid=self.valid)


@dataclass(frozen=True)
class CommonResolutionOverlap:
    """Two curves represented on one Q grid and at one Q resolution."""

    q: FloatArray
    low_i: FloatArray
    low_di: FloatArray
    high_i: FloatArray
    high_di: FloatArray
    target_dq: FloatArray
    valid: BoolArray
    degraded_curve: str
    geometric_overlap: tuple[float, float]


@dataclass(frozen=True)
class ScaleFit:
    """Result of the robust multiplicative scale fit."""

    factor: float
    factor_error: float
    chi2_red: float
    q: FloatArray
    z: FloatArray
    used: BoolArray
    n_points: int


@dataclass(frozen=True)
class LocalScaleProfile:
    """Sliding-window scale-factor diagnostic."""

    q: FloatArray
    factor: FloatArray
    factor_error: FloatArray
    n_points: FloatArray
    stability: float


@dataclass(frozen=True)
class StitchResult:
    """Result of stitching a low-Q and a high-Q configuration."""

    curve: StitchedCurve
    scale_factor: float
    scale_error: float
    chi2_red: float
    scale_stability: float
    fit_range: tuple[float, float]
    retained_overlap: tuple[float, float]
    degraded_curve: str
    n_fit_points: int
    comparison: CommonResolutionOverlap
    scale_fit: ScaleFit
    local_scale_profile: LocalScaleProfile


@dataclass(frozen=True)
class MultiStitchResult:
    """Result of sequentially stitching configurations ordered by increasing Q."""

    curve: StitchedCurve
    scaled_curves: tuple[SASCurve, ...]
    cumulative_factors: FloatArray
    pair_results: tuple[StitchResult, ...]


def overlap_bounds(
    first: SASCurve,
    second: SASCurve,
    *,
    q_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return the geometric Q overlap, optionally restricted by ``q_range``."""
    q_min = max(first.q[0], second.q[0])
    q_max = min(first.q[-1], second.q[-1])

    if q_range is not None:
        if q_range[0] >= q_range[1]:
            raise ValueError("q_range must satisfy q_min < q_max.")
        q_min = max(q_min, q_range[0])
        q_max = min(q_max, q_range[1])

    if q_min >= q_max:
        raise ValueError("The two curves have no usable Q overlap.")
    return float(q_min), float(q_max)


def _trapezoid_weights(q: FloatArray) -> FloatArray:
    """Integration weights for a non-uniform one-dimensional grid."""
    weights = np.empty_like(q)
    weights[0] = 0.5 * (q[1] - q[0])
    weights[-1] = 0.5 * (q[-1] - q[-2])
    weights[1:-1] = 0.5 * (q[2:] - q[:-2])
    return weights


def _linear_interpolation_row(q: FloatArray, q0: float) -> FloatArray:
    """Return linear interpolation weights at one Q value."""
    row = np.zeros(q.size, dtype=float)
    index = int(np.searchsorted(q, q0))

    if index == 0:
        if np.isclose(q0, q[0]):
            row[0] = 1.0
        return row
    if index == q.size:
        if np.isclose(q0, q[-1]):
            row[-1] = 1.0
        return row

    q_left = q[index - 1]
    q_right = q[index]
    fraction = (q0 - q_left) / (q_right - q_left)
    row[index - 1] = 1.0 - fraction
    row[index] = fraction
    return row


def degrade_to_resolution(
    source: SASCurve,
    *,
    q_eval: ArrayLike,
    target_dq: ArrayLike,
    min_coverage: float = 0.995,
    resolution_rtol: float = 0.02,
) -> DegradedCurve:
    """Degrade a measured curve to a broader Gaussian Q resolution.

    The source curve is convolved with an additional Gaussian kernel

        sigma_add(Q) = sqrt(target_dq(Q)^2 - source_dq(Q)^2).

    Only marginal intensity uncertainties are propagated:

        dI_out[j]^2 = sum_i W[j, i]^2 dI_in[i]^2.

    Correlations created by the convolution are intentionally ignored in this
    lightweight implementation.
    """
    q_eval = np.asarray(q_eval, dtype=float)
    target_dq = np.asarray(target_dq, dtype=float)

    if q_eval.ndim != 1 or target_dq.shape != q_eval.shape:
        raise ValueError("q_eval and target_dq must be one-dimensional and aligned.")
    if np.any(np.diff(q_eval) <= 0):
        raise ValueError("q_eval must be strictly increasing.")
    if np.any(~np.isfinite(target_dq)) or np.any(target_dq < 0):
        raise ValueError("target_dq must be finite and non-negative.")
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must lie in (0, 1].")

    source_dq = PchipInterpolator(
        source.q,
        source.dq,
        extrapolate=False,
    )(q_eval)

    finite_resolution = np.isfinite(source_dq)
    tolerance = resolution_rtol * np.maximum(target_dq, np.finfo(float).eps)
    correct_order = source_dq <= target_dq + tolerance

    sigma_add = np.sqrt(np.maximum(target_dq**2 - source_dq**2, 0.0))
    matrix = np.zeros((q_eval.size, source.q.size), dtype=float)
    coverage = np.zeros(q_eval.size, dtype=float)

    quadrature = _trapezoid_weights(source.q)
    local_step = PchipInterpolator(
        source.q,
        np.gradient(source.q),#(''),
        extrapolate=False,
    )(q_eval)

    sqrt_two = np.sqrt(2.0)

    for row_index, (q0, sigma, step) in enumerate(
        zip(q_eval, sigma_add, local_step, strict=True)
    ):
        if not (finite_resolution[row_index] and correct_order[row_index]):
            continue

        # If the extra smearing is narrower than the sampled Q spacing, the
        # kernel cannot be resolved reliably. Local interpolation is safer.
        if sigma <= 0.5 * step:
            matrix[row_index] = _linear_interpolation_row(source.q, q0)
            coverage[row_index] = 1.0
            continue

        coverage[row_index] = 0.5 * (
            erf((source.q[-1] - q0) / (sqrt_two * sigma))
            - erf((source.q[0] - q0) / (sqrt_two * sigma))
        )

        gaussian = np.exp(-0.5 * ((source.q - q0) / sigma) ** 2)
        row = gaussian * quadrature
        normalization = row.sum()
        if normalization > 0:
            matrix[row_index] = row / normalization

    valid = (
        finite_resolution
        & correct_order
        & (coverage >= min_coverage)
        & (matrix.sum(axis=1) > 0)
    )

    intensity = matrix @ source.i
    variance = (matrix * matrix) @ (source.di * source.di)
    uncertainty = np.sqrt(np.clip(variance, 0.0, None))

    intensity = intensity.astype(float, copy=True)
    uncertainty = uncertainty.astype(float, copy=True)
    intensity[~valid] = np.nan
    uncertainty[~valid] = np.nan

    return DegradedCurve(
        q=q_eval,
        i=intensity,
        di=uncertainty,
        dq=target_dq,
        valid=valid,
        coverage=coverage,
    )


def common_resolution_overlap(
    low_q_curve: SASCurve,
    high_q_curve: SASCurve,
    *,
    q_range: tuple[float, float] | None = None,
    resolution_metric: str = "relative",
    min_coverage: float = 0.995,
) -> CommonResolutionOverlap:
    """Represent two overlapping configurations at the poorer Q resolution.

    The comparison grid is the native Q grid of the poorer-resolved
    configuration. Only the better-resolved configuration is additionally
    smeared.
    """
    q_min, q_max = overlap_bounds(
        low_q_curve,
        high_q_curve,
        q_range=q_range,
    )

    low_mask = (low_q_curve.q >= q_min) & (low_q_curve.q <= q_max)
    high_mask = (high_q_curve.q >= q_min) & (high_q_curve.q <= q_max)

    if resolution_metric == "relative":
        low_score = np.median(low_q_curve.dq[low_mask] / low_q_curve.q[low_mask])
        high_score = np.median(high_q_curve.dq[high_mask] / high_q_curve.q[high_mask])
    elif resolution_metric == "absolute":
        low_score = np.median(low_q_curve.dq[low_mask])
        high_score = np.median(high_q_curve.dq[high_mask])
    else:
        raise ValueError("resolution_metric must be 'relative' or 'absolute'.")

    if low_score >= high_score:
        q = low_q_curve.q[low_mask]
        target_dq = low_q_curve.dq[low_mask]
        degraded = degrade_to_resolution(
            high_q_curve,
            q_eval=q,
            target_dq=target_dq,
            min_coverage=min_coverage,
        )
        low_i = low_q_curve.i[low_mask]
        low_di = low_q_curve.di[low_mask]
        high_i = degraded.i
        high_di = degraded.di
        valid = degraded.valid
        degraded_curve = "high_q"
    else:
        q = high_q_curve.q[high_mask]
        target_dq = high_q_curve.dq[high_mask]
        degraded = degrade_to_resolution(
            low_q_curve,
            q_eval=q,
            target_dq=target_dq,
            min_coverage=min_coverage,
        )
        low_i = degraded.i
        low_di = degraded.di
        high_i = high_q_curve.i[high_mask]
        high_di = high_q_curve.di[high_mask]
        valid = degraded.valid
        degraded_curve = "low_q"

    valid = (
        valid
        & np.isfinite(low_i)
        & np.isfinite(low_di)
        & np.isfinite(high_i)
        & np.isfinite(high_di)
        & (low_di > 0)
        & (high_di > 0)
    )

    return CommonResolutionOverlap(
        q=q,
        low_i=low_i,
        low_di=low_di,
        high_i=high_i,
        high_di=high_di,
        target_dq=target_dq,
        valid=valid,
        degraded_curve=degraded_curve,
        geometric_overlap=(q_min, q_max),
    )


def _initial_log_scale(low_i: FloatArray, high_i: FloatArray) -> float:
    """Return a robust initial estimate of log(scale)."""
    positive = (low_i > 0) & (high_i > 0)
    if positive.any():
        return float(np.median(np.log(low_i[positive] / high_i[positive])))

    denominator = float(np.dot(high_i, high_i))
    if denominator <= 0:
        raise ValueError("Cannot determine an initial scale factor.")

    initial_scale = abs(float(np.dot(low_i, high_i)) / denominator)
    return float(np.log(max(initial_scale, np.finfo(float).tiny)))


def _fit_scale_arrays(
    low_i: FloatArray,
    low_di: FloatArray,
    high_i: FloatArray,
    high_di: FloatArray,
    *,
    loss: str,
    f_scale: float,
    initial_log_scale: float | None = None,
) -> tuple[float, FloatArray]:
    """Fit one positive multiplicative factor to aligned arrays."""
    if initial_log_scale is None:
        initial_log_scale = _initial_log_scale(low_i, high_i)

    def residual(log_scale: FloatArray) -> FloatArray:
        scale = np.exp(log_scale[0])
        sigma = np.sqrt(low_di**2 + (scale * high_di) ** 2)
        return (low_i - scale * high_i) / sigma

    optimization = least_squares(
        residual,
        x0=np.array([initial_log_scale], dtype=float),
        loss=loss,
        f_scale=f_scale,
    )
    scale = float(np.exp(optimization.x[0]))
    return scale, residual(optimization.x)


def _approximate_scale_error(
    scale: float,
    low_di: FloatArray,
    high_i: FloatArray,
    high_di: FloatArray,
    *,
    chi2_red: float,
) -> float:
    """Approximate 1-sigma scale uncertainty, ignoring point correlations."""
    sigma = np.sqrt(low_di**2 + (scale * high_di) ** 2)
    jacobian = -scale * high_i / sigma
    information = float(np.dot(jacobian, jacobian))
    if information <= 0:
        return float("nan")
    log_scale_error = np.sqrt(max(chi2_red, 0.0) / information)
    return float(scale * log_scale_error)


def fit_scale_factor(
    overlap: CommonResolutionOverlap,
    *,
    loss: str = "soft_l1",
    f_scale: float = 1.5,
    outlier_sigma: float = 4.0,
) -> ScaleFit:
    """Fit the positive factor multiplying the high-Q curve.

    The fitted relation is

        I_low(Q) = scale_factor * I_high(Q)

    after both curves have been represented at the same Q resolution.

    Notes
    -----
    The propagated ``dI`` values are used as if points were independent.
    Correlations created by the smearing operation are intentionally ignored.
    """
    valid = overlap.valid.copy()
    if valid.sum() < 3:
        raise ValueError("At least three valid overlap points are required.")

    low_i = overlap.low_i[valid]
    low_di = overlap.low_di[valid]
    high_i = overlap.high_i[valid]
    high_di = overlap.high_di[valid]

    first_scale, first_z = _fit_scale_arrays(
        low_i,
        low_di,
        high_i,
        high_di,
        loss=loss,
        f_scale=f_scale,
    )
    accepted = np.abs(first_z) <= outlier_sigma
    if accepted.sum() < 3:
        raise ValueError("Fewer than three points remain after outlier rejection.")

    # Refit only the accepted points so that rejected points cannot bias the
    # final factor.
    scale, z_used = _fit_scale_arrays(
        low_i[accepted],
        low_di[accepted],
        high_i[accepted],
        high_di[accepted],
        loss=loss,
        f_scale=f_scale,
        initial_log_scale=np.log(first_scale),
    )

    sigma_all = np.sqrt(low_di**2 + (scale * high_di) ** 2)
    z_all = (low_i - scale * high_i) / sigma_all

    degrees_of_freedom = max(int(accepted.sum()) - 1, 1)
    chi2_red = float(np.sum(z_all[accepted] ** 2) / degrees_of_freedom)
    scale_error = _approximate_scale_error(
        scale,
        low_di[accepted],
        high_i[accepted],
        high_di[accepted],
        chi2_red=chi2_red,
    )

    z = np.full(overlap.q.shape, np.nan, dtype=float)
    used = np.zeros(overlap.q.shape, dtype=bool)
    valid_indices = np.flatnonzero(valid)
    z[valid_indices] = z_all
    used[valid_indices[accepted]] = True

    return ScaleFit(
        factor=scale,
        factor_error=scale_error,
        chi2_red=chi2_red,
        q=overlap.q,
        z=z,
        used=used,
        n_points=int(accepted.sum()),
    )


def local_scale_profile(
    overlap: CommonResolutionOverlap,
    scale_fit: ScaleFit,
    *,
    window_points: int = 9,
    min_points: int = 5,
    loss: str = "soft_l1",
    f_scale: float = 1.5,
) -> LocalScaleProfile:
    """Estimate the scale factor in sliding windows across the accepted overlap.

    ``stability`` is the robust relative spread

        1.4826 * MAD(local_scale) / global_scale.

    It is a practical diagnostic, not a formal statistical uncertainty.
    """
    accepted_indices = np.flatnonzero(scale_fit.used)
    if accepted_indices.size < min_points:
        return LocalScaleProfile(
            q=np.array([], dtype=float),
            factor=np.array([], dtype=float),
            factor_error=np.array([], dtype=float),
            n_points=np.array([], dtype=float),
            stability=float("nan"),
        )

    window_points = max(int(window_points), min_points)
    window_points = min(window_points, int(accepted_indices.size))
    if window_points % 2 == 0 and window_points > min_points:
        window_points -= 1

    q_values: list[float] = []
    factors: list[float] = []
    errors: list[float] = []
    counts: list[float] = []

    for start in range(0, accepted_indices.size - window_points + 1):
        index = accepted_indices[start : start + window_points]

        scale, z = _fit_scale_arrays(
            overlap.low_i[index],
            overlap.low_di[index],
            overlap.high_i[index],
            overlap.high_di[index],
            loss=loss,
            f_scale=f_scale,
            initial_log_scale=np.log(scale_fit.factor),
        )
        dof = max(index.size - 1, 1)
        chi2_red = float(np.sum(z**2) / dof)
        error = _approximate_scale_error(
            scale,
            overlap.low_di[index],
            overlap.high_i[index],
            overlap.high_di[index],
            chi2_red=chi2_red,
        )

        q_values.append(float(np.exp(np.mean(np.log(overlap.q[index])))))
        factors.append(scale)
        errors.append(error)
        counts.append(float(index.size))

    factor_array = np.asarray(factors, dtype=float)
    median_factor = float(np.median(factor_array))
    mad = float(np.median(np.abs(factor_array - median_factor)))
    stability = 1.4826 * mad / scale_fit.factor if scale_fit.factor > 0 else np.nan

    return LocalScaleProfile(
        q=np.asarray(q_values, dtype=float),
        factor=factor_array,
        factor_error=np.asarray(errors, dtype=float),
        n_points=np.asarray(counts, dtype=float),
        stability=float(stability),
    )


def choose_retained_overlap(
    overlap: CommonResolutionOverlap,
    scale_fit: ScaleFit,
    *,
    fraction: float = 0.20,
    min_points: int = 3,
) -> tuple[float, float]:
    """Choose a small central overlap within the accepted fit region."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")

    q = overlap.q[scale_fit.used]
    if q.size < 2 * min_points:
        raise ValueError("Not enough accepted points to choose a retained overlap.")

    log_q = np.log(q)
    center = 0.5 * (log_q.min() + log_q.max())
    half_width = 0.5 * fraction * (log_q.max() - log_q.min())

    requested_min = np.exp(center - half_width)
    requested_max = np.exp(center + half_width)

    left = q[q <= requested_min]
    right = q[q >= requested_max]
    keep_min = left[-1] if left.size else q[0]
    keep_max = right[0] if right.size else q[-1]

    if keep_min >= keep_max:
        center_index = q.size // 2
        keep_min = q[max(0, center_index - min_points)]
        keep_max = q[min(q.size - 1, center_index + min_points)]

    return float(keep_min), float(keep_max)


def _transition_weights(
    q: FloatArray,
    *,
    q_min: float,
    q_max: float,
) -> FloatArray:
    """Return monotonic blend weights spanning the overlap interval."""
    if q_min <= 0.0 or q_max <= 0.0:
        weights = (q - q_min) / (q_max - q_min)
    else:
        log_q = np.log(q)
        weights = (log_q - np.log(q_min)) / (np.log(q_max) - np.log(q_min))
    return np.clip(weights, 0.0, 1.0)


def _interpolate_curve_components(
    curve: SASCurve,
    q_eval: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate one curve on an overlap grid."""
    intensity = PchipInterpolator(curve.q, curve.i, extrapolate=False)(q_eval)
    uncertainty = np.interp(q_eval, curve.q, curve.di)
    resolution = np.interp(q_eval, curve.q, curve.dq)
    return (
        np.asarray(intensity, dtype=float),
        np.asarray(uncertainty, dtype=float),
        np.asarray(resolution, dtype=float),
    )


def _blend_stitched_curve(
    low_q_curve: SASCurve,
    high_q_curve: SASCurve,
    *,
    keep_min: float,
    keep_max: float,
) -> StitchedCurve:
    """Blend the overlap smoothly instead of keeping both raw point sets."""
    low_name = low_q_curve.name or "low_q"
    high_name = high_q_curve.name or "high_q"

    low_mask = low_q_curve.q < keep_min
    high_mask = high_q_curve.q > keep_max
    overlap_low = low_q_curve.q[(low_q_curve.q >= keep_min) & (low_q_curve.q <= keep_max)]
    overlap_high = high_q_curve.q[(high_q_curve.q >= keep_min) & (high_q_curve.q <= keep_max)]
    overlap_q = np.unique(np.concatenate((overlap_low, overlap_high)))

    q_parts: list[FloatArray] = [low_q_curve.q[low_mask]]
    i_parts: list[FloatArray] = [low_q_curve.i[low_mask]]
    di_parts: list[FloatArray] = [low_q_curve.di[low_mask]]
    dq_parts: list[FloatArray] = [low_q_curve.dq[low_mask]]
    source_parts: list[NDArray[np.object_]] = [
        np.full(low_mask.sum(), low_name, dtype=object)
    ]

    if overlap_q.size:
        low_i, low_di, low_dq = _interpolate_curve_components(low_q_curve, overlap_q)
        high_i, high_di, high_dq = _interpolate_curve_components(high_q_curve, overlap_q)
        weights = _transition_weights(overlap_q, q_min=keep_min, q_max=keep_max)

        overlap_i = (1.0 - weights) * low_i + weights * high_i
        overlap_di = np.sqrt(((1.0 - weights) * low_di) ** 2 + (weights * high_di) ** 2)
        overlap_dq = (1.0 - weights) * low_dq + weights * high_dq

        q_parts.append(overlap_q)
        i_parts.append(overlap_i)
        di_parts.append(overlap_di)
        dq_parts.append(overlap_dq)
        source_parts.append(
            np.full(overlap_q.size, f"blend:{low_name}|{high_name}", dtype=object)
        )

    q_parts.append(high_q_curve.q[high_mask])
    i_parts.append(high_q_curve.i[high_mask])
    di_parts.append(high_q_curve.di[high_mask])
    dq_parts.append(high_q_curve.dq[high_mask])
    source_parts.append(np.full(high_mask.sum(), high_name, dtype=object))

    q = np.concatenate(q_parts)
    i = np.concatenate(i_parts)
    di = np.concatenate(di_parts)
    dq = np.concatenate(dq_parts)
    source = np.concatenate(source_parts)

    return StitchedCurve(
        q=q,
        i=i,
        di=di,
        dq=dq,
        source=source,
    )


def stitch_pair(
    low_q_curve: SASCurve,
    high_q_curve: SASCurve,
    *,
    fit_range: tuple[float, float] | None = None,
    keep_range: tuple[float, float] | None = None,
    keep_fraction: float = 0.20,
    overlap_mode: Literal["keep_both", "blend"] = "keep_both",
    min_coverage: float = 0.995,
    loss: str = "soft_l1",
    f_scale: float = 1.5,
    outlier_sigma: float = 4.0,
    local_window_points: int = 9,
) -> StitchResult:
    """Scale and stitch two configurations while retaining a small overlap.

    The high-Q configuration is scaled to the low-Q configuration.
    Resolution degradation is used only to estimate the scale factor; original
    data and original dQ values are preserved in the final output.

    ``overlap_mode="keep_both"`` preserves both native point sets in the
    retained overlap. ``overlap_mode="blend"`` replaces that overlap with one
    smoothly blended transition curve.
    """
    if overlap_mode not in {"keep_both", "blend"}:
        raise ValueError("overlap_mode must be 'keep_both' or 'blend'.")

    comparison = common_resolution_overlap(
        low_q_curve,
        high_q_curve,
        q_range=fit_range,
        min_coverage=min_coverage,
    )
    scale_fit = fit_scale_factor(
        comparison,
        loss=loss,
        f_scale=f_scale,
        outlier_sigma=outlier_sigma,
    )
    profile = local_scale_profile(
        comparison,
        scale_fit,
        window_points=local_window_points,
        loss=loss,
        f_scale=f_scale,
    )

    used_q = comparison.q[scale_fit.used]
    actual_fit_range = (float(used_q.min()), float(used_q.max()))

    if keep_range is None:
        keep_range = choose_retained_overlap(
            comparison,
            scale_fit,
            fraction=keep_fraction,
        )

    keep_min, keep_max = keep_range
    if not (
        actual_fit_range[0] <= keep_min < keep_max <= actual_fit_range[1]
    ):
        raise ValueError(
            "keep_range must lie inside the accepted common-resolution fit range."
        )

    scaled_high = high_q_curve.scaled(scale_fit.factor)

    if overlap_mode == "keep_both":
        low_mask = low_q_curve.q <= keep_max
        high_mask = scaled_high.q >= keep_min

        q = np.concatenate((low_q_curve.q[low_mask], scaled_high.q[high_mask]))
        i = np.concatenate((low_q_curve.i[low_mask], scaled_high.i[high_mask]))
        di = np.concatenate((low_q_curve.di[low_mask], scaled_high.di[high_mask]))
        dq = np.concatenate((low_q_curve.dq[low_mask], scaled_high.dq[high_mask]))

        low_name = low_q_curve.name or "low_q"
        high_name = high_q_curve.name or "high_q"
        source = np.concatenate(
            (
                np.full(low_mask.sum(), low_name, dtype=object),
                np.full(high_mask.sum(), high_name, dtype=object),
            )
        )

        order = np.argsort(q, kind="stable")
        stitched = StitchedCurve(
            q=q[order],
            i=i[order],
            di=di[order],
            dq=dq[order],
            source=source[order],
        )
    else:
        stitched = _blend_stitched_curve(
            low_q_curve,
            scaled_high,
            keep_min=keep_min,
            keep_max=keep_max,
        )

    return StitchResult(
        curve=stitched,
        scale_factor=scale_fit.factor,
        scale_error=scale_fit.factor_error,
        chi2_red=scale_fit.chi2_red,
        scale_stability=profile.stability,
        fit_range=actual_fit_range,
        retained_overlap=(float(keep_min), float(keep_max)),
        degraded_curve=comparison.degraded_curve,
        n_fit_points=scale_fit.n_points,
        comparison=comparison,
        scale_fit=scale_fit,
        local_scale_profile=profile,
    )


def stitch_many(
    curves: Sequence[SASCurve],
    *,
    fit_ranges: Sequence[tuple[float, float] | None] | None = None,
    keep_ranges: Sequence[tuple[float, float] | None] | None = None,
    keep_fraction: float = 0.20,
    overlap_mode: Literal["keep_both", "blend"] = "keep_both",
    min_coverage: float = 0.995,
    loss: str = "soft_l1",
    f_scale: float = 1.5,
    outlier_sigma: float = 4.0,
    local_window_points: int = 9,
) -> MultiStitchResult:
    """Sequentially stitch configurations ordered from low Q to high Q.

    Each new configuration is scaled directly to the already-scaled previous
    configuration. Returned factors therefore put each original configuration
    on the scale of the first one.
    """
    curves = tuple(curves)
    if len(curves) < 2:
        raise ValueError("At least two curves are required.")

    n_pairs = len(curves) - 1
    if fit_ranges is None:
        fit_ranges = (None,) * n_pairs
    if keep_ranges is None:
        keep_ranges = (None,) * n_pairs

    if len(fit_ranges) != n_pairs or len(keep_ranges) != n_pairs:
        raise ValueError("fit_ranges and keep_ranges must contain n_curves - 1 items.")

    scaled_curves: list[SASCurve] = [curves[0]]
    cumulative_factors = [1.0]
    pair_results: list[StitchResult] = []

    for pair_index in range(n_pairs):
        previous = scaled_curves[-1]
        current = curves[pair_index + 1]

        result = stitch_pair(
            previous,
            current,
            fit_range=fit_ranges[pair_index],
            keep_range=keep_ranges[pair_index],
            keep_fraction=keep_fraction,
            overlap_mode=overlap_mode,
            min_coverage=min_coverage,
            loss=loss,
            f_scale=f_scale,
            outlier_sigma=outlier_sigma,
            local_window_points=local_window_points,
        )

        scaled_current = current.scaled(result.scale_factor)
        scaled_curves.append(scaled_current)
        cumulative_factors.append(result.scale_factor)
        pair_results.append(result)

    q_parts: list[FloatArray] = []
    i_parts: list[FloatArray] = []
    di_parts: list[FloatArray] = []
    dq_parts: list[FloatArray] = []
    source_parts: list[NDArray[np.object_]] = []

    for curve_index, curve in enumerate(scaled_curves):
        q_min = -np.inf
        q_max = np.inf
        if curve_index > 0:
            q_min = pair_results[curve_index - 1].retained_overlap[0]
        if curve_index < len(scaled_curves) - 1:
            q_max = pair_results[curve_index].retained_overlap[1]

        mask = (curve.q >= q_min) & (curve.q <= q_max)
        q_parts.append(curve.q[mask])
        i_parts.append(curve.i[mask])
        di_parts.append(curve.di[mask])
        dq_parts.append(curve.dq[mask])
        source_parts.append(
            np.full(mask.sum(), curve.name or f"config_{curve_index}", dtype=object)
        )

    q = np.concatenate(q_parts)
    i = np.concatenate(i_parts)
    di = np.concatenate(di_parts)
    dq = np.concatenate(dq_parts)
    source = np.concatenate(source_parts)
    order = np.argsort(q, kind="stable")

    stitched = StitchedCurve(
        q=q[order],
        i=i[order],
        di=di[order],
        dq=dq[order],
        source=source[order],
    )

    return MultiStitchResult(
        curve=stitched,
        scaled_curves=tuple(scaled_curves),
        cumulative_factors=np.asarray(cumulative_factors, dtype=float),
        pair_results=tuple(pair_results),
    )
