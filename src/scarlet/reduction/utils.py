from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _normalize_curve(
    q: Any,
    intensity: Any,
    *,
    intensity_error: Any | None = None,
    q_error: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Convert curve inputs to 1D float arrays and validate matching shapes."""
    q_array = np.asarray(q, dtype=np.float64)
    intensity_array = np.asarray(intensity, dtype=np.float64)
    if q_array.ndim != 1:
        raise ValueError(f"q must be a 1D array, got shape {q_array.shape}")
    if intensity_array.shape != q_array.shape:
        raise ValueError(f"intensity shape mismatch: expected {q_array.shape}, got {intensity_array.shape}")

    if intensity_error is not None:
        intensity_error_array = np.asarray(intensity_error, dtype=np.float64)
        if intensity_error_array.shape != q_array.shape:
            raise ValueError(
                f"intensity_error shape mismatch: expected {q_array.shape}, got {intensity_error_array.shape}"
            )
    else:
        intensity_error_array = None

    if q_error is not None:
        q_error_array = np.asarray(q_error, dtype=np.float64)
        if q_error_array.shape != q_array.shape:
            raise ValueError(f"q_error shape mismatch: expected {q_array.shape}, got {q_error_array.shape}")
    else:
        q_error_array = None

    return q_array, intensity_array, intensity_error_array, q_error_array


def crop_curve(
    q: Any,
    intensity: Any,
    *,
    q_min: float | None = None,
    q_max: float | None = None,
    intensity_error: Any | None = None,
    q_error: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Restrict a 1D I(q) curve to the inclusive range [q_min, q_max]."""
    q_array, intensity_array, intensity_error_array, q_error_array = _normalize_curve(
        q,
        intensity,
        intensity_error=intensity_error,
        q_error=q_error,
    )
    keep = np.ones(q_array.shape, dtype=bool)
    if q_min is not None:
        keep &= q_array >= float(q_min)
    if q_max is not None:
        keep &= q_array <= float(q_max)
    return (
        q_array[keep],
        intensity_array[keep],
        None if intensity_error_array is None else intensity_error_array[keep],
        None if q_error_array is None else q_error_array[keep],
    )


def apply_prefactor(
    q: Any,
    intensity: Any,
    prefactor: float | Any | Callable[[np.ndarray], Any],
    *,
    intensity_error: Any | None = None,
    q_error: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Multiply I(q) by a scalar, array, or q-dependent prefactor."""
    q_array, intensity_array, intensity_error_array, q_error_array = _normalize_curve(
        q,
        intensity,
        intensity_error=intensity_error,
        q_error=q_error,
    )
    if callable(prefactor):
        prefactor_values = np.asarray(prefactor(q_array), dtype=np.float64)
    else:
        prefactor_values = np.asarray(prefactor, dtype=np.float64)

    if prefactor_values.ndim == 0:
        prefactor_values = np.full(q_array.shape, float(prefactor_values), dtype=np.float64)
    if prefactor_values.shape != q_array.shape:
        raise ValueError(f"prefactor shape mismatch: expected {q_array.shape}, got {prefactor_values.shape}")

    scaled_intensity_error = None
    if intensity_error_array is not None:
        scaled_intensity_error = intensity_error_array * np.abs(prefactor_values)

    return (
        q_array.copy(),
        intensity_array * prefactor_values,
        scaled_intensity_error,
        None if q_error_array is None else q_error_array.copy(),
    )


def concatenate_curves(
    *curves: tuple[Any, Any] | tuple[Any, Any, Any | None] | tuple[Any, Any, Any | None, Any | None],
    sort: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Concatenate several 1D curves and optionally sort the result by q."""
    if not curves:
        raise ValueError("At least one curve is required for concatenation")

    normalized: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]] = []
    for curve in curves:
        if len(curve) == 2:
            q, intensity = curve
            intensity_error = None
            q_error = None
        elif len(curve) == 3:
            q, intensity, intensity_error = curve
            q_error = None
        elif len(curve) == 4:
            q, intensity, intensity_error, q_error = curve
        else:
            raise ValueError("Each curve must be a tuple of (q, intensity[, intensity_error[, q_error]])")
        normalized.append(
            _normalize_curve(
                q,
                intensity,
                intensity_error=intensity_error,
                q_error=q_error,
            )
        )

    q = np.concatenate([curve[0] for curve in normalized])
    intensity = np.concatenate([curve[1] for curve in normalized])

    if all(curve[2] is not None for curve in normalized):
        intensity_error = np.concatenate([curve[2] for curve in normalized if curve[2] is not None])
    else:
        intensity_error = None

    if all(curve[3] is not None for curve in normalized):
        q_error = np.concatenate([curve[3] for curve in normalized if curve[3] is not None])
    else:
        q_error = None

    if sort:
        order = np.argsort(q, kind="stable")
        q = q[order]
        intensity = intensity[order]
        if intensity_error is not None:
            intensity_error = intensity_error[order]
        if q_error is not None:
            q_error = q_error[order]

    return q, intensity, intensity_error, q_error


__all__ = [
    "apply_prefactor",
    "concatenate_curves",
    "crop_curve",
]
