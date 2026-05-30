from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#TODO: take care of error propagation for azimuthal averaging, and add tests for it


@dataclass(frozen=True)
class AzimuthalAverageResult:
    q: np.ndarray
    intensity: np.ndarray
    intensity_error: np.ndarray | None
    q_error: np.ndarray | None
    counts: np.ndarray


def azimuthal_average(
    image: Any,
    q_map: Any,
    *,
    mask: Any | None = None,
    intensity_error: Any | None = None,
    q_error: Any | None = None,
    n_bins: int = 200,
) -> AzimuthalAverageResult:
    """
    Compute a 1D azimuthal average I(q) from a detector image and a q-map.

    Non-finite pixels are ignored. If `mask` is provided, non-zero values are
    treated as masked pixels and excluded from the average.
    """
    image_array = np.asarray(image, dtype=np.float64)
    q_array = np.asarray(q_map, dtype=np.float64)
    if image_array.ndim != 2:
        raise ValueError(f"image must be a 2D array, got shape {image_array.shape}")
    if q_array.shape != image_array.shape:
        raise ValueError(f"q_map shape mismatch: expected {image_array.shape}, got {q_array.shape}")

    if intensity_error is not None:
        intensity_error_array = np.asarray(intensity_error, dtype=np.float64)
        if intensity_error_array.shape != image_array.shape:
            raise ValueError(
                f"intensity_error shape mismatch: expected {image_array.shape}, got {intensity_error_array.shape}"
            )
    else:
        intensity_error_array = None

    if q_error is not None:
        q_error_array = np.asarray(q_error, dtype=np.float64)
        if q_error_array.shape != image_array.shape:
            raise ValueError(f"q_error shape mismatch: expected {image_array.shape}, got {q_error_array.shape}")
    else:
        q_error_array = None

    if mask is not None:
        mask_array = np.asarray(mask)
        if mask_array.shape != image_array.shape:
            raise ValueError(f"mask shape mismatch: expected {image_array.shape}, got {mask_array.shape}")
        valid = ~mask_array.astype(bool)
    else:
        valid = np.ones(image_array.shape, dtype=bool)

    valid &= np.isfinite(image_array)
    valid &= np.isfinite(q_array)
    if intensity_error_array is not None:
        valid &= np.isfinite(intensity_error_array)
    if q_error_array is not None:
        valid &= np.isfinite(q_error_array)

    if not np.any(valid):
        raise ValueError("No valid pixels available for azimuthal averaging")
    if int(n_bins) <= 0:
        raise ValueError(f"n_bins must be > 0, got {n_bins!r}")

    q_values = q_array[valid].ravel()
    image_values = image_array[valid].ravel()
    q_min_value = float(np.min(q_values))
    q_max_value = float(np.max(q_values))
    if not np.isfinite(q_min_value) or not np.isfinite(q_max_value):
        raise ValueError("Invalid q values for azimuthal averaging")
    if q_max_value == q_min_value:
        edge_padding = max(abs(q_min_value), 1.0) * 1e-12
        q_min_value -= edge_padding
        q_max_value += edge_padding
    bin_edges = np.linspace(q_min_value, q_max_value, int(n_bins) + 1, dtype=np.float64)

    counts, _ = np.histogram(q_values, bins=bin_edges)
    intensity_sum, _ = np.histogram(q_values, bins=bin_edges, weights=image_values)
    q_sum, _ = np.histogram(q_values, bins=bin_edges, weights=q_values)

    intensity = np.full(int(n_bins), np.nan, dtype=np.float64)
    q_binned = np.full(int(n_bins), np.nan, dtype=np.float64)
    non_empty = counts > 0
    intensity[non_empty] = intensity_sum[non_empty] / counts[non_empty]
    q_binned[non_empty] = q_sum[non_empty] / counts[non_empty]

    if intensity_error_array is not None:
        error_values = intensity_error_array[valid].ravel()
        variance_sum, _ = np.histogram(
            q_values,
            bins=bin_edges,
            weights=np.square(error_values),
        )
        binned_intensity_error = np.full(int(n_bins), np.nan, dtype=np.float64)
        binned_intensity_error[non_empty] = np.sqrt(variance_sum[non_empty]) / counts[non_empty]
    else:
        binned_intensity_error = None

    if q_error_array is not None:
        q_error_values = q_error_array[valid].ravel()
        q_variance_sum, _ = np.histogram(
            q_values,
            bins=bin_edges,
            weights=np.square(q_error_values),
        )
        binned_q_error = np.full(int(n_bins), np.nan, dtype=np.float64)
        binned_q_error[non_empty] = np.sqrt(q_variance_sum[non_empty]) / counts[non_empty]
    else:
        binned_q_error = None

    return AzimuthalAverageResult(
        q=q_binned,
        intensity=intensity,
        intensity_error=binned_intensity_error,
        q_error=binned_q_error,
        counts=counts.astype(np.int64, copy=False),
    )
