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

def regiso(image, mask, x0, y0, x_pixel_size, y_pixel_size, bins=None, error=None):
    y, x = np.indices(image.shape, dtype='float')
    y = (y-y0)*y_pixel_size
    x = (x-x0)*x_pixel_size
    r_grid = np.ma.masked_array(data=np.sqrt(x**2+y**2), mask=mask).compressed()
    masked_data = np.ma.masked_array(data=image, mask=mask, dtype='float').compressed()
    # masked_data = np.ma.masked_invalid(masked_data)
    if bins is None:
        maxd = np.max(r_grid)
        bins = len(np.arange(0, maxd))
    edges = np.histogram_bin_edges(r_grid, bins=bins)
    # edges = np.arange(np.max(r_grid))
    indexes = np.digitize(r_grid, edges, right=False)
    counts = np.bincount(indexes)[1:]
    r_mean = np.bincount(indexes, weights=r_grid)[1:]/counts
    r_square = np.bincount(indexes, weights=r_grid**2)[1:]/counts
    intensity = np.bincount(indexes, weights=masked_data)[1:]/counts
    if error is None:
        di = np.sqrt(intensity*counts)/counts
    else:
        masked_error = np.ma.masked_array(data=error, mask=mask, dtype='float').compressed()
        di = np.sqrt(np.bincount(indexes, weights=masked_error**2)[1:]) / counts
    # TODO: check error bar on r
    dr = np.sqrt(r_square-r_mean**2+1/12)  # 1/12 is the variance of one pixel
    # dr = np.sqrt(r_square - r_mean ** 2 + 1 / 12)
    d = {'counts': np.bincount(indexes), 'edges': edges, 'indexes': indexes, 'r_grid': r_grid,
         'masked_data':  masked_data}
    return r_mean, intensity, di, dr


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
    bin_edges = np.histogram_bin_edges(q_values, bins=int(n_bins))
    indexes = np.digitize(q_values, bin_edges, right=False)
    indexes = np.clip(indexes, 1, int(n_bins))

    counts = np.bincount(indexes, minlength=int(n_bins) + 1)[1:]
    q_sum = np.bincount(indexes, weights=q_values, minlength=int(n_bins) + 1)[1:]
    intensity_sum = np.bincount(indexes, weights=image_values, minlength=int(n_bins) + 1)[1:]

    intensity = np.full(int(n_bins), np.nan, dtype=np.float64)
    q_binned = np.full(int(n_bins), np.nan, dtype=np.float64)
    non_empty = counts > 0
    intensity[non_empty] = intensity_sum[non_empty] / counts[non_empty]
    q_binned[non_empty] = q_sum[non_empty] / counts[non_empty]

    if intensity_error_array is not None:
        error_values = intensity_error_array[valid].ravel()
        variance_sum = np.bincount(
            indexes,
            weights=np.square(error_values),
            minlength=int(n_bins) + 1,
        )[1:]
        binned_intensity_error = np.full(int(n_bins), np.nan, dtype=np.float64)
        binned_intensity_error[non_empty] = np.sqrt(variance_sum[non_empty]) / counts[non_empty]
    else:
        binned_intensity_error = None

    if q_error_array is not None:
        q_error_values = q_error_array[valid].ravel()
        q_variance_sum = np.bincount(
            indexes,
            weights=np.square(q_error_values),
            minlength=int(n_bins) + 1,
        )[1:]
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
