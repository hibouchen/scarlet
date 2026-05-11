from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

ROI = tuple[int, int, int, int]


@dataclass(frozen=True)
class TransmissionComputation:
    """Transmission computed from a rectangular direct-beam ROI."""

    value: float
    sample_roi_sum: float
    empty_beam_roi_sum: float
    roi: ROI
    method: str


def require_same_shape(label: str, reference: np.ndarray, other: np.ndarray) -> None:
    """Raise if two arrays do not have the same shape."""
    if reference.shape != other.shape:
        raise ValueError(f"Shape mismatch for {label}: expected {reference.shape}, got {other.shape}")


def zeros_like(reference: np.ndarray) -> np.ndarray:
    """Return a float64 array of zeros with the same shape as *reference*."""
    return np.zeros_like(reference, dtype=np.float64)


def normalize_image(image: np.ndarray, normalization: float) -> np.ndarray:
    """Normalize an image by a strictly positive monitor/counting value."""
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError(f"Normalization must be positive and finite, got {normalization!r}")
    return np.asarray(image, dtype=np.float64) / float(normalization)


def subtract_dark(image: np.ndarray, dark: Optional[np.ndarray]) -> np.ndarray:
    """Subtract a dark image, or leave the image unchanged if no dark is supplied."""
    image = np.asarray(image, dtype=np.float64)
    if dark is None:
        return image.copy()
    dark = np.asarray(dark, dtype=np.float64)
    require_same_shape("dark", image, dark)
    return image - dark


def subtract_empty_beam(image: np.ndarray, empty_beam: Optional[np.ndarray]) -> np.ndarray:
    """Subtract an empty-beam scattering image, or leave unchanged if absent."""
    image = np.asarray(image, dtype=np.float64)
    if empty_beam is None:
        return image.copy()
    empty_beam = np.asarray(empty_beam, dtype=np.float64)
    require_same_shape("empty_beam", image, empty_beam)
    return image - empty_beam


def background_correct_scattering(
    scattering: np.ndarray,
    *,
    dark: Optional[np.ndarray] = None,
    empty_beam_scattering: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply the first common scattering-image correction.

    The operation is deliberately simple and explicit::

        corrected = scattering - dark - (empty_beam_scattering - dark)

    If ``empty_beam_scattering`` is absent, the operation reduces to
    ``scattering - dark``. In the current deterministic pipeline, reference
    images are already monitor-normalized before this function is called.
    """
    dark_data = zeros_like(scattering) if dark is None else np.asarray(dark, dtype=np.float64)
    require_same_shape("dark", np.asarray(scattering), dark_data)

    out = np.asarray(scattering, dtype=np.float64) - dark_data
    if empty_beam_scattering is not None:
        empty_beam_data = np.asarray(empty_beam_scattering, dtype=np.float64)
        require_same_shape("empty_beam_scattering", out, empty_beam_data)
        out = out - (empty_beam_data - dark_data)
    return out


def roi_view(data: np.ndarray, roi: ROI) -> np.ndarray:
    """Return a half-open Python view ``data[y0:y1, x0:x1]``."""
    x0, x1, y0, y1 = roi
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Transmission ROI requires a 2D detector image, got ndim={data.ndim}")
    ny, nx = data.shape
    if not (0 <= x0 < x1 <= nx and 0 <= y0 < y1 <= ny):
        raise ValueError(f"Invalid ROI {roi} for detector image shape {data.shape}")
    return data[y0:y1, x0:x1]


def roi_sum(data: np.ndarray, roi: ROI) -> float:
    """Return the finite sum of a rectangular ROI."""
    value = float(np.nansum(roi_view(data, roi)))
    if not np.isfinite(value):
        raise ValueError(f"Non-finite ROI sum for ROI={roi}")
    return value


def compute_transmission(
    sample_transmission: np.ndarray,
    empty_beam_transmission: np.ndarray,
    *,
    dark: Optional[np.ndarray],
    roi: ROI,
    method: str,
) -> TransmissionComputation:
    """
    Compute a transmission from normalized 2D detector images.

    The SCARLET ROI convention is half-open, matching Python slicing:
    ``data[y0:y1, x0:x1]``.
    """
    sample_transmission = np.asarray(sample_transmission, dtype=np.float64)
    empty_beam_transmission = np.asarray(empty_beam_transmission, dtype=np.float64)
    require_same_shape("empty_beam_transmission", sample_transmission, empty_beam_transmission)

    dark_data = zeros_like(sample_transmission) if dark is None else np.asarray(dark, dtype=np.float64)
    require_same_shape("dark", sample_transmission, dark_data)

    sample_roi_sum = roi_sum(sample_transmission - dark_data, roi)
    empty_beam_roi_sum = roi_sum(empty_beam_transmission - dark_data, roi)
    if empty_beam_roi_sum == 0:
        raise ValueError("Cannot compute transmission: empty-beam ROI sum is zero")

    return TransmissionComputation(
        value=sample_roi_sum / empty_beam_roi_sum,
        sample_roi_sum=sample_roi_sum,
        empty_beam_roi_sum=empty_beam_roi_sum,
        roi=roi,
        method=method,
    )


def subtract_empty_cell(
    sample_corrected: np.ndarray,
    empty_cell_corrected: Optional[np.ndarray],
    *,
    transmission: float,
) -> np.ndarray:
    """
    Subtract the empty-cell contribution from a background-corrected sample.

    The current deterministic convention is::

        sample_minus_cell = sample_corrected - T_sample * empty_cell_corrected
    """
    sample_corrected = np.asarray(sample_corrected, dtype=np.float64)
    if empty_cell_corrected is None:
        return sample_corrected.copy()
    empty_cell_corrected = np.asarray(empty_cell_corrected, dtype=np.float64)
    require_same_shape("empty_cell_corrected", sample_corrected, empty_cell_corrected)
    return sample_corrected - float(transmission) * empty_cell_corrected


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division, returning NaN where the denominator is invalid."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    require_same_shape("denominator", numerator, denominator)
    out = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    np.divide(numerator, denominator, out=out, where=valid)
    return out


def normalize_by_water(sample_corrected: np.ndarray, water_corrected: np.ndarray) -> np.ndarray:
    """Normalize a corrected sample image by a corrected water image."""
    return safe_divide(sample_corrected, water_corrected)


def apply_mask(data: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Return a copy of *data* with masked pixels set to NaN."""
    out = np.asarray(data, dtype=np.float64).copy()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        require_same_shape("mask", out, mask)
        out[mask] = np.nan
    return out
