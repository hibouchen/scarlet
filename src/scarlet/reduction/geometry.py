from __future__ import annotations

from typing import Any

import numpy as np


def _require_positive_scalar(value: float, *, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return scalar


def _normalize_pixel_size(pixel_size: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(pixel_size, tuple):
        return (
            _require_positive_scalar(pixel_size[0], name="pixel_size_x"),
            _require_positive_scalar(pixel_size[1], name="pixel_size_y"),
        )
    value = _require_positive_scalar(pixel_size, name="pixel_size")
    return value, value


def _as_image_shape(image: Any) -> tuple[int, int]:
    shape = np.asarray(image).shape
    if len(shape) != 2:
        raise ValueError(f"image must be a 2D array, got shape {shape}")
    return int(shape[0]), int(shape[1])


def _detector_coordinates(
    image: Any,
    beam_center: tuple[float, float],
    *,
    detector_distance: float,
    pixel_size: float | tuple[float, float],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    ny, nx = _as_image_shape(image)
    distance = _require_positive_scalar(detector_distance, name="detector_distance")
    pixel_size_x, pixel_size_y = _normalize_pixel_size(pixel_size)

    beam_center_x = float(beam_center[0])
    beam_center_y = float(beam_center[1])
    if not np.isfinite(beam_center_x) or not np.isfinite(beam_center_y):
        raise ValueError(f"beam_center must contain finite coordinates, got {beam_center!r}")

    x = (np.arange(nx, dtype=np.float64) - beam_center_x) * pixel_size_x
    y = (np.arange(ny, dtype=np.float64) - beam_center_y) * pixel_size_y
    xx, yy = np.meshgrid(x, y)
    return (xx, yy), (x, y)


def compute_qx_vector(
    image: Any,
    beam_center: tuple[float, float],
    detector_distance: float,
    pixel_size: float | tuple[float, float],
    wavelength: float,
) -> np.ndarray:
    """
    Return the qx values sampled along the detector X axis through the beam center.
    """
    (_, _), (x, _) = _detector_coordinates(
        image,
        beam_center,
        detector_distance=detector_distance,
        pixel_size=pixel_size,
    )
    lam = _require_positive_scalar(wavelength, name="wavelength")
    two_theta_x = np.arctan2(x, float(detector_distance))
    return (4.0 * np.pi / lam) * np.sin(0.5 * two_theta_x)


def compute_qy_vector(
    image: Any,
    beam_center: tuple[float, float],
    detector_distance: float,
    pixel_size: float | tuple[float, float],
    wavelength: float,
) -> np.ndarray:
    """
    Return the qy values sampled along the detector Y axis through the beam center.
    """
    (_, _), (_, y) = _detector_coordinates(
        image,
        beam_center,
        detector_distance=detector_distance,
        pixel_size=pixel_size,
    )
    lam = _require_positive_scalar(wavelength, name="wavelength")
    two_theta_y = np.arctan2(y, float(detector_distance))
    return (4.0 * np.pi / lam) * np.sin(0.5 * two_theta_y)


def compute_theta_map(
    image: Any,
    beam_center: tuple[float, float],
    detector_distance: float,
    pixel_size: float | tuple[float, float],
) -> np.ndarray:
    """
    Return the Bragg angle theta map for a detector image.

    The returned angle is half of the scattering angle 2theta.
    """
    (xx, yy), _ = _detector_coordinates(
        image,
        beam_center,
        detector_distance=detector_distance,
        pixel_size=pixel_size,
    )
    radial_distance = np.sqrt(xx * xx + yy * yy)
    two_theta = np.arctan2(radial_distance, float(detector_distance))
    return 0.5 * two_theta


def compute_chi_map(
    image: Any,
    beam_center: tuple[float, float],
    pixel_size: float | tuple[float, float],
) -> np.ndarray:
    """
    Return the azimuthal angle chi map for a detector image.

    The returned angles are in radians, computed with ``atan2(y, x)``.
    """
    (xx, yy), _ = _detector_coordinates(
        image,
        beam_center,
        detector_distance=1.0,
        pixel_size=pixel_size,
    )
    return np.arctan2(yy, xx)


def compute_q_norm_map(
    image: Any,
    beam_center: tuple[float, float],
    detector_distance: float,
    pixel_size: float | tuple[float, float],
    wavelength: float,
) -> np.ndarray:
    """
    Return the |q| map for a detector image.

    The output has the same shape as `image`. The detector is assumed to be a
    flat plane perpendicular to the direct beam. The beam center is expressed
    in pixel coordinates `(x, y)`.
    """
    lam = _require_positive_scalar(wavelength, name="wavelength")
    theta = compute_theta_map(
        image,
        beam_center,
        detector_distance,
        pixel_size,
    )
    return (4.0 * np.pi / lam) * np.sin(theta)
