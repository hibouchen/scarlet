from __future__ import annotations

from typing import Any

import numpy as np


def _detector_coordinates(
    image: Any,
    beam_center: tuple[float, float],
    detector_distance: float,
    pixel_size: float | tuple[float, float],
) -> tuple[tuple[np.ndarray, np.ndarray], float]:
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"image must be a 2D array, got shape {data.shape}")

    distance = float(detector_distance)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError(f"detector_distance must be > 0, got {detector_distance!r}")

    beam_center_x = float(beam_center[0])
    beam_center_y = float(beam_center[1])
    if not np.isfinite(beam_center_x) or not np.isfinite(beam_center_y):
        raise ValueError(f"beam_center must contain finite coordinates, got {beam_center!r}")

    if isinstance(pixel_size, tuple):
        pixel_size_x = float(pixel_size[0])
        pixel_size_y = float(pixel_size[1])
    else:
        pixel_size_x = float(pixel_size)
        pixel_size_y = float(pixel_size)
    if not np.isfinite(pixel_size_x) or pixel_size_x <= 0.0:
        raise ValueError(f"pixel_size_x must be > 0, got {pixel_size_x!r}")
    if not np.isfinite(pixel_size_y) or pixel_size_y <= 0.0:
        raise ValueError(f"pixel_size_y must be > 0, got {pixel_size_y!r}")

    ny, nx = data.shape
    x = (np.arange(nx, dtype=np.float64) - beam_center_x) * pixel_size_x
    y = (np.arange(ny, dtype=np.float64) - beam_center_y) * pixel_size_y
    return np.meshgrid(x, y), distance


def _detector_axis_vectors(
    image: Any,
    beam_center: tuple[float, float],
    pixel_size: float | tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"image must be a 2D array, got shape {data.shape}")

    beam_center_x = float(beam_center[0])
    beam_center_y = float(beam_center[1])
    if not np.isfinite(beam_center_x) or not np.isfinite(beam_center_y):
        raise ValueError(f"beam_center must contain finite coordinates, got {beam_center!r}")

    if isinstance(pixel_size, tuple):
        pixel_size_x = float(pixel_size[0])
        pixel_size_y = float(pixel_size[1])
    else:
        pixel_size_x = float(pixel_size)
        pixel_size_y = float(pixel_size)
    if not np.isfinite(pixel_size_x) or pixel_size_x <= 0.0:
        raise ValueError(f"pixel_size_x must be > 0, got {pixel_size_x!r}")
    if not np.isfinite(pixel_size_y) or pixel_size_y <= 0.0:
        raise ValueError(f"pixel_size_y must be > 0, got {pixel_size_y!r}")

    ny, nx = data.shape
    x = (np.arange(nx, dtype=np.float64) - beam_center_x) * pixel_size_x
    y = (np.arange(ny, dtype=np.float64) - beam_center_y) * pixel_size_y
    return x, y


def compute_qx_vector(
    image: Any,
    beam_center: tuple[float, float],
    detector_distance: float,
    pixel_size: float | tuple[float, float],
    wavelength: float,
) -> np.ndarray:
    """
    Return the 1D Qx values along the horizontal detector axis.

    The values are computed on the beam-center row, i.e. with ``y = 0``.
    """
    distance = float(detector_distance)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError(f"detector_distance must be > 0, got {detector_distance!r}")

    lam = float(wavelength)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"wavelength must be > 0, got {wavelength!r}")

    x, _ = _detector_axis_vectors(image, beam_center, pixel_size)
    ray_length = np.sqrt(x * x + distance * distance)
    return (2.0 * np.pi / lam) * (x / ray_length)


def compute_qy_vector(
    image: Any,
    beam_center: tuple[float, float],
    detector_distance: float,
    pixel_size: float | tuple[float, float],
    wavelength: float,
) -> np.ndarray:
    """
    Return the 1D Qy values along the vertical detector axis.

    The values are computed on the beam-center column, i.e. with ``x = 0``.
    """
    distance = float(detector_distance)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError(f"detector_distance must be > 0, got {detector_distance!r}")

    lam = float(wavelength)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"wavelength must be > 0, got {wavelength!r}")

    _, y = _detector_axis_vectors(image, beam_center, pixel_size)
    ray_length = np.sqrt(y * y + distance * distance)
    return (2.0 * np.pi / lam) * (y / ray_length)


def compute_theta_map(
    image: Any,
    beam_center: tuple[float, float],
    detector_distance: float,
    pixel_size: float | tuple[float, float],
) -> np.ndarray:
    """
    Return the half-scattering angle theta map for a detector image.

    The returned angles are in radians.
    """
    (xx, yy), distance = _detector_coordinates(
        image,
        beam_center,
        detector_distance,
        pixel_size,
    )
    radial_distance = np.sqrt(xx * xx + yy * yy)
    return 0.5 * np.arctan2(radial_distance, distance)


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
    lam = float(wavelength)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"wavelength must be > 0, got {wavelength!r}")

    theta = compute_theta_map(
        image,
        beam_center,
        detector_distance,
        pixel_size,
    )
    return (4.0 * np.pi / lam) * np.sin(theta)
