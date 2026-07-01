from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import compute_q_norm_map, compute_qx_vector, compute_qy_vector


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


def _centered_axis(length: int, beam_center: float, pixel_size: float) -> np.ndarray:
    return (np.arange(length, dtype=np.float64) - float(beam_center)) * pixel_size


def compute_beam_divergence(
    *,
    entry_slit_size: tuple[float, float],
    exit_slit_size: tuple[float, float],
    collimation_distance: float,
) -> tuple[float, float]:
    """Estimate RMS beam divergence in x and y from slit openings."""
    distance = _require_positive_scalar(collimation_distance, name="collimation_distance")
    entry_x, entry_y = _normalize_pixel_size(entry_slit_size)
    exit_x, exit_y = _normalize_pixel_size(exit_slit_size)
    scale = distance * np.sqrt(12.0)
    return (entry_x + exit_x) / scale, (entry_y + exit_y) / scale


def _qx_qy_sigma_components(
    image: Any,
    beam_center: tuple[float, float],
    *,
    detector_distance: float,
    pixel_size: float | tuple[float, float],
    wavelength: float,
    wavelength_uncertainty: float = 0.0,
    beam_divergence: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    shape = np.asarray(image).shape
    if len(shape) != 2:
        raise ValueError(f"image must be a 2D array, got shape {shape}")

    distance = _require_positive_scalar(detector_distance, name="detector_distance")
    lam = _require_positive_scalar(wavelength, name="wavelength")
    sigma_lambda = abs(float(wavelength_uncertainty))
    sigma_divergence = abs(float(beam_divergence))
    pixel_size_x, pixel_size_y = _normalize_pixel_size(pixel_size)

    x = _centered_axis(int(shape[1]), beam_center[0], pixel_size_x)
    y = _centered_axis(int(shape[0]), beam_center[1], pixel_size_y)
    qx = compute_qx_vector(image, beam_center, distance, (pixel_size_x, pixel_size_y), lam)
    qy = compute_qy_vector(image, beam_center, distance, (pixel_size_x, pixel_size_y), lam)

    q_scale = 4.0 * np.pi / lam
    pixel_sigma_x = pixel_size_x / np.sqrt(12.0)
    pixel_sigma_y = pixel_size_y / np.sqrt(12.0)

    dqx_dx = q_scale * 0.5 * np.cos(0.5 * np.arctan2(x, distance)) * distance / (distance**2 + x**2)
    dqy_dy = q_scale * 0.5 * np.cos(0.5 * np.arctan2(y, distance)) * distance / (distance**2 + y**2)

    sigma_qx = np.square(dqx_dx * pixel_sigma_x)
    sigma_qx += np.square(qx * (sigma_lambda / lam))
    sigma_qx += np.square(q_scale * 0.5 * sigma_divergence)

    sigma_qy = np.square(dqy_dy * pixel_sigma_y)
    sigma_qy += np.square(qy * (sigma_lambda / lam))
    sigma_qy += np.square(q_scale * 0.5 * sigma_divergence)
    return np.sqrt(sigma_qx), np.sqrt(sigma_qy)


def compute_qx_uncertainty_vector(
    image: Any,
    beam_center: tuple[float, float],
    detector_distance: float,
    pixel_size: float | tuple[float, float],
    wavelength: float,
    *,
    wavelength_uncertainty: float = 0.0,
    beam_divergence: float = 0.0,
) -> np.ndarray:
    """Return the 1-sigma uncertainty along the detector X axis through the beam center."""
    sigma_qx, _ = _qx_qy_sigma_components(
        image,
        beam_center,
        detector_distance=detector_distance,
        pixel_size=pixel_size,
        wavelength=wavelength,
        wavelength_uncertainty=wavelength_uncertainty,
        beam_divergence=beam_divergence,
    )
    return sigma_qx


def compute_qy_uncertainty_vector(
    image: Any,
    beam_center: tuple[float, float],
    detector_distance: float,
    pixel_size: float | tuple[float, float],
    wavelength: float,
    *,
    wavelength_uncertainty: float = 0.0,
    beam_divergence: float = 0.0,
) -> np.ndarray:
    """Return the 1-sigma uncertainty along the detector Y axis through the beam center."""
    _, sigma_qy = _qx_qy_sigma_components(
        image,
        beam_center,
        detector_distance=detector_distance,
        pixel_size=pixel_size,
        wavelength=wavelength,
        wavelength_uncertainty=wavelength_uncertainty,
        beam_divergence=beam_divergence,
    )
    return sigma_qy


def compute_q_uncertainty_map(
    image: Any,
    beam_center: tuple[float, float],
    detector_distance: float,
    pixel_size: float | tuple[float, float],
    wavelength: float,
    *,
    wavelength_uncertainty: float = 0.0,
    beam_divergence: float = 0.0,
) -> np.ndarray:
    """Return a 1-sigma |q| uncertainty map using pixel, wavelength, and divergence terms."""
    shape = np.asarray(image).shape
    if len(shape) != 2:
        raise ValueError(f"image must be a 2D array, got shape {shape}")

    distance = _require_positive_scalar(detector_distance, name="detector_distance")
    lam = _require_positive_scalar(wavelength, name="wavelength")
    sigma_lambda = abs(float(wavelength_uncertainty))
    sigma_divergence = abs(float(beam_divergence))
    pixel_size_x, pixel_size_y = _normalize_pixel_size(pixel_size)

    q_map = compute_q_norm_map(image, beam_center, distance, (pixel_size_x, pixel_size_y), lam)
    x = _centered_axis(int(shape[1]), beam_center[0], pixel_size_x)
    y = _centered_axis(int(shape[0]), beam_center[1], pixel_size_y)
    xx, yy = np.meshgrid(x, y)
    radial_distance = np.sqrt(xx * xx + yy * yy)
    theta = 0.5 * np.arctan2(radial_distance, distance)
    q_scale = 4.0 * np.pi / lam

    with np.errstate(divide="ignore", invalid="ignore"):
        radial_gradient = q_scale * 0.5 * np.cos(theta) * distance / (distance**2 + radial_distance**2)
        safe_radius = np.where(radial_distance > 0.0, radial_distance, 1.0)
        dq_dx = radial_gradient * (xx / safe_radius)
        dq_dy = radial_gradient * (yy / safe_radius)

    dq_dx = np.where(radial_distance > 0.0, dq_dx, 0.0)
    dq_dy = np.where(radial_distance > 0.0, dq_dy, 0.0)

    pixel_sigma_x = pixel_size_x / np.sqrt(12.0)
    pixel_sigma_y = pixel_size_y / np.sqrt(12.0)

    sigma_q2 = np.square(dq_dx * pixel_sigma_x) + np.square(dq_dy * pixel_sigma_y)
    sigma_q2 += np.square(q_map * (sigma_lambda / lam))
    sigma_q2 += np.square(q_scale * np.cos(theta) * sigma_divergence)
    sigma_q2 += np.square(np.where(radial_distance == 0.0, q_scale * 0.5 * pixel_sigma_x / distance, 0.0))
    return np.sqrt(sigma_q2)

def compute_q_resolution_circular(
    q: np.ndarray,
    r1: float,
    r2: float,
    collimation_distance: float,
    distance: float,
    wavelength_spread: float,
    wavelength: float,
    pixel_size: tuple[float, float],
) -> np.ndarray:
    """Compute the q-resolution for a circular beam profile. The resolution is computed as the quadrature sum of contributions from beam divergence, wavelength spread, and pixel size."""
    
    sigx = (distance / collimation_distance)**2 * r1**2 / 4
    sigx += ((distance+collimation_distance) / collimation_distance)**2 * r2**2 / 4
    sigx += 1/3 * pixel_size[0]**2

    sigy = (distance / collimation_distance)**2 * r1**2 / 4
    sigy += ((distance+collimation_distance) / collimation_distance)**2 * r2**2 / 4
    sigy += 1/3 * pixel_size[1]**2

    sigq2 = (2* np.pi / wavelength)**2 * (sigx + sigy) / distance**2
    sigq2 += q**2 * (wavelength_spread / wavelength)**2 * 1/6
    return np.sqrt(sigq2)
