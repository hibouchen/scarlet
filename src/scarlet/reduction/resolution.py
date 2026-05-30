from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import compute_q_norm_map, compute_qx_vector, compute_qy_vector

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
