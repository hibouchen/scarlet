from __future__ import annotations

import unittest

import numpy as np

from scarlet.reduction import (
    compute_beam_divergence,
    compute_q_norm_map,
    compute_q_uncertainty_map,
    compute_qx_uncertainty_vector,
    compute_qy_uncertainty_vector,
)


class TestResolution(unittest.TestCase):
    def test_compute_beam_divergence_uses_slit_sizes_and_collimation_distance(self) -> None:
        sigma_div_x, sigma_div_y = compute_beam_divergence(
            entry_slit_size=(0.004, 0.006),
            exit_slit_size=(0.002, 0.003),
            collimation_distance=3.0,
        )

        self.assertAlmostEqual(sigma_div_x, (0.004 + 0.002) / (3.0 * np.sqrt(12.0)))
        self.assertAlmostEqual(sigma_div_y, (0.006 + 0.003) / (3.0 * np.sqrt(12.0)))

    def test_compute_qx_qy_uncertainty_vectors_match_wavelength_contribution(self) -> None:
        image = np.zeros((3, 3), dtype=np.float64)
        baseline_qx = compute_qx_uncertainty_vector(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
        )
        baseline_qy = compute_qy_uncertainty_vector(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
        )

        sigma_qx = compute_qx_uncertainty_vector(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
            wavelength_uncertainty=0.3,
        )
        sigma_qy = compute_qy_uncertainty_vector(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
            wavelength_uncertainty=0.3,
        )

        qx_wavelength_only = np.sqrt(np.square(sigma_qx) - np.square(baseline_qx))
        qy_wavelength_only = np.sqrt(np.square(sigma_qy) - np.square(baseline_qy))

        self.assertGreater(float(baseline_qx[1]), 0.0)
        self.assertGreater(float(baseline_qy[1]), 0.0)
        np.testing.assert_allclose(qx_wavelength_only[1], 0.0)
        np.testing.assert_allclose(qy_wavelength_only[1], 0.0)
        np.testing.assert_allclose(qx_wavelength_only[0], qx_wavelength_only[2])
        np.testing.assert_allclose(qy_wavelength_only[0], qy_wavelength_only[2])
        self.assertGreater(float(qx_wavelength_only[0]), 0.0)
        self.assertGreater(float(qy_wavelength_only[0]), 0.0)

    def test_compute_q_uncertainty_map_has_pixel_limited_floor(self) -> None:
        image = np.zeros((3, 3), dtype=np.float64)

        sigma_q = compute_q_uncertainty_map(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
        )

        self.assertGreater(float(sigma_q[1, 1]), 0.0)
        self.assertGreater(float(sigma_q[0, 0]), 0.0)

    def test_compute_q_uncertainty_map_matches_wavelength_contribution(self) -> None:
        image = np.zeros((3, 3), dtype=np.float64)
        baseline_sigma_q = compute_q_uncertainty_map(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
        )
        q_map = compute_q_norm_map(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
        )

        sigma_q = compute_q_uncertainty_map(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
            wavelength_uncertainty=0.3,
        )

        wavelength_only = np.sqrt(np.square(sigma_q) - np.square(baseline_sigma_q))
        np.testing.assert_allclose(wavelength_only, q_map * (0.3 / 6.0))

    def test_compute_q_uncertainty_map_includes_beam_divergence(self) -> None:
        image = np.zeros((3, 3), dtype=np.float64)

        sigma_q = compute_q_uncertainty_map(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
            beam_divergence=0.01,
        )

        self.assertGreater(float(sigma_q[1, 1]), 0.0)
        self.assertGreater(float(sigma_q[0, 2]), 0.0)

    def test_compute_qx_qy_uncertainty_vectors_include_beam_divergence(self) -> None:
        image = np.zeros((3, 3), dtype=np.float64)

        sigma_qx = compute_qx_uncertainty_vector(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
            beam_divergence=0.01,
        )
        sigma_qy = compute_qy_uncertainty_vector(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
            beam_divergence=0.01,
        )

        self.assertGreater(float(sigma_qx[1]), 0.0)
        self.assertGreater(float(sigma_qy[1]), 0.0)
        self.assertGreater(float(sigma_qx[0]), 0.0)
        self.assertGreater(float(sigma_qy[0]), 0.0)

    def test_compute_q_uncertainty_map_includes_pixel_size_spread_without_divergence(self) -> None:
        image = np.zeros((3, 3), dtype=np.float64)

        sigma_q = compute_q_uncertainty_map(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
        )

        self.assertGreater(float(sigma_q[1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
