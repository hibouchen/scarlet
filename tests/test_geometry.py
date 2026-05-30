from __future__ import annotations

import unittest

import numpy as np

from scarlet.reduction import (
    compute_chi_map,
    compute_q_norm_map,
    compute_qx_vector,
    compute_qy_vector,
    compute_theta_map,
)


class TestGeometry(unittest.TestCase):
    def test_compute_qx_qy_vectors_are_centered_and_antisymmetric(self) -> None:
        image = np.zeros((3, 3), dtype=np.float64)

        qx = compute_qx_vector(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
        )
        qy = compute_qy_vector(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
        )

        self.assertEqual(qx.shape, (3,))
        self.assertEqual(qy.shape, (3,))
        self.assertAlmostEqual(float(qx[1]), 0.0)
        self.assertAlmostEqual(float(qy[1]), 0.0)
        self.assertAlmostEqual(float(qx[0]), -float(qx[2]))
        self.assertAlmostEqual(float(qy[0]), -float(qy[2]))

    def test_compute_theta_map_is_zero_at_beam_center(self) -> None:
        image = np.zeros((3, 3), dtype=np.float64)

        theta_map = compute_theta_map(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
        )

        self.assertEqual(theta_map.shape, image.shape)
        self.assertAlmostEqual(float(theta_map[1, 1]), 0.0)

    def test_compute_chi_map_returns_expected_cartesian_angles(self) -> None:
        image = np.zeros((3, 3), dtype=np.float64)

        chi_map = compute_chi_map(
            image,
            beam_center=(1.0, 1.0),
            pixel_size=1.0,
        )

        self.assertAlmostEqual(float(chi_map[1, 2]), 0.0)
        self.assertAlmostEqual(float(chi_map[2, 1]), np.pi / 2.0)
        self.assertAlmostEqual(float(chi_map[1, 0]), np.pi)
        self.assertAlmostEqual(float(chi_map[0, 1]), -np.pi / 2.0)

    def test_compute_q_norm_map_is_zero_at_beam_center(self) -> None:
        image = np.zeros((3, 3), dtype=np.float64)

        q_map = compute_q_norm_map(
            image,
            beam_center=(1.0, 1.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
        )

        self.assertEqual(q_map.shape, image.shape)
        self.assertAlmostEqual(float(q_map[1, 1]), 0.0)

    def test_compute_q_norm_map_matches_single_pixel_scattering_angle(self) -> None:
        image = np.zeros((1, 2), dtype=np.float64)

        q_map = compute_q_norm_map(
            image,
            beam_center=(0.0, 0.0),
            detector_distance=2.0,
            pixel_size=0.5,
            wavelength=6.0,
        )

        two_theta = np.arctan2(0.5, 2.0)
        expected = (4.0 * np.pi / 6.0) * np.sin(0.5 * two_theta)
        self.assertAlmostEqual(float(q_map[0, 1]), float(expected))


if __name__ == "__main__":
    unittest.main()
