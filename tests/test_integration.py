from __future__ import annotations

import unittest

import numpy as np

from scarlet.reduction import azimuthal_average


class TestIntegration(unittest.TestCase):
    def test_azimuthal_average_bins_intensity_q_and_applies_mask(self) -> None:
        image = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float64)
        q_map = np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float64)
        mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)

        result = azimuthal_average(image, q_map, mask=mask, n_bins=2)

        np.testing.assert_allclose(result.q, np.array([0.1, 0.2]))
        np.testing.assert_allclose(result.intensity, np.array([3.0, 7.0]))
        np.testing.assert_array_equal(result.counts, np.array([2, 1]))
        self.assertIsNone(result.intensity_error)
        self.assertIsNone(result.q_error)

    def test_azimuthal_average_ignores_non_finite_pixels_and_propagates_errors(self) -> None:
        image = np.array([[1.0, np.nan], [5.0, 7.0]], dtype=np.float64)
        q_map = np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float64)
        intensity_error = np.array([[1.0, 10.0], [3.0, 4.0]], dtype=np.float64)
        q_error = np.array([[0.01, 0.50], [0.03, 0.04]], dtype=np.float64)

        result = azimuthal_average(
            image,
            q_map,
            intensity_error=intensity_error,
            q_error=q_error,
            n_bins=2,
        )

        np.testing.assert_allclose(result.q, np.array([0.1, 0.2]))
        np.testing.assert_allclose(result.intensity, np.array([3.0, 7.0]))
        np.testing.assert_array_equal(result.counts, np.array([2, 1]))
        np.testing.assert_allclose(result.intensity_error, np.array([np.sqrt(10.0) / 2.0, 4.0]))
        np.testing.assert_allclose(result.q_error, np.array([np.sqrt(0.0010) / 2.0, 0.04]))


if __name__ == "__main__":
    unittest.main()
