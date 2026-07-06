from __future__ import annotations

import importlib.util
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

    def test_azimuthal_average_supports_log_q_binning(self) -> None:
        image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        q_map = np.array([[0.11, 0.30], [0.50, 0.79]], dtype=np.float64)

        result = azimuthal_average(image, q_map, n_bins=3, q_scale="log")

        np.testing.assert_allclose(result.q, np.array([0.11, 0.30, 0.645]))
        np.testing.assert_allclose(result.intensity, np.array([1.0, 2.0, 3.5]))
        np.testing.assert_array_equal(result.counts, np.array([1, 1, 2]))

    def test_azimuthal_average_log_q_binning_requires_positive_q(self) -> None:
        image = np.array([[1.0, 2.0]], dtype=np.float64)
        q_map = np.array([[0.0, -0.1]], dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "No valid pixels available"):
            azimuthal_average(image, q_map, n_bins=2, q_scale="log")


@unittest.skipIf(importlib.util.find_spec("scipp") is None, "scipp is required for DataArray integration tests")
class TestIntegrationWithScippDataArray(unittest.TestCase):
    def test_azimuthal_average_uses_q_coord_variances_and_masks_from_dataarray(self) -> None:
        import scipp as sc

        image = sc.DataArray(
            data=sc.array(
                dims=["y", "x"],
                values=np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float64),
                variances=np.array([[1.0, 9.0], [25.0, 49.0]], dtype=np.float64),
            ),
            coords={
                "q": sc.array(
                    dims=["y", "x"],
                    values=np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float64),
                    variances=np.array([[0.0001, 0.25], [0.0009, 0.0016]], dtype=np.float64),
                )
            },
            masks={
                "user": sc.array(
                    dims=["y", "x"],
                    values=np.array([[False, True], [False, False]], dtype=bool),
                )
            },
        )

        result = azimuthal_average(image, n_bins=2)

        np.testing.assert_allclose(result.q, np.array([0.1, 0.2]))
        np.testing.assert_allclose(result.intensity, np.array([3.0, 7.0]))
        np.testing.assert_allclose(result.intensity_error, np.array([np.sqrt(26.0) / 2.0, 7.0]))
        np.testing.assert_allclose(result.q_error, np.array([np.sqrt(0.0010) / 2.0, 0.04]))
        np.testing.assert_array_equal(result.counts, np.array([2, 1]))

        integrated = result.to_data_array()
        np.testing.assert_allclose(integrated.data.values, result.intensity)
        np.testing.assert_allclose(integrated.coords["q"].values, result.q)
        self.assertIsNone(integrated.coords["q"].variances)
        np.testing.assert_allclose(integrated.coords["q_error"].values, result.q_error)
        np.testing.assert_array_equal(integrated.coords["counts"].values, result.counts)

    def test_azimuthal_average_requires_q_coord_when_q_map_is_omitted(self) -> None:
        import scipp as sc

        image = sc.DataArray(
            data=sc.array(
                dims=["y", "x"],
                values=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            )
        )

        with self.assertRaisesRegex(ValueError, "has no 'q' coord"):
            azimuthal_average(image, n_bins=2)


if __name__ == "__main__":
    unittest.main()
