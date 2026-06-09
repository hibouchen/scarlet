from __future__ import annotations

import unittest

import numpy as np

from scarlet.reduction import apply_prefactor, concatenate_curves, crop_curve


class TestReductionUtils(unittest.TestCase):
    def test_crop_curve_applies_q_range_and_keeps_errors(self) -> None:
        q, intensity, intensity_error, q_error = crop_curve(
            [0.1, 0.2, 0.3, 0.4],
            [1.0, 2.0, 3.0, 4.0],
            q_min=0.15,
            q_max=0.35,
            intensity_error=[0.1, 0.2, 0.3, 0.4],
            q_error=[0.01, 0.02, 0.03, 0.04],
        )

        np.testing.assert_allclose(q, np.array([0.2, 0.3]))
        np.testing.assert_allclose(intensity, np.array([2.0, 3.0]))
        np.testing.assert_allclose(intensity_error, np.array([0.2, 0.3]))
        np.testing.assert_allclose(q_error, np.array([0.02, 0.03]))

    def test_apply_prefactor_supports_callable_and_scales_errors(self) -> None:
        q, intensity, intensity_error, q_error = apply_prefactor(
            [0.1, 0.2, 0.3],
            [10.0, 20.0, 30.0],
            lambda q: q**2,
            intensity_error=[1.0, 2.0, 3.0],
            q_error=[0.01, 0.02, 0.03],
        )

        np.testing.assert_allclose(q, np.array([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(intensity, np.array([0.1, 0.8, 2.7]))
        np.testing.assert_allclose(intensity_error, np.array([0.01, 0.08, 0.27]))
        np.testing.assert_allclose(q_error, np.array([0.01, 0.02, 0.03]))

    def test_concatenate_curves_can_sort_output(self) -> None:
        q, intensity, intensity_error, q_error = concatenate_curves(
            (
                np.array([0.3, 0.4]),
                np.array([3.0, 4.0]),
                np.array([0.3, 0.4]),
                np.array([0.03, 0.04]),
            ),
            (
                np.array([0.1, 0.2]),
                np.array([1.0, 2.0]),
                np.array([0.1, 0.2]),
                np.array([0.01, 0.02]),
            ),
            sort=True,
        )

        np.testing.assert_allclose(q, np.array([0.1, 0.2, 0.3, 0.4]))
        np.testing.assert_allclose(intensity, np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(intensity_error, np.array([0.1, 0.2, 0.3, 0.4]))
        np.testing.assert_allclose(q_error, np.array([0.01, 0.02, 0.03, 0.04]))

    def test_concatenate_curves_drops_partial_error_series(self) -> None:
        q, intensity, intensity_error, q_error = concatenate_curves(
            (np.array([0.1]), np.array([1.0]), np.array([0.1])),
            (np.array([0.2]), np.array([2.0])),
        )

        np.testing.assert_allclose(q, np.array([0.1, 0.2]))
        np.testing.assert_allclose(intensity, np.array([1.0, 2.0]))
        self.assertIsNone(intensity_error)
        self.assertIsNone(q_error)

    def test_apply_prefactor_validates_prefactor_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "prefactor shape mismatch"):
            apply_prefactor([0.1, 0.2], [1.0, 2.0], np.array([2.0, 3.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
