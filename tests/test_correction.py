from __future__ import annotations

import unittest

import numpy as np

from scarlet.reduction import (
    correct_detector_dead_time,
    normalize_by_monitor,
    normalize_by_solid_angle,
)


class TestCorrection(unittest.TestCase):
    def test_normalize_by_solid_angle_uses_central_pixel_solid_angle(self) -> None:
        image = np.array([[5.0]], dtype=np.float64)

        normalized = normalize_by_solid_angle(
            image,
            detector_distance=2.0,
            beam_center=(0.0, 0.0),
            pixel_size=0.5,
        )

        expected_solid_angle = (0.5 * 0.5) / (2.0 * 2.0)
        np.testing.assert_allclose(normalized, image / expected_solid_angle)

    def test_normalize_by_solid_angle_increases_off_axis_pixels(self) -> None:
        image = np.ones((3, 3), dtype=np.float64)

        normalized = normalize_by_solid_angle(
            image,
            detector_distance=1.0,
            beam_center=(1.0, 1.0),
            pixel_size=(1.0, 1.0),
        )

        self.assertAlmostEqual(float(normalized[1, 1]), 1.0)
        self.assertGreater(float(normalized[0, 0]), float(normalized[1, 1]))

    def test_normalize_by_monitor_divides_image_by_monitor(self) -> None:
        image = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)

        normalized = normalize_by_monitor(image, monitor=10.0)

        np.testing.assert_allclose(normalized, image / 10.0)

    def test_normalize_by_monitor_requires_positive_monitor(self) -> None:
        with self.assertRaisesRegex(ValueError, "monitor must be > 0"):
            normalize_by_monitor(np.ones((2, 2), dtype=np.float64), monitor=0.0)

    def test_correct_detector_dead_time_returns_dead_time_corrected_rate(self) -> None:
        image = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)

        corrected = correct_detector_dead_time(image, acq_time=10.0, deadtime=0.1)

        expected_rate = image / 10.0
        expected = expected_rate / (1.0 - expected_rate * 0.1)
        np.testing.assert_allclose(corrected, expected)

    def test_correct_detector_dead_time_rejects_invalid_denominator(self) -> None:
        image = np.array([[20.0]], dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "undefined"):
            correct_detector_dead_time(image, acq_time=10.0, deadtime=0.5)


if __name__ == "__main__":
    unittest.main()
