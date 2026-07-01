from __future__ import annotations

import math
import unittest

from scarlet.workflow.configuration import (
    Configuration,
    compare_configurations_wavelength,
)


class TestCompareConfigurationsWavelength(unittest.TestCase):
    def test_compare_wavelength_only_accepts_values_within_tolerance(self) -> None:
        a = Configuration(wavelength=6.0, sample_detector_distance=4.2)
        b = Configuration(wavelength=6.05, sample_detector_distance=9.9)

        same, diffs = compare_configurations_wavelength(a, b, tol_a=0.1)

        self.assertTrue(same)
        self.assertEqual(diffs, [])

    def test_compare_wavelength_only_reports_difference_outside_tolerance(self) -> None:
        a = Configuration(wavelength=6.0, sample_detector_distance=4.2)
        b = Configuration(wavelength=6.2, sample_detector_distance=4.2)

        same, diffs = compare_configurations_wavelength(a, b, tol_a=0.1)

        self.assertFalse(same)
        self.assertEqual(len(diffs), 1)
        self.assertIn("wavelength", diffs[0])

    def test_compare_wavelength_only_reports_missing_value_for_nan(self) -> None:
        a = Configuration(wavelength=math.nan, sample_detector_distance=4.2)
        b = Configuration(wavelength=6.0, sample_detector_distance=4.2)

        same, diffs = compare_configurations_wavelength(a, b)

        self.assertFalse(same)
        self.assertEqual(diffs, ["wavelength: missing value(s) (a=nan, b=6.0)"])
