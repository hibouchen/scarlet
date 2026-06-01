from __future__ import annotations

import unittest

import numpy as np

from scarlet.reduction import normalize_by_solid_angle, subtract_scattering_references
from scarlet.workflow.pipeline import PipelineData, ReductionInputs, ReductionPipeline


class TestReductionPipeline(unittest.TestCase):
    def test_empty_cell_error_is_derived_from_monitor_normalized_image(self) -> None:
        empty_cell = PipelineData(
            image=np.full((2, 2), 2.5, dtype=np.float64),
            transmission=0.8,
            monitor=10.0,
            acquisition_time=10.0,
            deadtime=0.1,
        )
        np.testing.assert_allclose(empty_cell.error, np.full((2, 2), 0.5, dtype=np.float64))
        self.assertAlmostEqual(empty_cell.deadtime or 0.0, 0.1)
        np.testing.assert_allclose(
            empty_cell.deadtime_corrected_image,
            np.full((2, 2), 10.0 / 3.0, dtype=np.float64),
        )
        np.testing.assert_allclose(
            empty_cell.deadtime_corrected_error,
            np.full((2, 2), 8.0 / 9.0, dtype=np.float64),
        )

    def test_default_pipeline_declares_reduction_order(self) -> None:
        pipeline = ReductionPipeline.default()
        self.assertEqual(
            pipeline.step_names,
            (
                "subtract_references",
                "normalize_by_water",
                "normalize_by_solid_angle",
                "compute_q",
                "integrate_azimuthally",
            ),
        )

    def test_default_pipeline_applies_solid_angle_after_water_normalization(self) -> None:
        sample = np.full((3, 3), 10.0, dtype=np.float64)
        dark = np.full((3, 3), 2.0, dtype=np.float64)
        water = np.full((3, 3), 4.0, dtype=np.float64)
        inputs = ReductionInputs(
            sample_image=sample,
            sample_error=np.sqrt(sample),
            sample_transmission=0.5,
            detector_distance=2.0,
            beam_center=(1.0, 1.0),
            pixel_size=(0.001, 0.001),
            wavelength=6.0,
            n_bins=4,
            dark_image=dark,
            dark_error=np.sqrt(dark),
            water_corrected_image=water,
        )

        state = ReductionPipeline.default().run(inputs)

        corrected = subtract_scattering_references(
            sample,
            0.5,
            dark=dark,
            distance=2.0,
            beam_center=(1.0, 1.0),
        )
        expected = normalize_by_solid_angle(
            corrected / water,
            detector_distance=2.0,
            beam_center=(1.0, 1.0),
            pixel_size=(0.001, 0.001),
        )
        np.testing.assert_allclose(state.solid_angle_corrected, expected)


if __name__ == "__main__":
    unittest.main()
