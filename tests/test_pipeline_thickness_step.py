from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from scarlet.workflow.context import WorkflowContext
from scarlet.workflow.pipeline import ReductionPipeline, ReductionState, normalize_by_thickness


@unittest.skipIf(importlib.util.find_spec("scipp") is None, "scipp is required for pipeline thickness step tests")
class TestPipelineThicknessStep(unittest.TestCase):
    def test_default_pipeline_includes_thickness_normalization(self) -> None:
        pipeline = ReductionPipeline.default()

        self.assertIn("normalize by thickness", pipeline.step_names)

    def _make_state(self) -> ReductionState:
        import scipp as sc

        state = ReductionState(
            sample_name="sample_a",
            config_id="cfg",
            workflow=WorkflowContext(),
            transmission=1.0,
        )
        state.detectors = {
            0: sc.DataArray(
                data=sc.array(
                    dims=["y", "x"],
                    values=np.full((2, 2), 4.0, dtype=np.float64),
                    variances=np.full((2, 2), 16.0, dtype=np.float64),
                )
            )
        }
        return state

    def test_normalize_by_thickness_divides_detector_data(self) -> None:
        state = self._make_state()
        state.workflow.set_sample_thickness("sample_a", "cfg", 2.0)

        updated = normalize_by_thickness(state)

        np.testing.assert_allclose(updated.detectors[0].data.values, np.full((2, 2), 2.0))
        np.testing.assert_allclose(updated.detectors[0].data.variances, np.full((2, 2), 4.0))
        self.assertIn("Normalized detector data by sample thickness 2 mm", updated.notes)

    def test_normalize_by_thickness_warns_and_leaves_data_unchanged_when_missing(self) -> None:
        state = self._make_state()

        updated = normalize_by_thickness(state)

        np.testing.assert_allclose(updated.detectors[0].data.values, np.full((2, 2), 4.0))
        self.assertEqual(len(updated.workflow.issues), 1)
        self.assertEqual(updated.workflow.issues[0].level, "WARN")
        self.assertIn("Missing sample thickness", updated.workflow.issues[0].message)

    def test_normalize_by_thickness_rejects_non_positive_values(self) -> None:
        state = self._make_state()
        state.workflow.set_sample_thickness("sample_a", "cfg", 0.0)

        with self.assertRaisesRegex(ValueError, "sample thickness must be > 0 mm"):
            normalize_by_thickness(state)


if __name__ == "__main__":
    unittest.main()
