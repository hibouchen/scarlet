from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.reduction.geometry import compute_q_norm_map
from scarlet.reduction.integration import azimuthal_average
from scarlet.workflow.configuration import Configuration
from scarlet.workflow.context import RunKey, WorkflowContext
from scarlet.workflow.pipeline import ReductionPipeline, ReductionState, as_reduction_step, azimuthal_averaging_step


def _write_raw_file(
    path: Path,
    *,
    data: np.ndarray,
    monitor_integral: float = 1.0,
    beam_center: tuple[float, float] = (1.5, 1.5),
    pixel_size: tuple[float, float] = (0.001, 0.001),
) -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("raw_data")
        control = entry.create_group("control")
        control.create_dataset("integral", data=float(monitor_integral))
        instrument = entry.create_group("instrument")
        detector = instrument.create_group("detector0")
        detector.create_dataset("data", data=np.asarray(data, dtype=np.float64))
        detector.create_dataset("beam_center_x", data=float(beam_center[0]))
        detector.create_dataset("beam_center_y", data=float(beam_center[1]))
        detector.create_dataset("x_pixel_size", data=float(pixel_size[0]))
        detector.create_dataset("y_pixel_size", data=float(pixel_size[1]))


@unittest.skipIf(importlib.util.find_spec("scipp") is None, "scipp is required for pipeline azimuthal step tests")
class TestPipelineAzimuthalStep(unittest.TestCase):
    def test_azimuthal_averaging_step_integrates_detector_dataarrays(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample_scattering.nxs"
            sample_data = np.array(
                [
                    [0.0, 1.0, 2.0, 0.0],
                    [1.0, 5.0, 7.0, 2.0],
                    [2.0, 7.0, 5.0, 1.0],
                    [0.0, 2.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            )
            _write_raw_file(sample_path, data=sample_data, monitor_integral=2.0)

            workflow = WorkflowContext(root_dir=root, output_dir=root / "out")
            workflow.add_run(
                RunKey(
                    config_id="cfg",
                    entity="sample",
                    mode="scattering",
                    sample_name="sample",
                ),
                sample_path,
            )
            workflow.configurations["cfg"] = Configuration(
                wavelength=6.0,
                sample_detector_distance=[4.2],
                config_id="cfg",
            )

            state = ReductionState(
                sample_name="sample",
                config_id="cfg",
                workflow=workflow,
                azimuthal_n_bins=3,
            )
            detector = state.detectors[0]
            q_map = compute_q_norm_map(
                detector.data.values,
                beam_center=(1.5, 1.5),
                detector_distance=4.2,
                pixel_size=(0.001, 0.001),
                wavelength=6.0,
            )
            expected = azimuthal_average(detector, q_map, n_bins=3).to_data_array()

            pipeline = ReductionPipeline(steps=(as_reduction_step(azimuthal_averaging_step),))
            updated = pipeline.run(state)

            integrated = updated.detectors[0]
            self.assertEqual(tuple(integrated.dims), ("q",))
            np.testing.assert_allclose(integrated.data.values, expected.data.values)
            np.testing.assert_allclose(integrated.data.variances, expected.data.variances)
            np.testing.assert_allclose(integrated.coords["q"].values, expected.coords["q"].values)
            np.testing.assert_array_equal(integrated.coords["counts"].values, expected.coords["counts"].values)
            self.assertEqual(updated.reductions_steps, ["azimuthal averaging"])
            self.assertIn("Computed azimuthal average", updated.notes[-1])


if __name__ == "__main__":
    unittest.main()
