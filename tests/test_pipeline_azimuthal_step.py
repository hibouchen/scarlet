from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.reduction.geometry import compute_q_norm_map
from scarlet.reduction.integration import azimuthal_average
from scarlet.workflow.configuration import Aperture, Collimation, Configuration
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


def _append_raw_detector(
    path: Path,
    *,
    detector_number: int,
    data: np.ndarray,
    beam_center: tuple[float, float] = (1.5, 1.5),
    pixel_size: tuple[float, float] = (0.001, 0.001),
) -> None:
    with h5py.File(path, "a") as handle:
        detector = handle["/raw_data/instrument"].create_group(
            f"detector{detector_number}"
        )
        detector.create_dataset("data", data=np.asarray(data, dtype=np.float64))
        detector.create_dataset("beam_center_x", data=float(beam_center[0]))
        detector.create_dataset("beam_center_y", data=float(beam_center[1]))
        detector.create_dataset("x_pixel_size", data=float(pixel_size[0]))
        detector.create_dataset("y_pixel_size", data=float(pixel_size[1]))


@unittest.skipIf(importlib.util.find_spec("scipp") is None, "scipp is required for pipeline azimuthal step tests")
class TestPipelineAzimuthalStep(unittest.TestCase):
    def test_pipeline_uses_detector_specific_bin_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample_scattering.nxs"
            sample_data = np.arange(1.0, 17.0, dtype=np.float64).reshape(4, 4)
            _write_raw_file(sample_path, data=sample_data)
            _append_raw_detector(
                sample_path,
                detector_number=1,
                data=sample_data[::-1],
            )

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
                sample_detector_distance=[4.2, 2.1],
                config_id="cfg",
            )
            state = ReductionState(
                sample_name="sample",
                config_id="cfg",
                workflow=workflow,
            )
            pipeline = ReductionPipeline(
                workflow=workflow,
                steps=(as_reduction_step(azimuthal_averaging_step),),
                n_bins={0: 3, 1: 5},
                q_scale="log",
            )

            updated = pipeline.run(state)

            self.assertEqual(updated.detectors[0].sizes["q"], 3)
            self.assertEqual(updated.detectors[1].sizes["q"], 5)
            self.assertEqual(updated.azimuthal_n_bins, {0: 3, 1: 5})
            self.assertEqual(updated.azimuthal_q_scale, "log")
            self.assertIn("detector0=3", updated.notes[-1])
            self.assertIn("detector1=5", updated.notes[-1])
            self.assertIn("(log)", updated.notes[-1])

    def test_pipeline_rejects_missing_detector_bin_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample_scattering.nxs"
            _write_raw_file(sample_path, data=np.ones((4, 4), dtype=np.float64))

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
            )
            pipeline = ReductionPipeline(
                workflow=workflow,
                steps=(as_reduction_step(azimuthal_averaging_step),),
                n_bins={1: 5},
            )

            with self.assertRaisesRegex(ValueError, "Missing n_bins for detector0"):
                pipeline.run(state)

    def test_default_pipeline_accepts_detector_specific_bin_counts(self) -> None:
        pipeline = ReductionPipeline.default(
            WorkflowContext(),
            n_bins={0: 100, 1: 150, 2: 250},
            q_scale="log",
        )

        self.assertEqual(pipeline.n_bins, {0: 100, 1: 150, 2: 250})
        self.assertEqual(pipeline.q_scale, "log")

    def test_pipeline_rejects_invalid_bin_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_bins for detector1 must be > 0"):
            ReductionPipeline(
                workflow=WorkflowContext(),
                n_bins={0: 100, 1: 0},
            )

    def test_pipeline_rejects_invalid_q_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "q_scale must be 'linear' or 'log'"):
            ReductionPipeline(
                workflow=WorkflowContext(),
                q_scale="quadratic",  # type: ignore[arg-type]
            )

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

            pipeline = ReductionPipeline(
                workflow=workflow,
                steps=(as_reduction_step(azimuthal_averaging_step),),
            )
            updated = pipeline.run(state)

            integrated = updated.detectors[0]
            self.assertEqual(tuple(integrated.dims), ("q",))
            np.testing.assert_allclose(integrated.data.values, expected.data.values)
            np.testing.assert_allclose(integrated.data.variances, expected.data.variances)
            np.testing.assert_allclose(integrated.coords["q"].values, expected.coords["q"].values)
            np.testing.assert_array_equal(integrated.coords["counts"].values, expected.coords["counts"].values)
            self.assertEqual(updated.reductions_steps, ["azimuthal averaging"])
            self.assertIn("Computed azimuthal average", updated.notes[-1])

    def test_azimuthal_averaging_step_supports_pinhole_aperture(self) -> None:
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
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.002),
                    aperture2=Aperture(type="pinhole", diameter=0.004),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )

            state = ReductionState(
                sample_name="sample",
                config_id="cfg",
                workflow=workflow,
                azimuthal_n_bins=3,
            )
            updated = azimuthal_averaging_step(state)

            integrated = updated.detectors[0]
            self.assertEqual(tuple(integrated.dims), ("q",))
            self.assertIn("q_error", integrated.coords)
            self.assertTrue(np.isfinite(integrated.coords["q_error"].values).any())


if __name__ == "__main__":
    unittest.main()
