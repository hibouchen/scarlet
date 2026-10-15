from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from scarlet.workflow.context import WorkflowContext
from scarlet.workflow.pipeline import (
    ReductionPipeline,
    ReductionState,
    absolute_normalization_step,
)


def _write_flatfield_calibration(
    path: Path,
    *,
    direct_flux: float = 100.0,
    flux_detector_number: int = 2,
    factors: dict[int, float] | None = None,
) -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        flux = entry.create_dataset("direct_flux", data=direct_flux)
        flux.attrs["detector_number"] = flux_detector_number
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = np.bytes_("NXinstrument")
        for detector_number, factor in (factors or {}).items():
            detector = instrument.create_group(f"detector{detector_number}")
            detector.attrs["NX_class"] = np.bytes_("NXdetector")
            detector.create_dataset("normalisation_factor", data=factor)


@unittest.skipIf(importlib.util.find_spec("scipp") is None, "scipp is required")
class TestAbsoluteNormalizationStep(unittest.TestCase):
    def _make_state(self, root: Path, factors: dict[int, float]) -> tuple[ReductionState, Path]:
        import scipp as sc

        flatfield_path = root / "flatfield.nxs"
        _write_flatfield_calibration(flatfield_path, factors=factors)
        context = WorkflowContext(output_dir=root / "out")
        context.set_flatfield("cfg", flatfield_path)
        state = ReductionState(
            sample_name="sample_a",
            config_id="cfg",
            workflow=context,
            transmission=1.0,
        )
        state.detectors = {
            detector_number: sc.DataArray(
                data=sc.array(
                    dims=["y", "x"],
                    values=np.full((2, 2), 3.0, dtype=np.float64),
                    variances=np.full((2, 2), 4.0, dtype=np.float64),
                ),
                masks={
                    "pixel_mask": sc.array(
                        dims=["y", "x"],
                        values=np.array([[False, True], [False, False]]),
                    )
                },
            )
            for detector_number in factors
        }
        return state, flatfield_path

    def test_applies_per_detector_factor_and_flatfield_to_configuration_flux_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state, flatfield_path = self._make_state(root, {0: 2.0, 1: 3.0})
            direct_beam_path = root / "direct_beam.nxs"

            with (
                mock.patch.object(
                    state.workflow,
                    "prepare_reference",
                    return_value=direct_beam_path,
                ),
                mock.patch(
                    "scarlet.workflow.pipeline.compute_direct_flux",
                    return_value=50.0,
                ) as compute_flux,
            ):
                updated = absolute_normalization_step(state)

            # Flux ratio = 100 / 50 = 2. Detector scales are therefore 4 and 6.
            np.testing.assert_allclose(updated.detectors[0].data.values, 12.0)
            np.testing.assert_allclose(updated.detectors[0].data.variances, 64.0)
            np.testing.assert_allclose(updated.detectors[1].data.values, 18.0)
            np.testing.assert_allclose(updated.detectors[1].data.variances, 144.0)
            self.assertEqual(str(updated.detectors[0].unit), "1/cm")
            np.testing.assert_array_equal(
                updated.detectors[0].masks["pixel_mask"].values,
                np.array([[False, True], [False, False]]),
            )
            compute_flux.assert_called_once_with(direct_beam_path, detector_number=2)
            self.assertIn(flatfield_path.name, updated.notes[-1])
            self.assertIn("flatfield/configuration=2", updated.notes[-1])

    def test_rejects_flatfield_without_detector_factor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state, _flatfield_path = self._make_state(root, {0: 2.0})
            del state.detectors[0]

            import scipp as sc

            state.detectors[1] = sc.DataArray(
                sc.array(dims=["y", "x"], values=np.ones((2, 2)))
            )
            with self.assertRaisesRegex(
                ValueError,
                "Missing normalisation_factor for detector1",
            ):
                absolute_normalization_step(state)

    def test_default_pipelines_apply_absolute_normalization_before_azimuthal_average(self) -> None:
        for pipeline in (
            ReductionPipeline.default(WorkflowContext()),
            ReductionPipeline.with_azimuthal_text_output(WorkflowContext()),
        ):
            self.assertLess(
                pipeline.step_names.index("absolute normalization"),
                pipeline.step_names.index("azimuthal averaging"),
            )
            self.assertEqual(
                pipeline.step_names[pipeline.step_names.index("absolute normalization") - 1],
                "normalize by thickness",
            )


if __name__ == "__main__":
    unittest.main()
