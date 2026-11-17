from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.io import read_processed_sectors
from scarlet.reduction import Sector
from scarlet.workflow.configuration import Configuration
from scarlet.workflow.context import RunKey, WorkflowContext
from scarlet.workflow.pipeline import (
    ReductionPipeline,
    ReductionState,
    as_reduction_step,
    azimuthal_averaging_step,
    save_azimuthal_text_step,
    save_processed_detectors_step,
    save_sector_text_step,
    sector_averaging_step,
)


def _write_raw_file(path: Path, data: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("raw_data")
        control = entry.create_group("control")
        control.create_dataset("integral", data=1.0)
        instrument = entry.create_group("instrument")
        detector = instrument.create_group("detector0")
        detector.create_dataset("data", data=np.asarray(data, dtype=np.float64))
        detector.create_dataset("beam_center_x", data=2.0)
        detector.create_dataset("beam_center_y", data=2.0)
        detector.create_dataset("x_pixel_size", data=0.001)
        detector.create_dataset("y_pixel_size", data=0.001)


def _make_state(root: Path) -> tuple[ReductionState, Path]:
    sample_path = root / "sample_scattering.nxs"
    _write_raw_file(
        sample_path,
        np.arange(1.0, 26.0, dtype=np.float64).reshape(5, 5),
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
        sample_detector_distance=[4.2],
        config_id="cfg",
    )
    return (
        ReductionState(
            sample_name="sample",
            config_id="cfg",
            workflow=workflow,
        ),
        sample_path,
    )


@unittest.skipIf(
    importlib.util.find_spec("scipp") is None,
    "scipp is required for sector pipeline tests",
)
class TestPipelineSectorStep(unittest.TestCase):
    def test_pipeline_adds_sector_steps_only_when_sectors_are_configured(self) -> None:
        isotropic = ReductionPipeline.default(WorkflowContext())
        anisotropic = ReductionPipeline.default(
            WorkflowContext(),
            sectors=(
                Sector(
                    "parallel",
                    center_deg=0.0,
                    width_deg=30.0,
                    include_opposite=True,
                ),
            ),
        )

        self.assertNotIn("sector averaging", isotropic.step_names)
        self.assertNotIn("save sector text", isotropic.step_names)
        self.assertEqual(
            anisotropic.step_names[anisotropic.step_names.index("azimuthal averaging") + 1],
            "sector averaging",
        )
        self.assertEqual(anisotropic.step_names[-1], "save sector text")

    def test_pipeline_rejects_duplicate_sector_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "sector names must be unique"):
            ReductionPipeline(
                workflow=WorkflowContext(),
                sectors=(
                    Sector("parallel", center_deg=0.0, width_deg=20.0),
                    Sector("parallel", center_deg=90.0, width_deg=20.0),
                ),
            )

    def test_sector_step_preserves_corrected_2d_and_radial_products(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state, _sample_path = _make_state(Path(td))
            original = state.detectors[0].data.values.copy()
            pipeline = ReductionPipeline(
                workflow=state.workflow,
                steps=(
                    as_reduction_step(azimuthal_averaging_step),
                    as_reduction_step(sector_averaging_step),
                ),
                n_bins=3,
                sectors=(
                    Sector(
                        "parallel",
                        center_deg=0.0,
                        width_deg=60.0,
                        include_opposite=True,
                    ),
                ),
            )

            updated = pipeline.run(state)

            self.assertEqual(updated.detectors[0].ndim, 1)
            self.assertEqual(updated.products["corrected_2d"][0].ndim, 2)
            self.assertIs(updated.products["radial"][0], updated.detectors[0])
            np.testing.assert_allclose(
                updated.products["corrected_2d"][0].data.values,
                original,
            )
            sector_curve = updated.products["sectors"]["parallel"][0]
            self.assertEqual(tuple(sector_curve.dims), ("q",))
            self.assertEqual(sector_curve.sizes["q"], 3)
            self.assertAlmostEqual(sector_curve.coords["sector_center"].value, 0.0)
            self.assertAlmostEqual(sector_curve.coords["sector_width"].value, 60.0)
            self.assertTrue(sector_curve.coords["sector_include_opposite"].value)
            self.assertGreater(int(np.sum(sector_curve.coords["counts"].values)), 0)

    def test_sector_products_are_saved_to_nexus_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state, sample_path = _make_state(root)
            pipeline = ReductionPipeline(
                workflow=state.workflow,
                steps=(
                    as_reduction_step(azimuthal_averaging_step),
                    as_reduction_step(sector_averaging_step),
                    as_reduction_step(save_processed_detectors_step),
                    as_reduction_step(save_azimuthal_text_step),
                    as_reduction_step(save_sector_text_step),
                ),
                n_bins=3,
                sectors=(
                    Sector(
                        "parallel",
                        center_deg=0.0,
                        width_deg=60.0,
                        include_opposite=True,
                    ),
                ),
            )

            updated = pipeline.run(state)

            with h5py.File(sample_path, "r") as handle:
                self.assertIn("/processed/products/corrected_2d/detector0/data", handle)
                self.assertIn("/processed/products/radial/detector0/data", handle)
                sector_root = "/processed/products/sectors/parallel/detector0"
                self.assertIn(f"{sector_root}/data", handle)
                self.assertEqual(
                    handle["/processed/products/sectors/parallel"].attrs["label"],
                    b"parallel",
                )
                self.assertAlmostEqual(
                    float(handle[f"{sector_root}/sector_center"][()]),
                    0.0,
                )
                self.assertAlmostEqual(
                    float(handle[f"{sector_root}/sector_width"][()]),
                    60.0,
                )

            sectors = read_processed_sectors(sample_path)
            self.assertEqual(set(sectors), {"parallel"})
            self.assertEqual(sectors["parallel"].definition, (0.0, 60.0, True))
            self.assertEqual(len(sectors["parallel"].detectors), 1)
            self.assertEqual(sectors["parallel"].detectors[0].shape, (3, 4))

            sector_path = root / "out" / "sample_config_cfg_sector_parallel.txt"
            self.assertTrue(sector_path.exists())
            header = sector_path.read_text(encoding="utf-8")
            self.assertIn("sector_name: parallel", header)
            self.assertIn("sector_include_opposite: true", header)
            self.assertIn(sector_path.resolve(), [artifact.path for artifact in updated.workflow.artifacts])


if __name__ == "__main__":
    unittest.main()
