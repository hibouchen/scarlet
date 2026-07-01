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
from scarlet.workflow.pipeline import (
    ReductionPipeline,
    ReductionState,
    azimuthal_averaging_step,
    save_azimuthal_text_step,
    save_processed_detectors_step,
    subtract_references_step,
    write_azimuthal_text_file,
)


def _write_detector_file(
    path: Path,
    *,
    sample_name: str,
    data: np.ndarray,
    monitor_integral: float = 1.0,
) -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = b"NXentry"
        entry.create_dataset("title", data=np.bytes_(sample_name))

        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = b"NXsample"
        sample.create_dataset("name", data=np.bytes_(sample_name))

        control = entry.create_group("control")
        control.attrs["NX_class"] = b"NXmonitor"
        control.create_dataset("integral", data=float(monitor_integral))

        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = b"NXinstrument"
        detector = instrument.create_group("detector0")
        detector.attrs["NX_class"] = b"NXdetector"
        detector.create_dataset("data", data=np.asarray(data, dtype=np.float64))
        detector.create_dataset("x_pixel_size", data=0.001)
        detector.create_dataset("y_pixel_size", data=0.001)
        detector.create_dataset("beam_center_x", data=0.5)
        detector.create_dataset("beam_center_y", data=0.5)


def _write_mask_bundle(path: Path, masks: dict[int, np.ndarray]) -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = b"NXentry"
        entry.create_dataset("definition", data=np.bytes_("SCARLET_masks"))
        entry.create_dataset("schema_version", data=np.bytes_("1.0"))

        configuration = entry.create_group("configuration")
        configuration.attrs["NX_class"] = b"NXcollection"
        configuration.create_dataset("wavelength", data=6.0)
        configuration.create_dataset("sample_detector_distance", data=4.2)

        mask_group = entry.create_group("mask")
        mask_group.attrs["NX_class"] = b"NXcollection"
        for detector_number, mask in sorted(masks.items()):
            mask_group.create_dataset(f"mask_detector{detector_number}", data=np.asarray(mask, dtype=np.uint8))

        meta = entry.create_group("meta")
        meta.attrs["NX_class"] = b"NXcollection"
        meta.create_dataset("created_utc", data=np.bytes_("2026-01-01T00:00:00Z"))
        meta.create_dataset("mask_convention", data=np.bytes_("1=masked, 0=valid"))
        meta.create_dataset("source_file", data=np.bytes_(str(path.resolve())))
        meta.create_dataset("source_entry_path", data=np.bytes_("/entry"))


class TestReductionPipelineFactories(unittest.TestCase):
    def test_with_processed_output_includes_save_step_after_normalization(self) -> None:
        pipeline = ReductionPipeline.with_processed_output()

        self.assertEqual(
            pipeline.step_names,
            (
                "subtract references",
                "water normalization",
                "save processed detectors",
            ),
        )

    def test_with_azimuthal_text_output_includes_save_text_step(self) -> None:
        pipeline = ReductionPipeline.with_azimuthal_text_output()

        self.assertEqual(
            pipeline.step_names,
            (
                "subtract references",
                "water normalization",
                "azimuthal averaging",
                "save azimuthal text",
            ),
        )


class TestAzimuthalTextWriter(unittest.TestCase):
    def test_write_azimuthal_text_file_writes_four_columns_and_header(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_path = write_azimuthal_text_file(
                Path(td) / "sample_config_cfg.txt",
                q=np.asarray([0.1, 0.2], dtype=np.float64),
                intensity=np.asarray([10.0, 20.0], dtype=np.float64),
                intensity_error=np.asarray([1.0, 2.0], dtype=np.float64),
                q_error=np.asarray([0.01, 0.02], dtype=np.float64),
                sample_name="sample_a",
                config_id="cfg",
                transmission=0.5,
            )

            self.assertEqual(output_path.name, "sample_config_cfg.txt")
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "sample_name: sample_a")
            self.assertEqual(lines[1], "config_id: cfg")
            self.assertEqual(lines[2], "transmission: 0.5")
            self.assertEqual(lines[3], "q I I_error q_error")
            self.assertEqual(len(lines[4].split()), 4)
            self.assertEqual(len(lines[5].split()), 4)


@unittest.skipIf(importlib.util.find_spec("scipp") is None, "scipp is required for workflow pipeline tests")
class TestWorkflowPipeline(unittest.TestCase):
    def test_subtract_references_step_applies_workflow_mask_to_output_dataarray(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample_scattering.nxs"
            sample_data = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float64)
            _write_detector_file(sample_path, sample_name="sample_a", data=sample_data)

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.add_run(
                RunKey(config_id="cfg", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )
            mask_path = root / "masks.nxs"
            _write_mask_bundle(mask_path, {0: np.asarray([[0, 1], [0, 0]], dtype=np.uint8)})
            ctx.set_mask_file("cfg", mask_path)

            state = ReductionState(sample_name="sample_a", config_id="cfg", workflow=ctx, transmission=1.0)
            updated = subtract_references_step(state)

            np.testing.assert_allclose(updated.detectors[0].data.values, sample_data)
            self.assertIn("workflow_config", updated.detectors[0].masks)
            np.testing.assert_array_equal(
                updated.detectors[0].masks["workflow_config"].values,
                np.asarray([[False, True], [False, False]], dtype=bool),
            )

    def test_azimuthal_averaging_step_integrates_detector_with_workflow_mask(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample_scattering.nxs"
            sample_data = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float64)
            _write_detector_file(sample_path, sample_name="sample_a", data=sample_data)

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.add_run(
                RunKey(config_id="cfg", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )
            ctx.configurations["cfg"] = Configuration(
                wavelength=6.0,
                sample_detector_distance=[4.2],
                config_id="cfg",
            )
            mask_path = root / "masks.nxs"
            _write_mask_bundle(mask_path, {0: np.asarray([[0, 1], [0, 0]], dtype=np.uint8)})
            ctx.set_mask_file("cfg", mask_path)

            state = ReductionState(sample_name="sample_a", config_id="cfg", workflow=ctx, transmission=1.0)
            original_detector = state.detectors[0]
            updated = azimuthal_averaging_step(state)

            q_map = compute_q_norm_map(
                sample_data,
                beam_center=(0.5, 0.5),
                detector_distance=4.2,
                pixel_size=(0.001, 0.001),
                wavelength=6.0,
            )
            expected = azimuthal_average(
                original_detector,
                q_map,
                mask=np.asarray([[0, 1], [0, 0]], dtype=np.uint8),
                n_bins=state.azimuthal_n_bins,
                q_scale=state.azimuthal_q_scale,
            ).to_data_array()

            self.assertEqual(updated.detectors[0].ndim, 1)
            np.testing.assert_allclose(updated.detectors[0].data.values, expected.data.values)
            np.testing.assert_allclose(updated.detectors[0].coords["q"].values, expected.coords["q"].values)
            np.testing.assert_array_equal(updated.detectors[0].coords["counts"].values, expected.coords["counts"].values)
            self.assertIn("Computed azimuthal average", " ".join(updated.notes))

    def test_save_processed_detectors_step_writes_processed_nxentry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample_scattering.nxs"
            sample_data = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float64)
            _write_detector_file(sample_path, sample_name="sample_a", data=sample_data)

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.add_run(
                RunKey(config_id="cfg", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )

            state = ReductionState(sample_name="sample_a", config_id="cfg", workflow=ctx, transmission=1.0)
            updated = save_processed_detectors_step(state)

            self.assertIs(updated, state)
            with h5py.File(sample_path, "r") as handle:
                self.assertEqual(handle.attrs["default"], b"processed")
                self.assertIn("/processed", handle)
                self.assertEqual(handle["/processed"].attrs["NX_class"], b"NXentry")
                self.assertEqual(handle["/processed"].attrs["default"], b"data0")
                self.assertEqual(handle["/processed/data"].attrs["NX_class"], b"NXcollection")
                self.assertEqual(handle["/processed/data/detector0"].attrs["NX_class"], b"NXdata")
                self.assertEqual(handle["/processed/data/detector0"].attrs["signal"], b"data")
                self.assertEqual(handle["/processed/data0"].attrs["NX_class"], b"NXdata")
                self.assertEqual(handle["/processed/data0"].attrs["signal"], b"data")
                np.testing.assert_allclose(handle["/processed/data/detector0/data"][()], sample_data)
                np.testing.assert_allclose(handle["/processed/data0/data"][()], sample_data)
                np.testing.assert_allclose(handle["/processed/data/detector0/x"][()], np.array([0.0, 1.0]))
                np.testing.assert_allclose(handle["/processed/data/detector0/y"][()], np.array([0.0, 1.0]))
                np.testing.assert_allclose(handle["/processed/data0/x"][()], np.array([0.0, 1.0]))
                np.testing.assert_allclose(handle["/processed/data0/y"][()], np.array([0.0, 1.0]))
                np.testing.assert_allclose(handle["/processed/data/detector0/errors"][()], np.sqrt(sample_data))
                np.testing.assert_allclose(handle["/processed/data0/errors"][()], np.sqrt(sample_data))
                self.assertEqual(handle["/processed/meta/source_entry"][()].decode(), "/entry")
                self.assertEqual(handle["/processed/meta/sample_name"][()].decode(), "sample_a")

    def test_save_azimuthal_text_step_writes_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample_scattering.nxs"
            sample_data = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float64)
            _write_detector_file(sample_path, sample_name="sample_a", data=sample_data)

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.add_run(
                RunKey(config_id="cfg", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )
            ctx.configurations["cfg"] = Configuration(
                wavelength=6.0,
                sample_detector_distance=[4.2],
                config_id="cfg",
            )

            state = ReductionState(sample_name="sample_a", config_id="cfg", workflow=ctx, transmission=1.0)
            state = azimuthal_averaging_step(state)
            updated = save_azimuthal_text_step(state)

            output_path = root / "out" / "sample_a_config_cfg.txt"
            self.assertIs(updated, state)
            self.assertTrue(output_path.exists())
            self.assertEqual(ctx.artifacts[-1].path, output_path.resolve())
            self.assertEqual(ctx.artifacts[-1].kind, "txt")


if __name__ == "__main__":
    unittest.main()
