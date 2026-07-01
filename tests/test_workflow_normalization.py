from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.reduction.correction import normalize_by_solid_angle
from scarlet.workflow.context import RunKey, WorkflowContext
from scarlet.workflow.configuration import Aperture, Collimation, Configuration
from scarlet.workflow.normalization import _extract_flatfield_payload, load_flatfield_file, save_flatfield_file
from scarlet.workflow.pipeline import ReductionState, normalization_step


def _write_detector_file(
    path: Path,
    *,
    sample_name: str,
    data: np.ndarray,
    monitor_integral: float = 1.0,
    wavelength: float = 6.0,
    detector_distance: float = 4.2,
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
        monochromator = instrument.create_group("monochromator")
        monochromator.attrs["NX_class"] = b"NXmonochromator"
        monochromator.create_dataset("wavelength", data=float(wavelength))
        detector = instrument.create_group("detector0")
        detector.attrs["NX_class"] = b"NXdetector"
        detector.create_dataset("data", data=np.asarray(data, dtype=np.float64))
        detector.create_dataset("distance", data=float(detector_distance))
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
        collimation = configuration.create_group("collimation")
        collimation.attrs["NX_class"] = b"NXcollection"
        collimation.create_dataset("collimation_distance", data=1.5)
        collimation.create_dataset("last_aperture_to_sample_distance", data=0.5)
        aperture1 = collimation.create_group("aperture1")
        aperture1.attrs["NX_class"] = b"NXslit"
        aperture1.create_dataset("x_gap", data=0.002)
        aperture1.create_dataset("y_gap", data=0.003)
        aperture2 = collimation.create_group("aperture2")
        aperture2.attrs["NX_class"] = b"NXpinhole"
        aperture2.create_dataset("diameter", data=0.004)

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


def _test_collimation() -> Collimation:
    return Collimation(
        aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
        aperture2=Aperture(type="pinhole", diameter=0.004),
        collimation_distance=1.5,
        last_aperture_to_sample_distance=0.5,
    )


class TestWorkflowNormalizationRegistry(unittest.TestCase):
    def test_flatfield_registry_round_trips_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            flatfield_path = root / "flatfield_cfg.nxs"
            ctx = WorkflowContext(output_dir=root / "out")

            self.assertIsNone(ctx.get_flatfield("cfg"))

            ctx.set_flatfield("cfg", flatfield_path)

            self.assertEqual(ctx.get_flatfield("cfg"), flatfield_path.resolve())

    def test_flatfield_source_resolves_to_source_configuration(self) -> None:
        ctx = WorkflowContext()
        ctx.set_flatfield("cfg_source", Path("/tmp/flatfield_cfg_source.nxs"))

        resolved = ctx.set_flatfield_source("cfg_target", "cfg_source")

        self.assertEqual(resolved, "cfg_source")
        self.assertEqual(ctx.get_flatfield_source("cfg_target"), "cfg_source")
        self.assertEqual(ctx.resolve_flatfield_config("cfg_target"), "cfg_source")
        self.assertEqual(ctx.get_flatfield("cfg_target"), Path("/tmp/flatfield_cfg_source.nxs").resolve())

    def test_flatfield_source_rejects_cycles(self) -> None:
        ctx = WorkflowContext()
        ctx.set_flatfield_source("cfg_b", "cfg_c")

        with self.assertRaisesRegex(ValueError, "Cycle detected"):
            ctx.set_flatfield_source("cfg_c", "cfg_b")

    def test_mask_file_registry_round_trips_detector_masks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = WorkflowContext()
            mask_path = root / "masks.nxs"
            mask = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
            _write_mask_bundle(mask_path, {0: mask})

            self.assertIsNone(ctx.get_mask("cfg", 0))

            ctx.set_mask_file("cfg", mask_path)

            self.assertEqual(ctx.get_mask_file("cfg"), mask_path.resolve())
            stored = ctx.get_mask("cfg", 0)
            assert stored is not None
            np.testing.assert_array_equal(stored, mask)
            np.testing.assert_array_equal(ctx.get_masks("cfg")[0], mask)

    def test_attach_mask_bundle_uses_embedded_config_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = WorkflowContext(output_dir=root)
            mask_path = root / "config_mask.nxs"
            _write_mask_bundle(mask_path, {0: np.asarray([[0, 1], [1, 0]], dtype=np.uint8)})
            with h5py.File(mask_path, "r+") as handle:
                handle["/entry"].create_dataset("config_id", data=np.bytes_("cfg"))

            attached = ctx.attach_mask_bundle(mask_path)

            self.assertEqual(attached, "cfg")
            self.assertEqual(ctx.get_mask_file("cfg"), mask_path.resolve())
            np.testing.assert_array_equal(ctx.get_mask("cfg", 0), np.asarray([[0, 1], [1, 0]], dtype=np.uint8))

    def test_attach_mask_bundle_matches_workflow_configuration_without_config_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = WorkflowContext(output_dir=root)
            ctx.configurations["cfg_a"] = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                collimation=_test_collimation(),
                config_id="cfg_a",
            )
            mask_path = root / "config_mask.nxs"
            _write_mask_bundle(mask_path, {0: np.asarray([[1, 0], [0, 1]], dtype=np.uint8)})

            attached = ctx.attach_mask_bundle(mask_path)

            self.assertEqual(attached, "cfg_a")
            self.assertEqual(ctx.get_mask_file("cfg_a"), mask_path.resolve())

    def test_attach_mask_bundles_from_output_dir_discovers_matching_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = WorkflowContext(output_dir=root)
            ctx.configurations["cfg_a"] = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                collimation=_test_collimation(),
                config_id="cfg_a",
            )
            mask_path = root / "cfg_a_masks.nxs"
            _write_mask_bundle(mask_path, {0: np.asarray([[1, 0], [0, 1]], dtype=np.uint8)})

            attached = ctx.attach_mask_bundles_from_output_dir()

            self.assertEqual(attached, {"cfg_a": mask_path.resolve()})
            self.assertEqual(ctx.get_mask_file("cfg_a"), mask_path.resolve())

    def test_attach_mask_bundles_from_output_dir_skips_incomplete_configuration_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = WorkflowContext(output_dir=root)
            ctx.configurations["cfg_a"] = Configuration(
                wavelength=6.0,
                sample_detector_distance=9.3,
                collimation=_test_collimation(),
                config_id="cfg_a",
            )
            mask_path = root / "cfg_a_masks.nxs"
            _write_mask_bundle(mask_path, {0: np.asarray([[1, 0], [0, 1]], dtype=np.uint8)})
            with h5py.File(mask_path, "r+") as handle:
                del handle["/entry/configuration/sample_detector_distance"]

            attached = ctx.attach_mask_bundles_from_output_dir()

            self.assertEqual(attached, {})
            self.assertIsNone(ctx.get_mask_file("cfg_a"))
            self.assertTrue(any("No workflow configuration matched mask bundle" in issue.message for issue in ctx.issues))

    def test_attach_mask_bundles_from_output_dir_skips_unmatched_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = WorkflowContext(output_dir=root)
            ctx.configurations["cfg_a"] = Configuration(
                wavelength=9.0,
                sample_detector_distance=4.2,
                collimation=_test_collimation(),
                config_id="cfg_a",
            )
            mask_path = root / "cfg_a_masks.nxs"
            _write_mask_bundle(mask_path, {0: np.asarray([[1, 0], [0, 1]], dtype=np.uint8)})

            attached = ctx.attach_mask_bundles_from_output_dir()

            self.assertEqual(attached, {})
            self.assertIsNone(ctx.get_mask_file("cfg_a"))
            self.assertTrue(any("No workflow configuration matched mask bundle" in issue.message for issue in ctx.issues))

    def test_setting_mask_invalidates_cached_flatfield(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = WorkflowContext()
            ctx.set_flatfield("cfg", root / "flatfield_cfg.nxs")
            mask_path = root / "masks.nxs"
            _write_mask_bundle(mask_path, {0: np.asarray([[0, 1], [0, 0]], dtype=np.uint8)})

            ctx.set_mask_file("cfg", mask_path)

            self.assertIsNone(ctx.get_flatfield("cfg"))

    def test_extract_flatfield_payload_applies_workflow_mask(self) -> None:
        class _FakeData:
            def __init__(self, values: np.ndarray):
                self.values = values
                self.variances = np.ones_like(values, dtype=np.float64)

        class _FakeArray:
            def __init__(self, values: np.ndarray):
                self.data = _FakeData(values)

        corrected = _FakeArray(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        workflow_mask = np.asarray([[0, 1], [0, 0]], dtype=np.uint8)

        flatfield, errors, pixel_mask = _extract_flatfield_payload(corrected, external_mask=workflow_mask)

        np.testing.assert_allclose(flatfield, np.asarray([[1.0, 1.0], [3.0, 4.0]], dtype=np.float64))
        np.testing.assert_array_equal(pixel_mask, np.asarray([[0, 1], [0, 0]], dtype=np.uint8))
        np.testing.assert_allclose(errors[[0, 1, 1], [0, 0, 1]], np.full(3, 1.0, dtype=np.float64))
        self.assertEqual(float(errors[0, 1]), 0.0)

    def test_save_and_load_flatfield_nexus_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_path = root / "water_scattering.nxs"
            flatfield_path = root / "flatfield_cfg.nxs"
            _write_detector_file(
                source_path,
                sample_name="water",
                data=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            )

            written = save_flatfield_file(
                {0: np.asarray([[0.5, 1.0], [1.5, 1.0]], dtype=np.float64)},
                errors={0: np.asarray([[0.1, 0.1], [0.2, 0.2]], dtype=np.float64)},
                masks={0: np.asarray([[0, 0], [1, 0]], dtype=np.uint8)},
                file_path=flatfield_path,
                config_id="cfg",
                water_scattering_path=source_path,
                water_transmission_path=source_path,
                dark_path=None,
                empty_cell_path=None,
                mask_file_path=None,
            )

            self.assertEqual(written, flatfield_path.resolve())

            with h5py.File(written, "r") as handle:
                self.assertEqual(handle["/entry/definition"][()].decode(), "SCARLET_flatfield")
                self.assertEqual(handle["/entry/config_id"][()].decode(), "cfg")
                np.testing.assert_allclose(
                    handle["/entry/instrument/detector0/flatfield"][()],
                    np.asarray([[0.5, 1.0], [1.5, 1.0]], dtype=np.float64),
                )
                np.testing.assert_allclose(
                    handle["/entry/instrument/detector0/flatfield_errors"][()],
                    np.asarray([[0.1, 0.1], [0.2, 0.2]], dtype=np.float64),
                )
                np.testing.assert_array_equal(
                    handle["/entry/instrument/detector0/pixel_mask"][()],
                    np.asarray([[0, 0], [1, 0]], dtype=np.uint8),
                )
                self.assertEqual(
                    handle["/entry/provenance/water_scattering_file"][()].decode(),
                    str(source_path.resolve()),
                )


@unittest.skipIf(importlib.util.find_spec("scipp") is None, "scipp is required for workflow normalization tests")
class TestWorkflowNormalizationPipeline(unittest.TestCase):
    def test_build_water_flatfield_and_apply_normalization_step(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample_scattering.nxs"
            water_scattering_path = root / "water_scattering.nxs"
            water_transmission_path = root / "water_transmission.nxs"
            _write_detector_file(
                sample_path,
                sample_name="sample_a",
                data=np.asarray([[4.0, 8.0], [12.0, 16.0]], dtype=np.float64),
            )
            _write_detector_file(
                water_scattering_path,
                sample_name="water",
                data=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            )
            _write_detector_file(
                water_transmission_path,
                sample_name="water",
                data=np.ones((2, 2), dtype=np.float64),
            )

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.add_run(
                RunKey(config_id="cfg", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )
            ctx.add_run(
                RunKey(config_id="cfg", entity="water", mode="scattering", sample_name="water"),
                water_scattering_path,
            )
            ctx.add_run(
                RunKey(config_id="cfg", entity="water", mode="transmission", sample_name="water"),
                water_transmission_path,
            )
            ctx.set_transmission("water", "cfg", 1.0)
            mask_path = root / "masks.nxs"
            _write_mask_bundle(mask_path, {0: np.asarray([[0, 1], [0, 0]], dtype=np.uint8)})
            ctx.set_mask_file("cfg", mask_path)

            flatfield_path = ctx.build_water_flatfield("cfg")
            flatfields = load_flatfield_file(flatfield_path)
            expected_flatfield = normalize_by_solid_angle(
                np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
                detector_distance=4.2,
                beam_center=(0.5, 0.5),
                pixel_size=(0.001, 0.001),
            )
            expected_flatfield[0, 1] = 1.0
            expected_variances = np.square(
                normalize_by_solid_angle(
                    np.asarray([[1.0, 1.0], [np.sqrt(3.0), 2.0]], dtype=np.float64),
                    detector_distance=4.2,
                    beam_center=(0.5, 0.5),
                    pixel_size=(0.001, 0.001),
                )
            )
            expected_variances[0, 1] = 0.0
            np.testing.assert_allclose(
                flatfields[0].data.values,
                expected_flatfield,
            )
            np.testing.assert_allclose(
                flatfields[0].data.variances,
                expected_variances,
            )
            np.testing.assert_array_equal(
                flatfields[0].masks["pixel_mask"].values,
                np.asarray([[False, True], [False, False]], dtype=bool),
            )

            with h5py.File(flatfield_path, "r") as handle:
                np.testing.assert_array_equal(
                    handle["/entry/instrument/detector0/pixel_mask"][()],
                    np.asarray([[0, 1], [0, 0]], dtype=np.uint8),
                )

            state = ReductionState(sample_name="sample_a", config_id="cfg", workflow=ctx, transmission=1.0)
            state = normalization_step(state)

            expected_final = np.asarray(
                [[4.0, np.nan], [4.0, 4.0]],
                dtype=np.float64,
            )
            observed = np.asarray(state.detectors[0].data.values, dtype=np.float64)
            valid_pixels = np.asarray([[True, False], [True, True]], dtype=bool)
            np.testing.assert_allclose(observed[valid_pixels], expected_final[valid_pixels])
            self.assertIsNotNone(state.detectors[0].data.variances)
            self.assertIn("pixel_mask", state.detectors[0].masks)
            np.testing.assert_array_equal(
                state.detectors[0].masks["pixel_mask"].values,
                np.asarray([[False, True], [False, False]], dtype=bool),
            )
            self.assertIn("solid-angle correction", " ".join(state.notes))

    def test_normalization_step_preserves_sample_and_flatfield_masks(self) -> None:
        import scipp as sc

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample_scattering.nxs"
            water_scattering_path = root / "water_scattering.nxs"
            water_transmission_path = root / "water_transmission.nxs"
            _write_detector_file(
                sample_path,
                sample_name="sample_a",
                data=np.asarray([[4.0, 8.0], [12.0, 16.0]], dtype=np.float64),
            )
            _write_detector_file(
                water_scattering_path,
                sample_name="water",
                data=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            )
            _write_detector_file(
                water_transmission_path,
                sample_name="water",
                data=np.ones((2, 2), dtype=np.float64),
            )

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.add_run(
                RunKey(config_id="cfg", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )
            ctx.add_run(
                RunKey(config_id="cfg_ff", entity="water", mode="scattering", sample_name="water"),
                water_scattering_path,
            )
            ctx.add_run(
                RunKey(config_id="cfg_ff", entity="water", mode="transmission", sample_name="water"),
                water_transmission_path,
            )
            ctx.set_transmission("water", "cfg_ff", 1.0)
            ctx.set_flatfield_source("cfg", "cfg_ff")

            sample_mask_path = root / "sample_masks.nxs"
            _write_mask_bundle(sample_mask_path, {0: np.asarray([[1, 0], [0, 0]], dtype=np.uint8)})
            ctx.set_mask_file("cfg", sample_mask_path)

            flatfield_mask_path = root / "flatfield_masks.nxs"
            _write_mask_bundle(flatfield_mask_path, {0: np.asarray([[0, 1], [0, 0]], dtype=np.uint8)})
            ctx.set_mask_file("cfg_ff", flatfield_mask_path)

            state = ReductionState(sample_name="sample_a", config_id="cfg", workflow=ctx, transmission=1.0)
            state.detectors[0] = state.detectors[0].copy(deep=False)
            state.detectors[0].masks["sample_only"] = sc.array(
                dims=["y", "x"],
                values=np.asarray([[False, False], [True, False]], dtype=bool),
            )

            state = normalization_step(state)

            self.assertIn("workflow_config", state.detectors[0].masks)
            self.assertIn("sample_only", state.detectors[0].masks)
            self.assertIn("pixel_mask", state.detectors[0].masks)
            np.testing.assert_array_equal(
                state.detectors[0].masks["workflow_config"].values,
                np.asarray([[True, False], [False, False]], dtype=bool),
            )
            np.testing.assert_array_equal(
                state.detectors[0].masks["sample_only"].values,
                np.asarray([[False, False], [True, False]], dtype=bool),
            )
            np.testing.assert_array_equal(
                state.detectors[0].masks["pixel_mask"].values,
                np.asarray([[False, True], [False, False]], dtype=bool),
            )

    def test_build_water_flatfield_can_use_source_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            water_scattering_path = root / "water_scattering.nxs"
            water_transmission_path = root / "water_transmission.nxs"
            _write_detector_file(
                water_scattering_path,
                sample_name="water",
                data=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            )
            _write_detector_file(
                water_transmission_path,
                sample_name="water",
                data=np.ones((2, 2), dtype=np.float64),
            )

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.add_run(
                RunKey(config_id="cfg_source", entity="water", mode="scattering", sample_name="water"),
                water_scattering_path,
            )
            ctx.add_run(
                RunKey(config_id="cfg_source", entity="water", mode="transmission", sample_name="water"),
                water_transmission_path,
            )
            ctx.set_transmission("water", "cfg_source", 1.0)
            ctx.set_flatfield_source("cfg_target", "cfg_source")

            flatfield_path = ctx.build_water_flatfield("cfg_target")

            self.assertEqual(flatfield_path, (ctx.output_dir / "flatfield_cfg_source.nxs").resolve())
            self.assertEqual(ctx.get_flatfield("cfg_target"), flatfield_path)
            expected_flatfield = normalize_by_solid_angle(
                np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
                detector_distance=4.2,
                beam_center=(0.5, 0.5),
                pixel_size=(0.001, 0.001),
            )
            np.testing.assert_allclose(
                load_flatfield_file(flatfield_path)[0].data.values,
                expected_flatfield,
            )


if __name__ == "__main__":
    unittest.main()
