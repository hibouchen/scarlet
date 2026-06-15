from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

from scarlet.workflow.configuration import Aperture, Collimation, Configuration
from scarlet.workflow.context import (
    RunKey,
    WorkflowContext,
    generate_reference_files_from_workflow_context,
    integrate_scattering_from_workflow_context,
    initialize_workflow_context_from_raw_directory,
    load_workflow_context,
    run_reduction_pipeline_from_workflow_context,
    save_workflow_context,
    update_reference_masks_from_workflow_context,
    update_workflow_context_from_raw_directory,
    update_transmissions_from_workflow_context,
    update_workflow_context_from_runs_report_csv,
    write_runs_report_csv,
)
from scarlet.workflow.reference import write_refs_norm_file, write_refs_sub_file

from test_workflow_configuration import _write_minimal_masks_file, _write_minimal_raw_nexus_file


def _write_transmission_file(
    path: Path,
    *,
    data: np.ndarray,
    monitor_integral: float,
    entry_name: str = "entry",
    dead_time_s: float | None = None,
) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group(entry_name)
        entry.attrs["NX_class"] = b"NXentry"
        control = entry.create_group("control")
        control.attrs["NX_class"] = b"NXmonitor"
        control.create_dataset("integral", data=float(monitor_integral))
        control.create_dataset("count_time", data=1.0)
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = b"NXinstrument"
        detector = instrument.create_group("detector0")
        detector.attrs["NX_class"] = b"NXdetector"
        detector.create_dataset("data", data=np.asarray(data, dtype=np.float64))
        if dead_time_s is not None:
            detector.create_dataset("dead_time", data=float(dead_time_s))


def _write_scattering_file(
    path: Path,
    *,
    data: np.ndarray,
    monitor_integral: float,
    pixel_size_m: tuple[float, float] = (0.0005, 0.0005),
    extra_detectors: dict[int, np.ndarray] | None = None,
) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = b"NXentry"
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = b"NXsample"
        sample.create_dataset("name", data=np.bytes_(path.stem))
        control = entry.create_group("control")
        control.attrs["NX_class"] = b"NXmonitor"
        control.create_dataset("integral", data=float(monitor_integral))
        control.create_dataset("count_time", data=1.0)
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = b"NXinstrument"
        detector = instrument.create_group("detector0")
        detector.attrs["NX_class"] = b"NXdetector"
        detector.create_dataset("data", data=np.asarray(data, dtype=np.float64))
        detector.create_dataset("x_pixel_size", data=float(pixel_size_m[0]))
        detector.create_dataset("y_pixel_size", data=float(pixel_size_m[1]))
        for detector_number, detector_data in sorted((extra_detectors or {}).items()):
            extra_detector = instrument.create_group(f"detector{int(detector_number)}")
            extra_detector.attrs["NX_class"] = b"NXdetector"
            extra_detector.create_dataset("data", data=np.asarray(detector_data, dtype=np.float64))
            extra_detector.create_dataset("x_pixel_size", data=float(pixel_size_m[0]))
            extra_detector.create_dataset("y_pixel_size", data=float(pixel_size_m[1]))


def _write_constant_water_corrected(refs_norm_path: Path, *, values: dict[int, float]) -> None:
    with h5py.File(refs_norm_path, "r+") as f:
        references = f["/entry/references"]
        if "water_corrected" in references:
            del references["water_corrected"]
        f.copy("/entry/references/water_scattering", references, name="water_corrected")
        corrected_entry = f["/entry/references/water_corrected/entry"]
        control = corrected_entry["control"]
        if "integral" in control:
            del control["integral"]
        control.create_dataset("integral", data=1.0)
        for detector_number, value in sorted(values.items()):
            detector = corrected_entry[f"instrument/detector{detector_number}"]
            shape = detector["data"].shape
            del detector["data"]
            detector.create_dataset("data", data=np.full(shape, float(value), dtype=np.float64))


class TestWorkflowContextRunsReport(unittest.TestCase):
    def test_update_workflow_context_from_runs_report_csv_applies_manual_edits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_dir = root / "out"
            output_dir.mkdir()

            sample_path = root / "sample.nxs"
            dark_path = root / "dark.nxs"
            _write_minimal_raw_nexus_file(sample_path, sample_name="sample_a", count_time_s=10.0)
            _write_minimal_raw_nexus_file(dark_path, sample_name="B4C", count_time_s=5.0)

            ctx = WorkflowContext(output_dir=output_dir)
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="dark", mode="scattering", sample_name=None),
                dark_path,
            )
            ctx.set_mask("config_1", 0, np.zeros((3, 4), dtype=np.uint8))
            ctx.set_transmission("sample_a", "config_1", 0.9)
            ctx.set_refs_sub("config_1", output_dir / "refs_sub_config_1.nxs")
            ctx.set_refs_norm("config_1", output_dir / "refs_norm_config_1.nxs")

            csv_path = write_runs_report_csv(ctx, overwrite=True)

            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            sample_row = next(row for row in rows if row["entity"] == "sample")
            edited_rows = [
                {
                    "sample_name": "water",
                    "config_id": "config_2",
                    "mode": "transmission",
                    "entity": "sample",
                    "transmission": "0.82",
                    "file_path": sample_row["file_path"],
                }
            ]

            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_name", "config_id", "mode", "entity", "transmission", "file_path"],
                )
                writer.writeheader()
                writer.writerows(edited_rows)

            update_workflow_context_from_runs_report_csv(ctx, csv_path)

            self.assertEqual(len(ctx.runs), 1)
            (run_key, run_path), = ctx.runs.items()
            self.assertEqual(run_key.config_id, "config_2")
            self.assertEqual(run_key.entity, "sample")
            self.assertEqual(run_key.mode, "transmission")
            self.assertEqual(run_key.sample_name, "water")
            self.assertAlmostEqual(run_key.transmission or 0.0, 0.82)
            self.assertEqual(run_path, sample_path.resolve())

            self.assertEqual(set(ctx.configurations), {"config_2"})
            self.assertEqual(ctx.configurations["config_2"].config_id, "config_2")
            self.assertEqual(ctx.refs_sub_files, {})
            self.assertEqual(ctx.refs_norm_files, {})
            self.assertEqual(ctx.transmissions, {})
            self.assertEqual(ctx.masks, {})

    def test_update_workflow_context_from_runs_report_csv_preserves_non_sample_names(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_dir = root / "out"
            output_dir.mkdir()

            empty_cell_path = root / "empty_cell.nxs"
            _write_minimal_raw_nexus_file(empty_cell_path, sample_name="EmptyCell42", count_time_s=10.0)

            ctx = WorkflowContext(output_dir=output_dir)
            ctx.add_run(
                RunKey(
                    config_id="config_1",
                    entity="empty_cell",
                    mode="transmission",
                    sample_name="EmptyCell42",
                ),
                empty_cell_path,
            )

            csv_path = write_runs_report_csv(ctx, overwrite=True)
            update_workflow_context_from_runs_report_csv(ctx, csv_path)

            self.assertEqual(len(ctx.runs), 1)
            (run_key, run_path), = ctx.runs.items()
            self.assertEqual(run_key.entity, "empty_cell")
            self.assertEqual(run_key.mode, "transmission")
            self.assertEqual(run_key.sample_name, "EmptyCell42")
            self.assertEqual(run_path, empty_cell_path.resolve())

    def test_runs_table_uses_same_columns_as_runs_report_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample.nxs"
            dark_path = root / "dark.nxs"
            _write_minimal_raw_nexus_file(sample_path, sample_name="sample_a", count_time_s=10.0)
            _write_minimal_raw_nexus_file(dark_path, sample_name="B4C", count_time_s=5.0)

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="dark", mode="scattering", sample_name="B4C"),
                dark_path,
            )

            table = ctx.runs_table()

            self.assertEqual(
                table.columns,
                ("sample_name", "config_id", "mode", "entity", "transmission", "file_path"),
            )
            self.assertEqual(len(table.rows), 2)
            self.assertEqual(table.rows[0]["sample_name"], "sample_a")
            self.assertEqual(table.rows[0]["config_id"], "config_1")
            self.assertEqual(table.rows[0]["mode"], "scattering")
            self.assertEqual(table.rows[0]["entity"], "sample")
            self.assertEqual(table.rows[0]["transmission"], "")
            self.assertEqual(table.rows[0]["file_path"], str(sample_path.resolve()))
            self.assertIn("<table>", table._repr_html_())

    def test_configurations_table_displays_configuration_properties(self) -> None:
        ctx = WorkflowContext()
        ctx.configurations["config_1"] = Configuration(
            wavelength=6.0,
            sample_detector_distance=4.2,
            config_id="config_1",
            notes="baseline",
            collimation=Collimation(
                aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
                aperture2=Aperture(type="pinhole", diameter=0.004),
                collimation_distance=1.5,
                last_aperture_to_sample_distance=0.5,
            ),
        )

        table = ctx.configurations_table()

        self.assertEqual(table.rows[0]["config_id"], "config_1")
        self.assertEqual(table.rows[0]["wavelength"], "6")
        self.assertEqual(table.rows[0]["sample_detector_distance"], "4.2")
        self.assertEqual(table.rows[0]["notes"], "baseline")
        self.assertEqual(table.rows[0]["has_collimation"], "True")
        self.assertEqual(table.rows[0]["collimation_distance"], "1.5")
        self.assertEqual(table.rows[0]["last_aperture_to_sample_distance"], "0.5")
        self.assertEqual(table.rows[0]["aperture1_type"], "slit")
        self.assertEqual(table.rows[0]["aperture1_x_gap"], "0.002")
        self.assertEqual(table.rows[0]["aperture1_y_gap"], "0.003")
        self.assertEqual(table.rows[0]["aperture2_type"], "pinhole")
        self.assertEqual(table.rows[0]["aperture2_diameter"], "0.004")
        self.assertIn("<table>", table._repr_html_())

    def test_transmissions_table_displays_sample_transmissions(self) -> None:
        ctx = WorkflowContext()
        ctx.set_transmission("polymer", "config_2", 0.75)
        ctx.set_transmission("water", "config_1", 0.5)

        table = ctx.transmissions_table()

        self.assertEqual(table.columns, ("sample_name", "config_id", "transmission"))
        self.assertEqual(len(table.rows), 2)
        self.assertEqual(table.rows[0]["sample_name"], "polymer")
        self.assertEqual(table.rows[0]["config_id"], "config_2")
        self.assertEqual(table.rows[0]["transmission"], "0.75")
        self.assertEqual(table.rows[1]["sample_name"], "water")
        self.assertEqual(table.rows[1]["config_id"], "config_1")
        self.assertEqual(table.rows[1]["transmission"], "0.5")
        self.assertIn("<table>", table._repr_html_())

    def test_generate_reference_files_from_workflow_context_refreshes_refs_properties(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_dir = root / "out"
            output_dir.mkdir()

            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            water_scattering = root / "water_scattering.nxs"
            water_transmission = root / "water_transmission.nxs"
            _write_minimal_raw_nexus_file(empty_beam_transmission, sample_name="empty_beam_open", count_time_s=10.0)
            _write_minimal_raw_nexus_file(water_scattering, sample_name="water", count_time_s=8.0)
            _write_minimal_raw_nexus_file(water_transmission, sample_name="water", count_time_s=9.0)

            ctx = WorkflowContext(output_dir=output_dir)
            ctx.add_run(
                RunKey(config_id="config_1", entity="empty_beam", mode="transmission", sample_name="empty_beam_open"),
                empty_beam_transmission,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="water"),
                water_scattering,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="transmission", sample_name="water"),
                water_transmission,
            )

            ctx.set_refs_sub("stale_config", output_dir / "refs_sub_stale_config.nxs")
            ctx.set_refs_norm("stale_config", output_dir / "refs_norm_stale_config.nxs")

            out = generate_reference_files_from_workflow_context(ctx)

            self.assertIs(out, ctx)
            self.assertEqual(set(ctx.refs_sub_files), {"config_1"})
            self.assertEqual(set(ctx.refs_norm_files), {"config_1"})
            self.assertTrue(ctx.get_refs_sub_path("config_1").exists())
            self.assertTrue(ctx.get_refs_norm_path("config_1").exists())

    def test_update_reference_masks_from_workflow_context_updates_matching_refs_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_dir = root / "out"
            output_dir.mkdir()

            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            water_scattering = root / "water_scattering.nxs"
            water_transmission = root / "water_transmission.nxs"
            _write_minimal_raw_nexus_file(empty_beam_transmission, sample_name="empty_beam_open", count_time_s=10.0)
            _write_minimal_raw_nexus_file(water_scattering, sample_name="water", count_time_s=8.0)
            _write_minimal_raw_nexus_file(water_transmission, sample_name="water", count_time_s=9.0)

            ctx = WorkflowContext(output_dir=output_dir)
            ctx.add_run(
                RunKey(config_id="config_1", entity="empty_beam", mode="transmission", sample_name="empty_beam_open"),
                empty_beam_transmission,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="water"),
                water_scattering,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="transmission", sample_name="water"),
                water_transmission,
            )

            generate_reference_files_from_workflow_context(ctx)

            mask0 = np.zeros((7, 7), dtype=np.uint8)
            mask0[1, 2] = 1
            mask_file = output_dir / "config_1_masks.nxs"
            _write_minimal_masks_file(mask_file, masks={0: mask0}, config_id="config_1")

            out = update_reference_masks_from_workflow_context(ctx)

            self.assertIs(out, ctx)
            self.assertEqual(ctx.get_masks_file_path("config_1"), mask_file.resolve())
            np.testing.assert_array_equal(ctx.get_mask("config_1", 0), mask0)

            with h5py.File(ctx.get_refs_sub_path("config_1"), "r") as f:
                np.testing.assert_array_equal(f["/entry/mask/mask_detector0"][()], mask0)
            with h5py.File(ctx.get_refs_norm_path("config_1"), "r") as f:
                np.testing.assert_array_equal(f["/entry/mask/mask_detector0"][()], mask0)

    def test_update_reference_masks_from_workflow_context_matches_legacy_scalar_distance_masks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            refs_dir = root / "refs"
            refs_dir.mkdir()
            output_dir = root / "out"
            output_dir.mkdir()

            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            water_scattering = root / "water_scattering.nxs"
            water_transmission = root / "water_transmission.nxs"
            _write_minimal_raw_nexus_file(
                empty_beam_transmission,
                sample_name="empty_beam_open",
                sample_detector_distance_m=4.2,
                beam_centers={0: (3.0, 3.0), 1: (3.0, 3.0), 2: (3.0, 3.0)},
            )
            _write_minimal_raw_nexus_file(
                water_scattering,
                sample_name="water",
                sample_detector_distance_m=4.2,
                beam_centers={0: (3.0, 3.0), 1: (3.0, 3.0), 2: (3.0, 3.0)},
            )
            _write_minimal_raw_nexus_file(
                water_transmission,
                sample_name="water",
                sample_detector_distance_m=4.2,
                beam_centers={0: (3.0, 3.0), 1: (3.0, 3.0), 2: (3.0, 3.0)},
            )

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=[4.2, 1.1, 1.2],
                config_id="config_1",
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
                    aperture2=Aperture(type="pinhole", diameter=0.004),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )

            refs_sub_path = write_refs_sub_file(
                refs_dir / "refs_sub_config_1.nxs",
                configuration,
                empty_beam_transmission=empty_beam_transmission,
                transmission_roi_detector=0,
                transmission_roi=(1, 4, 1, 4),
                beam_centers={0: (3.0, 3.0), 1: (3.0, 3.0), 2: (3.0, 3.0)},
                overwrite=True,
            ).resolve()
            refs_norm_path = write_refs_norm_file(
                refs_dir / "refs_norm_config_1.nxs",
                configuration,
                water_scattering=water_scattering,
                water_transmission=water_transmission,
                empty_beam_transmission=empty_beam_transmission,
                transmission_roi_detector=0,
                transmission_roi=(1, 4, 1, 4),
                overwrite=True,
            ).resolve()

            mask0 = np.zeros((7, 7), dtype=np.uint8)
            mask1 = np.zeros((7, 7), dtype=np.uint8)
            mask0[1, 2] = 1
            mask1[2, 3] = 1
            legacy_mask_path = output_dir / "masks_legacy.nxs"
            with h5py.File(legacy_mask_path, "w") as f:
                entry = f.create_group("entry")
                entry.attrs["NX_class"] = b"NXentry"
                entry.create_dataset("definition", data=np.bytes_("SCARLET_masks"))
                entry.create_dataset("schema_version", data=np.bytes_("1.0"))
                cfg = entry.create_group("configuration")
                cfg.attrs["NX_class"] = b"NXcollection"
                cfg.create_dataset("wavelength", data=6.0)
                cfg.create_dataset("sample_detector_distance", data=4.2)
                col = cfg.create_group("collimation")
                col.attrs["NX_class"] = b"NXcollection"
                ap1 = col.create_group("aperture1")
                ap1.attrs["NX_class"] = b"NXaperture"
                ap1.create_dataset("type", data=np.bytes_("slit"))
                ap1.create_dataset("x_gap", data=0.002)
                ap1.create_dataset("y_gap", data=0.003)
                ap2 = col.create_group("aperture2")
                ap2.attrs["NX_class"] = b"NXaperture"
                ap2.create_dataset("type", data=np.bytes_("pinhole"))
                ap2.create_dataset("diameter", data=0.004)
                col.create_dataset("collimation_distance", data=1.5)
                col.create_dataset("last_aperture_to_sample_distance", data=0.5)
                mask_group = entry.create_group("mask")
                mask_group.attrs["NX_class"] = b"NXcollection"
                mask_group.create_dataset("mask_detector0", data=mask0)
                mask_group.create_dataset("mask_detector1", data=mask1)
                meta = entry.create_group("meta")
                meta.attrs["NX_class"] = b"NXcollection"
                meta.create_dataset("mask_convention", data=np.bytes_("1=masked, 0=valid"))

            ctx = WorkflowContext(output_dir=output_dir)
            ctx.configurations["config_1"] = configuration
            ctx.set_refs_sub("config_1", refs_sub_path)
            ctx.set_refs_norm("config_1", refs_norm_path)

            out = update_reference_masks_from_workflow_context(ctx)

            self.assertIs(out, ctx)
            self.assertEqual(ctx.get_masks_file_path("config_1"), legacy_mask_path.resolve())
            np.testing.assert_array_equal(ctx.get_mask("config_1", 0), mask0)
            np.testing.assert_array_equal(ctx.get_mask("config_1", 1), mask1)
            with h5py.File(refs_sub_path, "r") as f:
                np.testing.assert_array_equal(f["/entry/mask/mask_detector0"][()], mask0)
                np.testing.assert_array_equal(f["/entry/mask/mask_detector1"][()], mask1)
            with h5py.File(refs_norm_path, "r") as f:
                np.testing.assert_array_equal(f["/entry/mask/mask_detector0"][()], mask0)
                np.testing.assert_array_equal(f["/entry/mask/mask_detector1"][()], mask1)

    def test_update_transmissions_from_workflow_context_uses_refs_sub_and_wavelength_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            refs_dir = root / "refs"
            refs_dir.mkdir()

            empty_beam_cfg1 = root / "empty_beam_cfg1.nxs"
            empty_beam_cfg2 = root / "empty_beam_cfg2.nxs"
            sample_transmission_cfg1 = root / "sample_transmission_cfg1.nxs"
            sample_scattering_cfg1 = root / "sample_scattering_cfg1.nxs"
            sample_scattering_cfg2 = root / "sample_scattering_cfg2.nxs"

            empty_beam_data = np.zeros((4, 4), dtype=np.float64)
            empty_beam_data[1:3, 1:3] = 100.0
            sample_transmission_data = np.zeros((4, 4), dtype=np.float64)
            sample_transmission_data[1:3, 1:3] = 50.0
            sample_scattering_data = np.zeros((4, 4), dtype=np.float64)
            sample_scattering_data[1:3, 1:3] = 10.0

            _write_transmission_file(empty_beam_cfg1, data=empty_beam_data, monitor_integral=10.0)
            _write_transmission_file(empty_beam_cfg2, data=empty_beam_data, monitor_integral=10.0)
            _write_transmission_file(sample_transmission_cfg1, data=sample_transmission_data, monitor_integral=10.0)
            _write_transmission_file(sample_scattering_cfg1, data=sample_scattering_data, monitor_integral=10.0)
            _write_transmission_file(sample_scattering_cfg2, data=sample_scattering_data, monitor_integral=10.0)

            configuration_1 = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="config_1",
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
                    aperture2=Aperture(type="pinhole", diameter=0.004),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )
            configuration_2 = Configuration(
                wavelength=6.0,
                sample_detector_distance=1.2,
                config_id="config_2",
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
                    aperture2=Aperture(type="pinhole", diameter=0.004),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )

            refs_sub_cfg1 = write_refs_sub_file(
                refs_dir / "refs_sub_config_1.nxs",
                configuration_1,
                empty_beam_transmission=empty_beam_cfg1,
                transmission_roi_detector=0,
                transmission_roi=(1, 3, 1, 3),
                beam_centers={0: (1.0, 1.0)},
                overwrite=True,
            ).resolve()
            refs_sub_cfg2 = write_refs_sub_file(
                refs_dir / "refs_sub_config_2.nxs",
                configuration_2,
                empty_beam_transmission=empty_beam_cfg2,
                transmission_roi_detector=0,
                transmission_roi=(1, 3, 1, 3),
                beam_centers={0: (1.0, 1.0)},
                overwrite=True,
            ).resolve()

            ctx = WorkflowContext(root_dir=root, output_dir=root / "out")
            ctx.configurations["config_1"] = configuration_1
            ctx.configurations["config_2"] = configuration_2
            ctx.set_refs_sub("config_1", refs_sub_cfg1)
            ctx.set_refs_sub("config_2", refs_sub_cfg2)
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="polymer"),
                sample_scattering_cfg1,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="transmission", sample_name="polymer"),
                sample_transmission_cfg1,
            )
            ctx.add_run(
                RunKey(config_id="config_2", entity="sample", mode="scattering", sample_name="polymer"),
                sample_scattering_cfg2,
            )

            out = update_transmissions_from_workflow_context(ctx)

            self.assertIs(out, ctx)
            self.assertAlmostEqual(ctx.get_transmission("polymer", "config_1") or 0.0, 0.5)
            self.assertAlmostEqual(ctx.get_transmission("polymer", "config_2") or 0.0, 0.5)

    def test_integrate_scattering_from_workflow_context_writes_four_column_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_dir = root / "out"
            output_dir.mkdir()
            refs_dir = root / "refs"
            refs_dir.mkdir()

            sample_scattering = root / "sample_scattering.nxs"
            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            empty_cell_transmission = root / "empty_cell_transmission.nxs"
            empty_cell_scattering = root / "empty_cell_scattering.nxs"
            water_scattering = root / "water_scattering.nxs"
            water_transmission = root / "water_transmission.nxs"
            dark = root / "dark.nxs"

            scattering_data = np.zeros((5, 5), dtype=np.float64)
            scattering_data[2, 2] = 100.0
            scattering_data[1:4, 1:4] += 10.0
            detector1_data = np.zeros((5, 5), dtype=np.float64)
            detector1_data[2, 1:4] = 40.0
            _write_scattering_file(
                sample_scattering,
                data=scattering_data,
                monitor_integral=20.0,
                extra_detectors={1: detector1_data},
            )

            reference_data = np.zeros((5, 5), dtype=np.float64)
            reference_data[2:4, 2:4] = 50.0
            _write_transmission_file(empty_beam_transmission, data=reference_data + 50.0, monitor_integral=10.0)
            _write_transmission_file(empty_cell_transmission, data=reference_data + 25.0, monitor_integral=10.0)
            _write_transmission_file(
                empty_cell_scattering,
                data=np.full((5, 5), 5.0, dtype=np.float64),
                monitor_integral=5.0,
                dead_time_s=1.2e-6,
            )
            _write_transmission_file(water_scattering, data=np.full((5, 5), 8.0, dtype=np.float64), monitor_integral=4.0)
            _write_transmission_file(water_transmission, data=np.full((5, 5), 8.0, dtype=np.float64), monitor_integral=4.0)
            _write_transmission_file(dark, data=np.full((5, 5), 1.0, dtype=np.float64), monitor_integral=5.0)

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="config_1",
                collimation=Collimation(
                    aperture1=Aperture(type="pinhole", diameter=0.002),
                    aperture2=Aperture(type="pinhole", diameter=0.001),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )

            refs_sub_path = write_refs_sub_file(
                refs_dir / "refs_sub_config_1.nxs",
                configuration,
                empty_beam_transmission=empty_beam_transmission,
                dark=dark,
                empty_cell_transmission=empty_cell_transmission,
                empty_cell_scattering=empty_cell_scattering,
                transmission_roi_detector=0,
                transmission_roi=(1, 4, 1, 4),
                beam_centers={0: (2.0, 2.0), 1: (2.0, 2.0)},
                overwrite=True,
            ).resolve()
            refs_norm_path = write_refs_norm_file(
                refs_dir / "refs_norm_config_1.nxs",
                configuration,
                water_scattering=water_scattering,
                water_transmission=water_transmission,
                dark=dark,
                empty_beam_transmission=empty_beam_transmission,
                empty_cell_scattering=empty_cell_scattering,
                transmission_roi_detector=0,
                transmission_roi=(1, 4, 1, 4),
                masks={0: np.zeros((5, 5), dtype=np.uint8), 1: np.zeros((5, 5), dtype=np.uint8)},
                overwrite=True,
            ).resolve()
            with h5py.File(refs_norm_path, "r+") as f:
                for reference_name, value in (
                    ("water_scattering", 8.0),
                    ("dark", 1.0),
                    ("empty_cell_scattering", 5.0),
                ):
                    if f"/entry/references/{reference_name}/entry/instrument/detector1" in f:
                        continue
                    instrument = f[f"/entry/references/{reference_name}/entry/instrument"]
                    det1 = instrument.create_group("detector1")
                    det1.attrs["NX_class"] = b"NXdetector"
                    det1.create_dataset("data", data=np.full((5, 5), value, dtype=np.float64))
                if "/entry/references/empty_beam_transmission/entry/instrument/detector1" not in f:
                    instrument = f["/entry/references/empty_beam_transmission/entry/instrument"]
                    det1 = instrument.create_group("detector1")
                    det1.attrs["NX_class"] = b"NXdetector"
                    det1.create_dataset("data", data=np.full((5, 5), 10.0, dtype=np.float64))
            _write_constant_water_corrected(refs_norm_path, values={0: 1.0, 1: 1.0})

            ctx = WorkflowContext(root_dir=root, output_dir=output_dir)
            ctx.configurations["config_1"] = configuration
            ctx.set_refs_sub("config_1", refs_sub_path)
            ctx.set_refs_norm("config_1", refs_norm_path)
            ctx.set_transmission("sampleA", "config_1", 0.5)
            ctx.set_transmission("empty_cell_A", "config_1", 0.8)
            ctx.set("wavelength_spread", 0.6)
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="sampleA"),
                sample_scattering,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="empty_cell", mode="transmission", sample_name="empty_cell_A"),
                empty_cell_transmission,
            )

            out = integrate_scattering_from_workflow_context(ctx, n_bins=8)

            self.assertIs(out, ctx)
            generated0 = output_dir / "sampleA_det0_config_1.txt"
            generated1 = output_dir / "sampleA_det1_config_1.txt"
            self.assertTrue(generated0.exists())
            self.assertTrue(generated1.exists())

            content0 = generated0.read_text(encoding="utf-8")
            self.assertIn("sample_name: sampleA", content0)
            self.assertIn("config_id: config_1", content0)
            self.assertIn("detector: detector0", content0)
            self.assertIn("transmission: 0.5", content0)

            content1 = generated1.read_text(encoding="utf-8")
            self.assertIn("detector: detector1", content1)

            data0 = np.loadtxt(generated0)
            data1 = np.loadtxt(generated1)
            self.assertEqual(data0.shape[1], 4)
            self.assertEqual(data1.shape[1], 4)
            self.assertTrue(np.all(np.isfinite(data0)))
            self.assertTrue(np.all(np.isfinite(data1)))

            baseline0 = data0.copy()
            baseline1 = data1.copy()

            _write_constant_water_corrected(refs_norm_path, values={0: 2.0, 1: 2.0})
            integrate_scattering_from_workflow_context(ctx, n_bins=8)
            divided0 = np.loadtxt(generated0)
            divided1 = np.loadtxt(generated1)
            np.testing.assert_allclose(divided0[:, 1], baseline0[:, 1] / 2.0)
            np.testing.assert_allclose(divided1[:, 1], baseline1[:, 1] / 2.0)
            np.testing.assert_allclose(divided0[:, 2], baseline0[:, 2] / 2.0)
            np.testing.assert_allclose(divided1[:, 2], baseline1[:, 2] / 2.0)

    def test_run_reduction_pipeline_from_workflow_context_returns_states(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_dir = root / "out"
            output_dir.mkdir()
            refs_dir = root / "refs"
            refs_dir.mkdir()

            sample_scattering = root / "sample_scattering.nxs"
            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            empty_cell_transmission = root / "empty_cell_transmission.nxs"
            empty_cell_scattering = root / "empty_cell_scattering.nxs"
            water_scattering = root / "water_scattering.nxs"
            water_transmission = root / "water_transmission.nxs"
            dark = root / "dark.nxs"

            scattering_data = np.zeros((5, 5), dtype=np.float64)
            scattering_data[2, 2] = 100.0
            scattering_data[1:4, 1:4] += 10.0
            _write_scattering_file(sample_scattering, data=scattering_data, monitor_integral=20.0)

            reference_data = np.zeros((5, 5), dtype=np.float64)
            reference_data[2:4, 2:4] = 50.0
            _write_transmission_file(empty_beam_transmission, data=reference_data + 50.0, monitor_integral=10.0)
            _write_transmission_file(empty_cell_transmission, data=reference_data + 25.0, monitor_integral=10.0)
            _write_transmission_file(
                empty_cell_scattering,
                data=np.full((5, 5), 5.0, dtype=np.float64),
                monitor_integral=5.0,
                dead_time_s=1.2e-6,
            )
            _write_transmission_file(water_scattering, data=np.full((5, 5), 8.0, dtype=np.float64), monitor_integral=4.0)
            _write_transmission_file(water_transmission, data=np.full((5, 5), 8.0, dtype=np.float64), monitor_integral=4.0)
            _write_transmission_file(dark, data=np.full((5, 5), 1.0, dtype=np.float64), monitor_integral=5.0)

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="config_1",
                collimation=Collimation(
                    aperture1=Aperture(type="pinhole", diameter=0.002),
                    aperture2=Aperture(type="pinhole", diameter=0.001),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )

            refs_sub_path = write_refs_sub_file(
                refs_dir / "refs_sub_config_1.nxs",
                configuration,
                empty_beam_transmission=empty_beam_transmission,
                dark=dark,
                empty_cell_transmission=empty_cell_transmission,
                empty_cell_scattering=empty_cell_scattering,
                transmission_roi_detector=0,
                transmission_roi=(1, 4, 1, 4),
                beam_centers={0: (2.0, 2.0)},
                overwrite=True,
            ).resolve()
            refs_norm_path = write_refs_norm_file(
                refs_dir / "refs_norm_config_1.nxs",
                configuration,
                water_scattering=water_scattering,
                water_transmission=water_transmission,
                dark=dark,
                empty_beam_transmission=empty_beam_transmission,
                empty_cell_scattering=empty_cell_scattering,
                transmission_roi_detector=0,
                transmission_roi=(1, 4, 1, 4),
                masks={0: np.zeros((5, 5), dtype=np.uint8)},
                overwrite=True,
            ).resolve()
            _write_constant_water_corrected(refs_norm_path, values={0: 1.0})

            ctx = WorkflowContext(root_dir=root, output_dir=output_dir)
            ctx.configurations["config_1"] = configuration
            ctx.set_refs_sub("config_1", refs_sub_path)
            ctx.set_refs_norm("config_1", refs_norm_path)
            ctx.set_transmission("sampleA", "config_1", 0.5)
            ctx.set_transmission("empty_cell_A", "config_1", 0.8)
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="sampleA"),
                sample_scattering,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="empty_cell", mode="transmission", sample_name="empty_cell_A"),
                empty_cell_transmission,
            )

            state = run_reduction_pipeline_from_workflow_context(
                ctx,
                n_bins=8,
                sample_name="sampleA",
                config_id="config_1",
                detector_number=0,
            )

            self.assertIsNotNone(state.corrected)
            self.assertIsNotNone(state.normalized_image)
            self.assertIsNotNone(state.solid_angle_corrected)
            self.assertIsNotNone(state.q_map)
            self.assertIsNotNone(state.integration)
            self.assertIsNotNone(state.inputs.empty_cell)
            assert state.inputs.empty_cell is not None
            self.assertAlmostEqual(state.inputs.empty_cell.transmission, 0.8)
            self.assertAlmostEqual(state.inputs.empty_cell.monitor, 5.0)
            self.assertAlmostEqual(state.inputs.empty_cell.acquisition_time, 1.0)
            self.assertAlmostEqual(state.inputs.empty_cell.deadtime or 0.0, 1.2e-6)
            np.testing.assert_allclose(
                state.inputs.empty_cell.deadtime_corrected_image,
                np.full((5, 5), 1.0 / (1.0 - 6.0e-6), dtype=np.float64),
            )
            np.testing.assert_allclose(
                state.inputs.empty_cell.deadtime_corrected_error,
                np.full((5, 5), np.sqrt(5.0) / (5.0 * (1.0 - 6.0e-6) ** 2), dtype=np.float64),
            )

    def test_integrate_scattering_from_workflow_context_can_filter_single_sample(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_dir = root / "out"
            output_dir.mkdir()
            refs_dir = root / "refs"
            refs_dir.mkdir()

            sample_a_scattering = root / "sample_a_scattering.nxs"
            sample_b_scattering = root / "sample_b_scattering.nxs"
            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            empty_cell_transmission = root / "empty_cell_transmission.nxs"
            empty_cell_scattering = root / "empty_cell_scattering.nxs"
            water_scattering = root / "water_scattering.nxs"
            water_transmission = root / "water_transmission.nxs"
            dark = root / "dark.nxs"

            scattering_data = np.zeros((5, 5), dtype=np.float64)
            scattering_data[2, 2] = 100.0
            scattering_data[1:4, 1:4] += 10.0
            _write_scattering_file(sample_a_scattering, data=scattering_data, monitor_integral=20.0)
            _write_scattering_file(sample_b_scattering, data=scattering_data * 0.5, monitor_integral=20.0)

            reference_data = np.zeros((5, 5), dtype=np.float64)
            reference_data[2:4, 2:4] = 50.0
            _write_transmission_file(empty_beam_transmission, data=reference_data + 50.0, monitor_integral=10.0)
            _write_transmission_file(empty_cell_transmission, data=reference_data + 25.0, monitor_integral=10.0)
            _write_transmission_file(empty_cell_scattering, data=np.full((5, 5), 5.0, dtype=np.float64), monitor_integral=5.0)
            _write_transmission_file(water_scattering, data=np.full((5, 5), 8.0, dtype=np.float64), monitor_integral=4.0)
            _write_transmission_file(water_transmission, data=np.full((5, 5), 8.0, dtype=np.float64), monitor_integral=4.0)
            _write_transmission_file(dark, data=np.full((5, 5), 1.0, dtype=np.float64), monitor_integral=5.0)

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="config_1",
                collimation=Collimation(
                    aperture1=Aperture(type="pinhole", diameter=0.002),
                    aperture2=Aperture(type="pinhole", diameter=0.001),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )

            refs_sub_path = write_refs_sub_file(
                refs_dir / "refs_sub_config_1.nxs",
                configuration,
                empty_beam_transmission=empty_beam_transmission,
                dark=dark,
                empty_cell_transmission=empty_cell_transmission,
                empty_cell_scattering=empty_cell_scattering,
                transmission_roi_detector=0,
                transmission_roi=(1, 4, 1, 4),
                beam_centers={0: (2.0, 2.0)},
                overwrite=True,
            ).resolve()
            refs_norm_path = write_refs_norm_file(
                refs_dir / "refs_norm_config_1.nxs",
                configuration,
                water_scattering=water_scattering,
                water_transmission=water_transmission,
                dark=dark,
                empty_beam_transmission=empty_beam_transmission,
                empty_cell_scattering=empty_cell_scattering,
                transmission_roi_detector=0,
                transmission_roi=(1, 4, 1, 4),
                masks={0: np.zeros((5, 5), dtype=np.uint8)},
                overwrite=True,
            ).resolve()
            _write_constant_water_corrected(refs_norm_path, values={0: 1.0})

            ctx = WorkflowContext(root_dir=root, output_dir=output_dir)
            ctx.configurations["config_1"] = configuration
            ctx.set_refs_sub("config_1", refs_sub_path)
            ctx.set_refs_norm("config_1", refs_norm_path)
            ctx.set_transmission("sampleA", "config_1", 0.5)
            ctx.set_transmission("sampleB", "config_1", 0.5)
            ctx.set_transmission("empty_cell_A", "config_1", 0.8)
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="sampleA"),
                sample_a_scattering,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="sampleB"),
                sample_b_scattering,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="empty_cell", mode="transmission", sample_name="empty_cell_A"),
                empty_cell_transmission,
            )

            out = integrate_scattering_from_workflow_context(ctx, n_bins=8, sample_name="sampleB")

            self.assertIs(out, ctx)
            self.assertFalse((output_dir / "sampleA_det0_config_1.txt").exists())
            self.assertTrue((output_dir / "sampleB_det0_config_1.txt").exists())

    def test_update_root_dir_rebases_registered_run_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            old_root = Path(td) / "raw_a"
            new_root = Path(td) / "raw_b"
            old_root.mkdir()
            new_root.mkdir()

            sample_path = old_root / "config_1" / "sample.nxs"
            sample_path.parent.mkdir()
            _write_minimal_raw_nexus_file(sample_path, sample_name="sample_a", count_time_s=10.0)

            external_path = Path(td) / "external_sample.nxs"
            _write_minimal_raw_nexus_file(external_path, sample_name="sample_b", count_time_s=10.0)

            ctx = WorkflowContext(root_dir=old_root, output_dir=Path(td) / "out")
            sample_key = RunKey(
                config_id="config_1",
                entity="sample",
                mode="scattering",
                sample_name="sample_a",
            )
            external_key = RunKey(
                config_id="config_1",
                entity="sample",
                mode="transmission",
                sample_name="sample_b",
            )
            ctx.add_run(sample_key, sample_path)
            ctx.add_run(external_key, external_path)

            out = ctx.update_root_dir(new_root)

            self.assertIs(out, ctx)
            self.assertEqual(ctx.root_dir, new_root.resolve())
            self.assertEqual(ctx.get_run_path(sample_key), (new_root / "config_1" / "sample.nxs").resolve())
            self.assertEqual(ctx.get_run_path(external_key), external_path.resolve())

    def test_update_output_dir_rebases_generated_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_output = root / "out_a"
            new_output = root / "out_b"
            old_output.mkdir()
            new_output.mkdir()

            ctx = WorkflowContext(root_dir=root, output_dir=old_output)
            ctx.set_refs_sub("config_1", old_output / "refs" / "refs_sub_config_1.nxs")
            ctx.set_refs_norm("config_1", old_output / "refs" / "refs_norm_config_1.nxs")
            ctx.set_masks_file("config_1", old_output / "masks" / "config_1_masks.nxs")
            ctx.add_artifact("runs_report.csv", old_output / "reports" / "runs_report.csv", kind="csv")
            ctx.add_artifact("external.txt", root / "notes.txt", kind="text")
            ctx.set("runs_report_csv", (old_output / "reports" / "runs_report.csv").resolve())

            out = ctx.update_output_dir(new_output)

            self.assertIs(out, ctx)
            self.assertEqual(ctx.output_dir, new_output.resolve())
            self.assertEqual(
                ctx.get_refs_sub_path("config_1"),
                (new_output / "refs" / "refs_sub_config_1.nxs").resolve(),
            )
            self.assertEqual(
                ctx.get_refs_norm_path("config_1"),
                (new_output / "refs" / "refs_norm_config_1.nxs").resolve(),
            )
            self.assertEqual(
                ctx.get_masks_file_path("config_1"),
                (new_output / "masks" / "config_1_masks.nxs").resolve(),
            )
            self.assertEqual(
                ctx.artifacts[0].path,
                (new_output / "reports" / "runs_report.csv").resolve(),
            )
            self.assertEqual(ctx.artifacts[1].path, (root / "notes.txt").resolve())
            self.assertEqual(
                ctx.get("runs_report_csv"),
                (new_output / "reports" / "runs_report.csv").resolve(),
            )

    def test_update_beam_center_updates_refs_sub_and_refs_norm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            refs_dir = root / "refs"
            refs_dir.mkdir()

            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            water_scattering = root / "water_scattering.nxs"
            water_transmission = root / "water_transmission.nxs"
            _write_transmission_file(empty_beam_transmission, data=np.full((5, 5), 9.0), monitor_integral=1.0)
            _write_transmission_file(water_scattering, data=np.full((5, 5), 8.0), monitor_integral=1.0)
            _write_transmission_file(water_transmission, data=np.full((5, 5), 7.0), monitor_integral=1.0)

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="config_1",
                collimation=Collimation(
                    aperture1=Aperture(type="pinhole", diameter=0.002),
                    aperture2=Aperture(type="pinhole", diameter=0.001),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )

            refs_sub_path = write_refs_sub_file(
                refs_dir / "refs_sub_config_1.nxs",
                configuration,
                empty_beam_transmission=empty_beam_transmission,
                transmission_roi_detector=0,
                transmission_roi=(1, 4, 1, 4),
                beam_centers={0: (2.0, 2.0)},
                overwrite=True,
            ).resolve()
            refs_norm_path = write_refs_norm_file(
                refs_dir / "refs_norm_config_1.nxs",
                configuration,
                water_scattering=water_scattering,
                water_transmission=water_transmission,
                empty_beam_transmission=empty_beam_transmission,
                transmission_roi_detector=0,
                transmission_roi=(1, 4, 1, 4),
                overwrite=True,
            ).resolve()

            ctx = WorkflowContext(root_dir=root, output_dir=root / "out")
            ctx.set_refs_sub("config_1", refs_sub_path)
            ctx.set_refs_norm("config_1", refs_norm_path)

            out = ctx.update_beam_center("config_1", 0, 12.5, 21.5)

            self.assertIs(out, ctx)
            with h5py.File(refs_sub_path, "r") as f:
                self.assertAlmostEqual(float(f["/entry/beam_center/detector0/beam_center_x"][()]), 12.5)
                self.assertAlmostEqual(float(f["/entry/beam_center/detector0/beam_center_y"][()]), 21.5)
            with h5py.File(refs_norm_path, "r") as f:
                self.assertAlmostEqual(float(f["/entry/beam_center/detector0/beam_center_x"][()]), 12.5)
                self.assertAlmostEqual(float(f["/entry/beam_center/detector0/beam_center_y"][()]), 21.5)

    def test_update_workflow_context_from_raw_directory_adds_new_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_dir = root / "raw"
            output_dir = root / "out"
            raw_dir.mkdir()
            output_dir.mkdir()

            raw1 = raw_dir / "sample_1.h5"
            raw2 = raw_dir / "sample_2.h5"
            _write_minimal_raw_nexus_file(raw1, sample_name="sample_a", count_time_s=10.0)

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="config_x",
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
                    aperture2=Aperture(type="pinhole", diameter=0.004),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )

            def fake_convert(_instrument_name, input_path, output_path, overwrite=False):
                del overwrite
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(Path(input_path).read_bytes())
                return SimpleNamespace(output_file=output_path)

            with (
                mock.patch("scarlet.io.converters.convert_to_scarlet_nxsas_raw", side_effect=fake_convert),
                mock.patch(
                    "scarlet.io.mode_inference.guess_measurement_mode_from_nexus_image",
                    return_value=SimpleNamespace(mode="scattering"),
                ),
                mock.patch(
                    "scarlet.workflow.configuration.configuration_from_nexus",
                    return_value=(configuration, []),
                ),
                mock.patch(
                    "scarlet.workflow.configuration.compare_configurations",
                    return_value=(True, []),
                ),
            ):
                ctx = WorkflowContext(
                    instrument_name="sansllb",
                    root_dir=raw_dir,
                    output_dir=output_dir,
                )

                out = update_workflow_context_from_raw_directory(ctx)

                self.assertIs(out, ctx)
                self.assertEqual(len(ctx.runs), 1)
                self.assertEqual(len(ctx.configurations), 1)
                self.assertEqual(
                    set(ctx.runs.values()),
                    {(output_dir / "sample_1.nxs").resolve()},
                )

                _write_minimal_raw_nexus_file(raw2, sample_name="sample_b", count_time_s=10.0)
                update_workflow_context_from_raw_directory(ctx)

                self.assertEqual(len(ctx.runs), 2)
                self.assertEqual(
                    set(ctx.runs.values()),
                    {
                        (output_dir / "sample_1.nxs").resolve(),
                        (output_dir / "sample_2.nxs").resolve(),
                    },
                )
                self.assertEqual(len(ctx.configurations), 1)

    def test_update_workflow_context_from_raw_directory_skips_auxiliary_hdf5_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_dir = root / "raw"
            output_dir = root / "out"
            raw_dir.mkdir()
            output_dir.mkdir()

            raw_path = raw_dir / "sample_1.h5"
            mask_path = raw_dir / "test_gui_mask.nxs"
            _write_minimal_raw_nexus_file(raw_path, sample_name="sample_a", count_time_s=10.0)

            with h5py.File(mask_path, "w") as f:
                entry = f.create_group("entry")
                entry.attrs["NX_class"] = b"NXentry"
                entry.create_dataset("definition", data=np.bytes_("SCARLET_masks"))

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="config_x",
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
                    aperture2=Aperture(type="pinhole", diameter=0.004),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )

            def fake_convert(_instrument_name, input_path, output_path, overwrite=False):
                del overwrite
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(Path(input_path).read_bytes())
                return SimpleNamespace(output_file=output_path)

            with (
                mock.patch("scarlet.io.converters.convert_to_scarlet_nxsas_raw", side_effect=fake_convert),
                mock.patch(
                    "scarlet.io.mode_inference.guess_measurement_mode_from_nexus_image",
                    return_value=SimpleNamespace(mode="scattering"),
                ),
                mock.patch(
                    "scarlet.workflow.configuration.configuration_from_nexus",
                    return_value=(configuration, []),
                ),
                mock.patch(
                    "scarlet.workflow.configuration.compare_configurations",
                    return_value=(True, []),
                ),
            ):
                ctx = WorkflowContext(
                    instrument_name="sansllb",
                    root_dir=raw_dir,
                    output_dir=output_dir,
                )

                out = update_workflow_context_from_raw_directory(ctx)

                self.assertIs(out, ctx)
                self.assertEqual(len(ctx.runs), 1)
                self.assertEqual(
                    set(ctx.runs.values()),
                    {(output_dir / "sample_1.nxs").resolve()},
                )
                self.assertTrue(
                    any(
                        issue.level == "WARN"
                        and issue.key == str(mask_path)
                        and "Skipping non-raw HDF5 input file" in issue.message
                        for issue in ctx.issues
                    )
                )

    def test_initialize_workflow_context_from_raw_directory_skips_non_2d_detector_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_dir = root / "raw"
            output_dir = root / "out"
            raw_dir.mkdir()

            raw_path = raw_dir / "sample_stack.h5"
            valid_path = raw_dir / "sample_2d.h5"
            with h5py.File(raw_path, "w") as f:
                entry = f.create_group("entry")
                entry.attrs["NX_class"] = b"NXentry"
                sample = entry.create_group("sample")
                sample.attrs["NX_class"] = b"NXsample"
                sample.create_dataset("name", data=np.bytes_("sample_a"))
                instrument = entry.create_group("instrument")
                instrument.attrs["NX_class"] = b"NXinstrument"
                detector = instrument.create_group("detector0")
                detector.attrs["NX_class"] = b"NXdetector"
                detector.create_dataset("data", data=np.zeros((3, 7, 7), dtype=np.float64))

            _write_minimal_raw_nexus_file(valid_path, sample_name="sample_b", count_time_s=10.0)

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="config_x",
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
                    aperture2=Aperture(type="pinhole", diameter=0.004),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )

            def fake_convert(_instrument_name, input_path, output_path, overwrite=False):
                del overwrite
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(Path(input_path).read_bytes())
                return SimpleNamespace(output_file=output_path)

            with (
                mock.patch("scarlet.io.converters.convert_to_scarlet_nxsas_raw", side_effect=fake_convert) as convert_mock,
                mock.patch(
                    "scarlet.io.mode_inference.guess_measurement_mode_from_nexus_image",
                    return_value=SimpleNamespace(mode="scattering"),
                ),
                mock.patch(
                    "scarlet.workflow.configuration.configuration_from_nexus",
                    return_value=(configuration, []),
                ),
                mock.patch(
                    "scarlet.workflow.configuration.compare_configurations",
                    return_value=(True, []),
                ),
            ):
                ctx = initialize_workflow_context_from_raw_directory(
                    raw_dir,
                    output_dir=output_dir,
                    instrument_name="sansllb",
                )

            convert_mock.assert_called_once()
            self.assertEqual(len(ctx.runs), 1)
            self.assertEqual(set(ctx.runs.values()), {(output_dir / "sample_2d.nxs").resolve()})
            self.assertEqual(len(ctx.configurations), 1)
            self.assertTrue(
                any(
                    issue.level == "WARN"
                    and issue.key == str(raw_path)
                    and "Skipping HDF5 input file with non-2D detector data" in issue.message
                    for issue in ctx.issues
                )
            )

    def test_save_and_load_workflow_context_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_dir = root / "out"
            output_dir.mkdir()
            context_path = root / "workflow_context.nxs"

            sample_path = root / "sample.nxs"
            dark_path = root / "dark.nxs"
            _write_minimal_raw_nexus_file(sample_path, sample_name="sample_a", count_time_s=10.0)
            _write_minimal_raw_nexus_file(dark_path, sample_name="B4C", count_time_s=5.0)

            ctx = WorkflowContext(
                experiment_id="exp-42",
                instrument_name="sans-test",
                root_dir=root,
                output_dir=output_dir,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="sample_a", transmission=0.87),
                sample_path,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="dark", mode="scattering", sample_name="B4C"),
                dark_path,
            )
            ctx.configurations["config_1"] = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="config_1",
                notes="baseline",
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
                    aperture2=Aperture(type="pinhole", diameter=0.004),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )
            ctx.set_refs_sub("config_1", output_dir / "refs_sub_config_1.nxs")
            ctx.set_refs_norm("config_1", output_dir / "refs_norm_config_1.nxs")
            ctx.set_masks_file("config_1", output_dir / "config_1_masks.nxs")
            ctx.set_mask("config_1", 0, np.array([[0, 1], [1, 0]], dtype=np.uint8))
            ctx.set_transmission("sample_a", "config_1", 0.9)
            ctx.info("saved workflow", where="test", step="round_trip")
            ctx.warn("check warning", where="test", key="config_1", hint="demo")
            ctx.add_artifact("runs_report.csv", output_dir / "runs_report.csv", kind="csv")
            ctx.timings["save"] = 1.25
            ctx.set("transmission_roi_detector", 0)
            ctx.set("custom_flags", {"resume": True, "label": "demo"})

            saved_path = save_workflow_context(ctx, context_path)
            loaded = load_workflow_context(saved_path)

            self.assertEqual(saved_path, context_path.resolve())
            self.assertEqual(loaded.experiment_id, "exp-42")
            self.assertEqual(loaded.instrument_name, "sans-test")
            self.assertEqual(loaded.root_dir, root.resolve())
            self.assertEqual(loaded.output_dir, output_dir.resolve())
            self.assertEqual(len(loaded.runs), 2)
            loaded_run_transmissions = {
                (run_key.config_id, run_key.entity, run_key.mode, run_key.sample_name): run_key.transmission
                for run_key in loaded.runs
            }
            self.assertAlmostEqual(
                loaded_run_transmissions[("config_1", "sample", "scattering", "sample_a")] or 0.0,
                0.87,
            )
            self.assertIsNone(loaded_run_transmissions[("config_1", "dark", "scattering", "B4C")])
            self.assertEqual(set(loaded.configurations), {"config_1"})
            self.assertEqual(loaded.configurations["config_1"].config_id, "config_1")
            self.assertEqual(loaded.get_refs_sub_path("config_1"), (output_dir / "refs_sub_config_1.nxs").resolve())
            self.assertEqual(loaded.get_refs_norm_path("config_1"), (output_dir / "refs_norm_config_1.nxs").resolve())
            self.assertEqual(loaded.get_masks_file_path("config_1"), (output_dir / "config_1_masks.nxs").resolve())
            np.testing.assert_array_equal(loaded.get_mask("config_1", 0), np.array([[0, 1], [1, 0]], dtype=np.uint8))
            self.assertAlmostEqual(loaded.get_transmission("sample_a", "config_1") or 0.0, 0.9)
            self.assertEqual(len(loaded.logs), 2)
            self.assertEqual(len(loaded.issues), 1)
            self.assertEqual(len(loaded.artifacts), 1)
            self.assertAlmostEqual(loaded.timings["save"], 1.25)
            self.assertEqual(loaded.get("transmission_roi_detector"), 0)
            self.assertEqual(loaded.get("custom_flags"), {"resume": True, "label": "demo"})

    def test_refs_norm_files_can_reference_another_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            output_dir.mkdir()
            refs_norm_path = output_dir / "refs_norm_config_1.nxs"
            with h5py.File(refs_norm_path, "w"):
                pass

            ctx = WorkflowContext(output_dir=output_dir)
            ctx.set_refs_norm("config_1", refs_norm_path)
            ctx.set_refs_norm("config_2", "config_1")

            self.assertTrue(ctx.is_refs_norm_alias("config_2"))
            self.assertEqual(ctx.resolve_refs_norm_config_id("config_2"), "config_1")
            self.assertEqual(ctx.get_refs_norm_path("config_2"), refs_norm_path.resolve())

            loaded = load_workflow_context(save_workflow_context(ctx, output_dir / "workflow_context.nxs"))
            self.assertEqual(loaded.refs_norm_files["config_2"], "config_1")
            self.assertEqual(loaded.get_refs_norm_path("config_2"), refs_norm_path.resolve())


if __name__ == "__main__":
    unittest.main()
