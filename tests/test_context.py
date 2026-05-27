from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.workflow.configuration import Aperture, Collimation, Configuration
from scarlet.workflow.context import (
    RunKey,
    WorkflowContext,
    generate_reference_files_from_workflow_context,
    update_reference_masks_from_workflow_context,
    update_workflow_context_from_runs_report_csv,
    write_runs_report_csv,
)

from test_workflow_configuration import _write_minimal_masks_file, _write_minimal_raw_nexus_file


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
                    "file_path": sample_row["file_path"],
                }
            ]

            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_name", "config_id", "mode", "entity", "file_path"],
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
                ("sample_name", "config_id", "mode", "entity", "file_path"),
            )
            self.assertEqual(len(table.rows), 2)
            self.assertEqual(table.rows[0]["sample_name"], "sample_a")
            self.assertEqual(table.rows[0]["config_id"], "config_1")
            self.assertEqual(table.rows[0]["mode"], "scattering")
            self.assertEqual(table.rows[0]["entity"], "sample")
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


if __name__ == "__main__":
    unittest.main()
