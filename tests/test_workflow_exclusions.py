from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scarlet.workflow.configuration import Configuration
from scarlet.workflow.context import RunKey, WorkflowContext


class TestWorkflowExclusions(unittest.TestCase):
    def test_add_run_records_reinstatement_of_excluded_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "run.hdf"
            context = WorkflowContext()
            key = RunKey(
                config_id="config_1",
                entity="sample",
                mode="scattering",
                sample_name="sample_a",
            )

            context.exclude_file(
                file_path,
                reason="Temporarily ignored",
                source="manual",
                run_key=key,
            )
            context.add_run(key, file_path)

            self.assertFalse(context.is_file_excluded(file_path))
            self.assertEqual(
                [event.action for event in context.excluded_files],
                ["excluded", "reinstated"],
            )
            self.assertEqual(context.excluded_files_table().rows, [])
            self.assertEqual(len(context.excluded_files_table(include_history=True).rows), 2)

    def test_update_from_runs_table_records_removed_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            kept_path = root / "kept.hdf"
            removed_path = root / "removed.hdf"
            kept_path.touch()
            removed_path.touch()

            context = WorkflowContext(instrument_name="d11", root_dir=root, output_dir=root / "out")
            kept_key = RunKey(
                config_id="config_1",
                entity="sample",
                mode="scattering",
                sample_name="kept",
            )
            removed_key = RunKey(
                config_id="config_1",
                entity="sample",
                mode="scattering",
                sample_name="removed",
            )
            context.add_run(kept_key, kept_path)
            context.add_run(removed_key, removed_path)
            context.set_sample_thickness("kept", "config_1", 1.0)
            context.set_sample_thickness("removed", "config_1", 1.0)

            csv_path = root / "runs_filtered.csv"
            table = context.runs_table()
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table.columns))
                writer.writeheader()
                writer.writerow(next(row for row in table.rows if row["sample_name"] == "kept"))

            inspected = SimpleNamespace(
                configuration=Configuration(
                    wavelength=6.0,
                    sample_detector_distance=[8.0],
                    config_id="config_1",
                )
            )
            with mock.patch("scarlet.io.raw_inspection.inspect_raw_run", return_value=inspected):
                context.update_from_runs_table_csv(csv_path)

            self.assertEqual(set(context.runs.values()), {kept_path.resolve()})
            self.assertTrue(context.is_file_excluded(removed_path))
            event = context.active_excluded_files()[0]
            self.assertEqual(event.source, "runs_table")
            self.assertEqual(event.run_key, removed_key.short())


if __name__ == "__main__":
    unittest.main()
