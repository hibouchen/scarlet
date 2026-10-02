from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.workflow.context import RunKey, WorkflowContext


def _write_pipeline_ready_nexus(path: Path, *, processed: bool) -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        instrument = entry.create_group("instrument")
        detector = instrument.create_group("detector0")
        detector.create_dataset("data", data=np.ones((2, 2), dtype=np.float64))
        entry.create_group("control")
        if processed:
            processed_entry = handle.create_group("processed")
            processed_entry.attrs["NX_class"] = np.bytes_("NXentry")


class TestWorkflowProcessedRuns(unittest.TestCase):
    def test_processed_runs_table_reads_live_nexus_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            processed_path = root / "processed_run.nxs"
            pending_path = root / "pending_run.nxs"
            _write_pipeline_ready_nexus(processed_path, processed=True)
            _write_pipeline_ready_nexus(pending_path, processed=False)

            context = WorkflowContext(root_dir=root, output_dir=root / "out")
            processed_key = RunKey(
                config_id="config_1",
                entity="sample",
                mode="scattering",
                sample_name="processed_sample",
            )
            pending_key = RunKey(
                config_id="config_1",
                entity="sample",
                mode="scattering",
                sample_name="pending_sample",
            )
            context.add_run(processed_key, processed_path)
            context.add_run(pending_key, pending_path)

            self.assertTrue(context.is_run_processed(processed_key))
            self.assertFalse(context.is_run_processed(pending_key))
            self.assertEqual(
                [row["run_key"] for row in context.processed_runs_table().rows],
                [processed_key.short()],
            )
            self.assertEqual(
                [row["status"] for row in context.runs_status_table().rows],
                ["processed", "unprocessed"],
            )

            with h5py.File(pending_path, "a") as handle:
                handle.create_group("processed")

            self.assertTrue(context.is_run_processed(pending_key))
            self.assertEqual(len(context.processed_runs_table().rows), 2)

    def test_processed_status_uses_expected_converted_nexus_without_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir = root / "raw"
            output_dir = root / "out"
            raw_dir.mkdir()
            output_dir.mkdir()
            raw_path = raw_dir / "run_001.hdf"
            raw_path.touch()

            context = WorkflowContext(root_dir=raw_dir, output_dir=output_dir)
            key = RunKey(
                config_id="config_1",
                entity="sample",
                mode="scattering",
                sample_name="sample_a",
            )
            context.add_run(key, raw_path)
            nexus_path = output_dir / "run_001.nxs"
            with h5py.File(nexus_path, "w") as handle:
                handle.create_group("processed")

            self.assertEqual(context.get_run_nexus_path(key), nexus_path.resolve())
            self.assertTrue(context.is_run_processed(key))

    def test_missing_associated_nexus_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_path = root / "run_001.hdf"
            raw_path.touch()
            context = WorkflowContext(root_dir=root, output_dir=root / "out")
            key = RunKey(
                config_id="config_1",
                entity="sample",
                mode="scattering",
                sample_name="sample_a",
            )
            context.add_run(key, raw_path)

            self.assertFalse(context.is_run_processed(key))
            self.assertEqual(context.runs_status_table().rows[0]["status"], "missing")


if __name__ == "__main__":
    unittest.main()
