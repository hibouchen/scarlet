from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from scarlet.workflow import RunKey, WorkflowContext


class TestWorkflowRunUpdates(unittest.TestCase):
    def test_user_beam_center_update_invalidates_flatfield(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            flatfield = root / "flatfield.nxs"
            flatfield.write_bytes(b"flatfield")
            workflow.set_flatfield("config_1", flatfield)

            center = workflow.update_beam_center("config_1", 0, (12.5, 13.5))

            self.assertEqual(center, (12.5, 13.5))
            self.assertEqual(workflow.get_beam_center("config_1", 0), center)
            self.assertIsNone(workflow.get_flatfield("config_1"))
            self.assertIn("config_1", workflow.stale_flatfields)

    def test_user_roi_update_invalidates_transmissions_and_flatfields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            workflow.set_transmission_source("config_2", "config_1")
            workflow.set_transmission("sample-a", "config_1", 0.8)
            workflow.set_empty_cell_transmission("config_1", 0.9)
            for config_id in ("config_1", "config_2"):
                flatfield = root / f"flatfield-{config_id}.nxs"
                flatfield.write_bytes(b"flatfield")
                workflow.set_flatfield(config_id, flatfield)

            roi = workflow.update_roi("config_1", (1, 4, 2, 6))

            self.assertEqual(roi, (1, 4, 2, 6))
            self.assertIsNone(workflow.get_transmission("sample-a", "config_1"))
            self.assertIsNone(workflow.get_empty_cell_transmission("config_1"))
            self.assertIsNone(workflow.get_flatfield("config_1"))
            self.assertIsNone(workflow.get_flatfield("config_2"))

    def test_user_center_and_roi_updates_validate_coordinates(self) -> None:
        workflow = WorkflowContext()

        self.assertEqual(
            workflow.update_beam_center("config_1", 0, (-1.0, 1.0)),
            (-1.0, 1.0),
        )
        with self.assertRaises(ValueError):
            workflow.update_beam_center("config_1", 0, (float("nan"), 1.0))
        for roi in ((1, 1, 0, 2), (-1, 2, 0, 2)):
            with self.subTest(roi=roi):
                with self.assertRaises(ValueError):
                    workflow.update_roi("config_1", roi)

    def test_rename_sample_updates_all_modes_and_moves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            keys = [
                RunKey("config_1", "sample", "scattering", "old-name"),
                RunKey("config_1", "sample", "transmission", "old-name"),
            ]
            for index, key in enumerate(keys):
                path = root / f"run-{index}.hdf"
                path.write_bytes(b"raw")
                workflow.add_run(key, path)
            workflow.set_sample_thickness("old-name", "config_1", 1.5)
            workflow.set_transmission("old-name", "config_1", 0.8)

            replacements = workflow.rename_sample(keys[0], "new-name")

            self.assertEqual(len(replacements), 2)
            self.assertEqual(
                {candidate.sample_name for candidate in workflow.runs},
                {"new-name"},
            )
            self.assertEqual(workflow.get_sample_thickness("new-name", "config_1"), 1.5)
            self.assertEqual(workflow.get_transmission("new-name", "config_1"), 0.8)
            self.assertIsNone(workflow.get_sample_thickness("old-name", "config_1"))

    def test_rename_sample_rejects_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            old_key = RunKey("config_1", "sample", "scattering", "old-name")
            new_key = RunKey("config_1", "sample", "scattering", "new-name")
            for index, key in enumerate((old_key, new_key)):
                path = root / f"run-{index}.hdf"
                path.write_bytes(b"raw")
                workflow.add_run(key, path)

            with self.assertRaisesRegex(ValueError, "already exists"):
                workflow.rename_sample(old_key, "new-name")

    def test_update_run_thickness_accepts_positive_value_or_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "sample.hdf"
            raw_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            key = RunKey("config_1", "sample", "scattering", "sample-a")
            workflow.add_run(key, raw_path)

            workflow.update_run_thickness(key, 1.25)
            self.assertEqual(workflow.get_sample_thickness("sample-a", "config_1"), 1.25)

            workflow.update_run_thickness(key, None)
            self.assertIsNone(workflow.get_sample_thickness("sample-a", "config_1"))

    def test_update_run_thickness_rejects_non_positive_and_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "sample.hdf"
            raw_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            key = RunKey("config_1", "sample", "scattering", "sample-a")
            workflow.add_run(key, raw_path)

            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "strictly positive"):
                        workflow.update_run_thickness(key, value)

    def test_update_run_thickness_rejects_reference_without_sample_thickness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "empty-beam.hdf"
            raw_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            key = RunKey("config_1", "empty_beam", "scattering", "EB")
            workflow.add_run(key, raw_path)

            with self.assertRaisesRegex(ValueError, "sample and water"):
                workflow.update_run_thickness(key, 1.0)

    def test_replace_run_changes_mode_and_rebuilds_reference_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / "empty_beam.hdf"
            file_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            old_key = RunKey("config_1", "empty_beam", "transmission", "EB")
            workflow.add_run(old_key, file_path)

            new_key = workflow.replace_run(old_key, replace(old_key, mode="scattering"))

            self.assertEqual(new_key.mode, "scattering")
            self.assertNotIn(old_key, workflow.runs)
            self.assertEqual(workflow.runs[new_key], file_path.resolve())
            self.assertIsNone(workflow.get_empty_beam("config_1", "transmission"))
            self.assertEqual(
                workflow.get_empty_beam("config_1", "scattering"),
                file_path.resolve(),
            )
            self.assertEqual(workflow.logs[-1].where, "replace_run")

    def test_replace_run_allocates_duplicate_key_without_losing_either_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scattering_path = root / "scattering.hdf"
            transmission_path = root / "transmission.hdf"
            scattering_path.write_bytes(b"scattering")
            transmission_path.write_bytes(b"transmission")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            scattering_key = RunKey("config_1", "sample", "scattering", "sample-a")
            transmission_key = RunKey("config_1", "sample", "transmission", "sample-a")
            workflow.add_run(scattering_key, scattering_path)
            workflow.add_run(transmission_key, transmission_path)

            stored_key = workflow.replace_run(
                transmission_key,
                replace(transmission_key, mode="scattering"),
            )

            self.assertEqual(stored_key.duplicate_index, 1)
            self.assertEqual(len(workflow.runs), 2)
            self.assertEqual(workflow.runs[scattering_key], scattering_path.resolve())
            self.assertEqual(workflow.runs[stored_key], transmission_path.resolve())

    def test_replaced_mode_round_trips_through_workflow_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "sample.hdf"
            raw_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            old_key = RunKey("config_1", "sample", "scattering", "sample-a")
            workflow.add_run(old_key, raw_path)
            new_key = workflow.replace_run(old_key, replace(old_key, mode="transmission"))

            saved_path = workflow.save(root / "workflow.nxs")
            loaded = WorkflowContext.load(saved_path)

            self.assertIn(new_key, loaded.runs)
            self.assertNotIn(old_key, loaded.runs)
            self.assertEqual(loaded.runs[new_key], raw_path.resolve())

    def test_replace_run_rejects_an_unknown_source_key(self) -> None:
        workflow = WorkflowContext()
        key = RunKey("config_1", "sample", "scattering", "sample-a")

        with self.assertRaisesRegex(KeyError, "not registered"):
            workflow.replace_run(key, replace(key, mode="transmission"))


if __name__ == "__main__":
    unittest.main()
