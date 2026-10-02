from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from scarlet.workflow.context import RunKey, initialize_workflow_context_from_raw_directory
from scarlet.workflow.pipeline import ReductionState


def _write_d11_raw(
    path: Path,
    *,
    sample_name: str = "sample_a",
    measurement_mode: str = "scattering",
) -> None:
    counts = np.zeros((7, 7, 1), dtype=np.float64)
    if measurement_mode == "scattering":
        counts[0, 0, 0] = 100.0
    elif measurement_mode == "transmission":
        counts[3, 3, 0] = 100.0
    else:
        raise ValueError(f"Unsupported measurement mode: {measurement_mode}")
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry0")
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        entry.create_dataset("sample_description", data=np.bytes_(sample_name))
        entry.create_dataset("duration", data=10.0)

        monitor = entry.create_group("monitor1")
        monitor.attrs["NX_class"] = np.bytes_("NXmonitor")
        monitor.create_dataset("mode", data=np.bytes_("monitor"))
        monitor.create_dataset("preset", data=1000.0)
        monitor.create_dataset("integral", data=1000.0)

        instrument = entry.create_group("D11")
        instrument.attrs["NX_class"] = np.bytes_("NXinstrument")
        beam = instrument.create_group("Beam")
        beam.create_dataset("center_x", data=3.0)
        beam.create_dataset("center_y", data=3.0)
        beam.create_dataset("sample_ap_x_or_diam", data=7.0)
        beam.create_dataset("sample_ap_y", data=10.0)

        selector = instrument.create_group("selector")
        selector.create_dataset("wavelength", data=6.0)
        selector.create_dataset("wavelength_res", data=9.0)

        collimation = instrument.create_group("collimation")
        collimation.create_dataset("actual_position", data=8.0)
        collimation.create_dataset("guide_exit_cross_section_width", data=45.0)
        collimation.create_dataset("guide_exit_cross_section_height", data=50.0)

        detector = instrument.create_group("detector")
        detector.create_dataset("data", data=counts)
        distance = detector.create_dataset("det_actual", data=8.0)
        distance.attrs["units"] = np.bytes_("m")


class TestWorkflowLazyConversion(unittest.TestCase):
    def test_semi_transparent_strategy_does_not_override_detected_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_directory = root / "raw"
            output_directory = root / "out"
            raw_directory.mkdir()
            raw_path = raw_directory / "transmission.nxs"
            _write_d11_raw(raw_path, measurement_mode="transmission")

            workflow = initialize_workflow_context_from_raw_directory(
                raw_directory,
                output_dir=output_directory,
                instrument_name="d11",
                transmission_strategy="semi_transparent_beamstop",
            )

            key = RunKey(
                config_id="config_1",
                entity="sample",
                mode="transmission",
                sample_name="sample_a",
            )
            self.assertEqual(workflow.get_run_path(key), raw_path.resolve())

    def test_initialization_registers_raw_paths_without_converting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_directory = root / "raw"
            output_directory = root / "out"
            raw_directory.mkdir()
            raw_path = raw_directory / "run_001.nxs"
            unused_raw_path = raw_directory / "run_002.nxs"
            ignored_path = raw_directory / "notes.txt"
            _write_d11_raw(raw_path)
            _write_d11_raw(unused_raw_path, sample_name="sample_b")
            ignored_path.write_text("not a data file", encoding="utf-8")

            with mock.patch("scarlet.io.converters.convert_to_scarlet_nxsas_raw") as converter:
                workflow = initialize_workflow_context_from_raw_directory(
                    raw_directory,
                    output_dir=output_directory,
                    instrument_name="d11",
                    transmission_strategy="semi_transparent_beamstop",
                )

            key = RunKey(
                config_id="config_1",
                entity="sample",
                mode="scattering",
                sample_name="sample_a",
            )
            converter.assert_not_called()
            self.assertEqual(workflow.get_run_path(key), raw_path.resolve())
            self.assertEqual(list(output_directory.glob("*.nxs")), [])
            self.assertEqual(workflow.artifacts, [])
            self.assertTrue(workflow.is_file_excluded(ignored_path))
            self.assertEqual(
                workflow.excluded_files_table().rows[0]["source"],
                "directory_scan",
            )

            with mock.patch("scarlet.io.converters.convert_to_scarlet_nxsas_raw") as converter:
                workflow.compute_transmissions()
            converter.assert_not_called()
            self.assertEqual(list(output_directory.glob("*.nxs")), [])

    def test_refresh_runs_adds_new_raw_files_and_respects_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_directory = root / "raw"
            output_directory = root / "out"
            raw_directory.mkdir()
            first_path = raw_directory / "run_001.hdf"
            second_path = raw_directory / "run_002.hdf"
            excluded_path = raw_directory / "run_003.hdf"
            _write_d11_raw(first_path, sample_name="sample_a")

            workflow = initialize_workflow_context_from_raw_directory(
                raw_directory,
                output_dir=output_directory,
                instrument_name="d11",
                transmission_strategy="semi_transparent_beamstop",
            )
            _write_d11_raw(second_path, sample_name="sample_b")
            _write_d11_raw(excluded_path, sample_name="sample_c")
            workflow.exclude_file(
                excluded_path,
                reason="Removed from runs table CSV",
                source="runs_table",
            )

            with mock.patch("scarlet.io.converters.convert_to_scarlet_nxsas_raw") as converter:
                returned = workflow.refresh_runs()

            converter.assert_not_called()
            self.assertIs(returned, workflow)
            self.assertEqual(set(workflow.runs.values()), {first_path.resolve(), second_path.resolve()})
            self.assertTrue(workflow.is_file_excluded(excluded_path))

            workflow.refresh_runs()
            self.assertEqual(len(workflow.runs), 2)

            workflow.refresh_runs(include_excluded=True)
            self.assertEqual(
                set(workflow.runs.values()),
                {first_path.resolve(), second_path.resolve(), excluded_path.resolve()},
            )
            self.assertFalse(workflow.is_file_excluded(excluded_path))

    @unittest.skipIf(importlib.util.find_spec("scipp") is None, "scipp is required to load pipeline data")
    def test_pipeline_converts_only_requested_run_and_keeps_raw_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_directory = root / "raw"
            output_directory = root / "out"
            raw_directory.mkdir()
            raw_path = raw_directory / "run_001.nxs"
            unused_raw_path = raw_directory / "run_002.nxs"
            _write_d11_raw(raw_path)
            _write_d11_raw(unused_raw_path, sample_name="sample_b")

            workflow = initialize_workflow_context_from_raw_directory(
                raw_directory,
                output_dir=output_directory,
                instrument_name="d11",
                transmission_strategy="semi_transparent_beamstop",
            )
            key = next(iter(workflow.runs))

            state = ReductionState(
                sample_name="sample_a",
                config_id="config_1",
                workflow=workflow,
                transmission=1.0,
            )

            converted_path = output_directory / "run_001.nxs"
            self.assertEqual(Path(state.file_path), converted_path.resolve())
            self.assertTrue(converted_path.exists())
            self.assertFalse((output_directory / "run_002.nxs").exists())
            self.assertEqual(workflow.get_run_path(key), raw_path.resolve())
            self.assertEqual([artifact.path for artifact in workflow.artifacts], [converted_path.resolve()])

            with mock.patch(
                "scarlet.io.converters.convert_to_scarlet_nxsas_raw",
                side_effect=AssertionError("cached conversion should be reused"),
            ):
                self.assertEqual(workflow.prepare_run(key), converted_path.resolve())


if __name__ == "__main__":
    unittest.main()
