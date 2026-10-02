from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

import scarlet.reduction.stiching as st
from scarlet.reduction.geometry import compute_q_norm_map
from scarlet.reduction.integration import azimuthal_average
from scarlet.workflow.configuration import Configuration
from scarlet.workflow.context import RunKey, WorkflowContext
from scarlet.workflow.pipeline import (
    ReductionPipeline,
    ReductionState,
    StichingPipeline,
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
    count_time: float | None = None,
    dead_time: float | None = None,
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
        if count_time is not None:
            control.create_dataset("count_time", data=float(count_time))

        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = b"NXinstrument"
        detector = instrument.create_group("detector0")
        detector.attrs["NX_class"] = b"NXdetector"
        detector.create_dataset("data", data=np.asarray(data, dtype=np.float64))
        if dead_time is not None:
            detector.create_dataset("dead_time", data=float(dead_time))
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
        workflow = WorkflowContext()
        pipeline = ReductionPipeline.with_processed_output(workflow)

        self.assertIs(pipeline.workflow, workflow)
        self.assertEqual(
            pipeline.step_names,
            (
                "subtract references",
                "water normalization",
                "save processed detectors",
            ),
        )

    def test_with_azimuthal_text_output_includes_save_text_step(self) -> None:
        pipeline = ReductionPipeline.with_azimuthal_text_output(WorkflowContext())

        self.assertEqual(
            pipeline.step_names,
            (
                "subtract references",
                "water normalization",
                "normalize by thickness",
                "azimuthal averaging",
                "save processed detectors",
                "save azimuthal text",
            ),
        )

    def test_run_for_sample_reports_missing_scattering_run_for_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample_scattering.nxs"
            _write_detector_file(
                sample_path,
                sample_name="sample_a",
                data=np.ones((2, 2), dtype=np.float64),
            )

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.add_run(
                RunKey(config_id="cfg_2", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )

            with self.assertRaisesRegex(
                ValueError,
                "Missing sample scattering run .*config_id='cfg_4'.*cfg_2",
            ):
                ReductionPipeline.default(ctx).run_for_sample(
                    sample_name="sample_a",
                    config_id="cfg_4",
                )

    def test_run_new_skips_processed_runs_without_refreshing_and_logs_failures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processed_path = root / "processed.nxs"
            successful_path = root / "successful.nxs"
            failing_path = root / "failing.nxs"
            for path, sample_name in (
                (processed_path, "already_done"),
                (successful_path, "new_sample"),
                (failing_path, "broken_sample"),
            ):
                _write_detector_file(
                    path,
                    sample_name=sample_name,
                    data=np.ones((2, 2), dtype=np.float64),
                )
            with h5py.File(processed_path, "a") as handle:
                handle.create_group("processed")

            workflow = WorkflowContext(root_dir=root, output_dir=root / "out")
            keys = {
                sample_name: RunKey(
                    config_id="cfg",
                    entity="sample",
                    mode="scattering",
                    sample_name=sample_name,
                )
                for sample_name in ("already_done", "new_sample", "broken_sample")
            }
            workflow.add_run(keys["already_done"], processed_path)
            workflow.add_run(keys["new_sample"], successful_path)
            workflow.add_run(keys["broken_sample"], failing_path)
            pipeline = ReductionPipeline(workflow=workflow)
            successful_state = mock.create_autospec(ReductionState, instance=True)

            def run_for_runkey(_pipeline: ReductionPipeline, run_key: RunKey) -> ReductionState:
                if run_key == keys["broken_sample"]:
                    raise RuntimeError("cannot reduce this run")
                return successful_state

            with (
                mock.patch.object(workflow, "refresh_runs", return_value=workflow) as refresh,
                mock.patch.object(
                    ReductionPipeline,
                    "run_for_runkey",
                    autospec=True,
                    side_effect=run_for_runkey,
                ) as run,
            ):
                states = pipeline.run_new()

            refresh.assert_not_called()
            self.assertEqual(states, [successful_state])
            self.assertEqual(
                [call.args[1] for call in run.call_args_list],
                [keys["new_sample"], keys["broken_sample"]],
            )
            self.assertTrue(
                any(
                    log.level == "ERROR"
                    and log.meta.get("key") == keys["broken_sample"].short()
                    and "cannot reduce this run" in log.meta.get("error", "")
                    for log in workflow.logs
                )
            )


class TestStichingPipeline(unittest.TestCase):
    def test_run_for_sample_reads_only_processed_nexus_files_without_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processed_path = root / "processed.nxs"
            pending_path = root / "pending.nxs"
            _write_detector_file(
                processed_path,
                sample_name="sample_a",
                data=np.ones((2, 2), dtype=np.float64),
            )
            _write_detector_file(
                pending_path,
                sample_name="sample_a",
                data=np.ones((2, 2), dtype=np.float64),
            )
            with h5py.File(processed_path, "a") as handle:
                handle.create_group("processed")

            workflow = WorkflowContext(root_dir=root, output_dir=root / "out")
            workflow.add_run(
                RunKey("cfg_1", "sample", "scattering", "sample_a"),
                processed_path,
            )
            workflow.add_run(
                RunKey("cfg_2", "sample", "scattering", "sample_a"),
                pending_path,
            )
            pipeline = StichingPipeline(workflow)
            result = SimpleNamespace(
                final_curve=SimpleNamespace(
                    q=np.asarray([0.1]),
                    i=np.asarray([2.0]),
                    di=np.asarray([0.2]),
                    dq=np.asarray([0.01]),
                ),
                origin_segment_id=np.asarray([0]),
                origin_map={0: "cfg_1/detector0"},
            )

            with (
                mock.patch.object(
                    workflow,
                    "prepare_run",
                    side_effect=AssertionError("stitching must not prepare or convert runs"),
                ) as prepare,
                mock.patch.object(st, "load_segment_from_nexus", return_value=[object()]) as load,
                mock.patch.object(st, "stitch_segments_greedy", return_value=result),
            ):
                final_data = pipeline.run_for_sample("sample_a")

            prepare.assert_not_called()
            load.assert_called_once_with(processed_path.resolve(), config_id="cfg_1")
            np.testing.assert_allclose(final_data[:, :4], [[0.1, 2.0, 0.2, 0.01]])
            self.assertTrue((root / "out" / "sample_a_merged.txt").exists())

    def test_run_new_stitches_only_ready_samples_and_logs_failures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_dir = root / "out"
            output_dir.mkdir()
            workflow = WorkflowContext(root_dir=root, output_dir=output_dir)
            keys: dict[str, RunKey] = {}
            paths: dict[str, Path] = {}
            for sample_name in ("already_stitched", "new_sample", "broken_sample", "waiting_sample"):
                path = root / f"{sample_name}.nxs"
                _write_detector_file(
                    path,
                    sample_name=sample_name,
                    data=np.ones((2, 2), dtype=np.float64),
                )
                if sample_name != "waiting_sample":
                    with h5py.File(path, "a") as handle:
                        handle.create_group("processed")
                key = RunKey("cfg", "sample", "scattering", sample_name)
                workflow.add_run(key, path)
                keys[sample_name] = key
                paths[sample_name] = path

            stitched_output = output_dir / "already_stitched_merged.txt"
            stitched_output.write_text("already merged", encoding="utf-8")
            input_mtime = paths["already_stitched"].stat().st_mtime_ns
            stitched_output.touch()
            if stitched_output.stat().st_mtime_ns < input_mtime:
                self.fail("test filesystem did not preserve output modification ordering")

            pipeline = StichingPipeline(workflow)
            successful_data = np.asarray([[0.1, 1.0, 0.1, 0.01, 0.0]])

            def run_for_sample(
                _pipeline: StichingPipeline,
                sample_name: str,
                scale_on: str | None = None,
                normalization_factor: float = 1.0,
            ) -> np.ndarray:
                del scale_on, normalization_factor
                if sample_name == "broken_sample":
                    raise RuntimeError("cannot stitch this sample")
                return successful_data

            with mock.patch.object(
                StichingPipeline,
                "run_for_sample",
                autospec=True,
                side_effect=run_for_sample,
            ) as run:
                results = pipeline.run_new()

            self.assertEqual(set(results), {"new_sample"})
            self.assertIs(results["new_sample"], successful_data)
            self.assertEqual(
                [call.args[1] for call in run.call_args_list],
                ["broken_sample", "new_sample"],
            )
            self.assertTrue(
                any(
                    log.level == "ERROR"
                    and log.meta.get("key") == "broken_sample"
                    and "cannot stitch this sample" in log.meta.get("error", "")
                    for log in workflow.logs
                )
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
                source_nexus_file="sample_scattering.nxs",
            )

            self.assertEqual(output_path.name, "sample_config_cfg.txt")
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "# sample_name: sample_a")
            self.assertEqual(lines[1], "# config_id: cfg")
            self.assertEqual(lines[2], "# transmission: 0.5")
            self.assertEqual(lines[3], "# source_nexus_file: sample_scattering.nxs")
            self.assertEqual(lines[4], "# q I I_error q_error")
            self.assertEqual(len(lines[5].split()), 4)
            self.assertEqual(len(lines[6].split()), 4)


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

    def test_subtract_references_step_handles_empty_cell_with_different_count_time(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample_scattering.nxs"
            empty_cell_path = root / "empty_cell_scattering.nxs"
            dead_time = 1.0e-2
            _write_detector_file(
                sample_path,
                sample_name="sample_a",
                data=np.full((2, 2), 50.0, dtype=np.float64),
                monitor_integral=20.0,
                count_time=10.0,
                dead_time=dead_time,
            )
            _write_detector_file(
                empty_cell_path,
                sample_name="EC",
                data=np.full((2, 2), 25.0, dtype=np.float64),
                monitor_integral=10.0,
                count_time=5.0,
                dead_time=dead_time,
            )

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.add_run(
                RunKey(config_id="cfg", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )
            ctx.add_run(
                RunKey(config_id="cfg", entity="empty_cell", mode="scattering", sample_name="EC"),
                empty_cell_path,
            )
            ctx.set_empty_cell_transmission("cfg", 1.0)

            state = ReductionState(sample_name="sample_a", config_id="cfg", workflow=ctx, transmission=1.0)
            updated = subtract_references_step(state)

            np.testing.assert_allclose(updated.detectors[0].data.values, np.zeros((2, 2), dtype=np.float64))

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
            self.assertIsNone(updated.detectors[0].coords["q"].variances)
            self.assertIsNone(expected.coords["q"].variances)
            self.assertNotIn("q_error", updated.detectors[0].coords)
            self.assertNotIn("q_error", expected.coords)
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
