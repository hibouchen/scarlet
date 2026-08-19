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
    _initialize_transmission_geometry,
    initialize_workflow_context_from_raw_directory,
)

from test_workflow_configuration import _write_minimal_raw_nexus_file


def _write_transmission_file(
    path: Path,
    *,
    data: np.ndarray,
    monitor_integral: float,
) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = b"NXentry"
        control = entry.create_group("control")
        control.attrs["NX_class"] = b"NXmonitor"
        control.create_dataset("integral", data=float(monitor_integral))
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = b"NXinstrument"
        detector = instrument.create_group("detector0")
        detector.attrs["NX_class"] = b"NXdetector"
        detector.create_dataset("data", data=np.asarray(data, dtype=np.float64))
        detector.create_dataset("beam_center_x", data=1.5)
        detector.create_dataset("beam_center_y", data=1.5)


def _write_sample_thickness(path: Path, thickness_mm: float, *, units: str | None = "mm") -> None:
    with h5py.File(path, "a") as f:
        for entry_name in ("entry", "raw_data", "entry0", "entry1"):
            if entry_name not in f:
                continue
            sample = f[entry_name].get("sample")
            if not isinstance(sample, h5py.Group):
                continue
            if "thickness" in sample:
                del sample["thickness"]
            dataset = sample.create_dataset("thickness", data=float(thickness_mm))
            if units is not None:
                dataset.attrs["units"] = np.bytes_(units)
            return
    raise ValueError(f"Missing sample group in {path}")


class TestWorkflowContext(unittest.TestCase):
    def test_missing_getters_return_none(self) -> None:
        ctx = WorkflowContext()

        self.assertIsNone(
            ctx.get_run_path(RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="sample_a"))
        )
        self.assertIsNone(ctx.get_beam_center("config_1", 0))
        self.assertIsNone(ctx.get_roi("config_1"))
        self.assertIsNone(ctx.get_dark("config_1"))
        self.assertIsNone(ctx.get_empty_beam("config_1", "transmission"))
        self.assertIsNone(ctx.get_empty_cell("config_1", "scattering"))
        self.assertIsNone(ctx.get_water("config_1", "scattering"))
        self.assertIsNone(ctx.get_transmission("sample_a", "config_1"))
        self.assertIsNone(ctx.get_empty_cell_transmission("config_1"))
        self.assertIsNone(ctx.get_sample_thickness("sample_a", "config_1"))
        self.assertIsNone(ctx.get_reference_file("water", "scattering", "config_1"))
        self.assertEqual(ctx.get_transmission_strategy(), "opaque_beamstop")
        self.assertEqual(ctx.get_transmission_source_mode(), "transmission")

    def test_transmission_strategy_can_switch_to_semi_transparent_beamstop(self) -> None:
        ctx = WorkflowContext()

        ctx.set_transmission_strategy("semi_transparent")

        self.assertEqual(ctx.get_transmission_strategy(), "semi_transparent_beamstop")
        self.assertEqual(ctx.get_transmission_source_mode(), "scattering")

    def test_transmission_source_resolves_sample_and_empty_cell_values(self) -> None:
        ctx = WorkflowContext()
        ctx.set_transmission("sample_a", "config_source", 0.42)
        ctx.set_empty_cell_transmission("config_source", 0.84)

        resolved = ctx.set_transmission_source("config_target", "config_source")

        self.assertEqual(resolved, "config_source")
        self.assertEqual(ctx.get_transmission_source("config_target"), "config_source")
        self.assertEqual(ctx.resolve_transmission_config("config_target"), "config_source")
        self.assertEqual(ctx.get_transmission("sample_a", "config_target"), 0.42)
        self.assertEqual(ctx.get_empty_cell_transmission("config_target"), 0.84)

    def test_transmission_source_rejects_cycles(self) -> None:
        ctx = WorkflowContext()
        ctx.set_transmission_source("config_b", "config_c")

        with self.assertRaisesRegex(ValueError, "Cycle detected"):
            ctx.set_transmission_source("config_c", "config_b")

        self.assertIsNone(ctx.get_transmission_source("config_c"))

    def test_add_run_preserves_duplicate_logical_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dark_a = root / "dark_a.nxs"
            dark_b = root / "dark_b.nxs"
            _write_minimal_raw_nexus_file(dark_a, sample_name="Cd", count_time_s=5.0)
            _write_minimal_raw_nexus_file(dark_b, sample_name="Cd", count_time_s=6.0)

            ctx = WorkflowContext(output_dir=root / "out")
            key = RunKey(config_id="config_1", entity="dark", mode="scattering", sample_name="Cd")

            first_key = ctx.add_run(key, dark_a)
            second_key = ctx.add_run(key, dark_b)

            self.assertEqual(first_key, key)
            self.assertNotEqual(second_key, key)
            self.assertEqual(len(ctx.runs), 2)
            self.assertEqual(ctx.get_run_path(first_key), dark_a.resolve())
            self.assertEqual(ctx.get_run_path(second_key), dark_b.resolve())
            self.assertEqual([path.name for _, path in ctx.iter_runs(entity="dark")], ["dark_a.nxs", "dark_b.nxs"])

    def test_runs_table_displays_runs_with_transmission_column(self) -> None:
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
            ctx.set_transmission("sample_a", "config_1", 0.9)

            table = ctx.runs_table()

            self.assertEqual(
                table.columns,
                ("sample_name", "config_id", "mode", "entity", "thickness", "transmission", "file_path"),
            )
            self.assertEqual(len(table.rows), 2)
            self.assertEqual(table.rows[0]["sample_name"], "sample_a")
            self.assertEqual(table.rows[0]["config_id"], "config_1")
            self.assertEqual(table.rows[0]["mode"], "scattering")
            self.assertEqual(table.rows[0]["entity"], "sample")
            self.assertEqual(table.rows[0]["thickness"], "")
            self.assertEqual(table.rows[0]["transmission"], "0.9")
            self.assertEqual(table.rows[0]["file_path"], sample_path.name)
            self.assertEqual(table.rows[1]["transmission"], "")
            self.assertIn("<table>", table._repr_html_())

    def test_write_runs_table_csv_saves_table_content(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample.nxs"
            _write_minimal_raw_nexus_file(sample_path, sample_name="sample_a", count_time_s=10.0)

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )
            ctx.set_transmission("sample_a", "config_1", 0.9)

            csv_path = ctx.write_runs_table_csv(root / "tables" / "runs.csv")

            self.assertEqual(csv_path, (root / "tables" / "runs.csv").resolve())
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_name"], "sample_a")
            self.assertEqual(rows[0]["config_id"], "config_1")
            self.assertEqual(rows[0]["mode"], "scattering")
            self.assertEqual(rows[0]["entity"], "sample")
            self.assertEqual(rows[0]["thickness"], "")
            self.assertEqual(rows[0]["transmission"], "0.9")
            self.assertEqual(rows[0]["file_path"], "sample.nxs")
            self.assertEqual(ctx.artifacts[-1].path, csv_path)
            self.assertEqual(ctx.artifacts[-1].kind, "csv")

    def test_configurations_table_displays_readable_configuration_parameters(self) -> None:
        ctx = WorkflowContext()
        ctx.configurations["config_2"] = Configuration(
            wavelength=6.0,
            sample_detector_distance=[4.2, 1.8],
            collimation=Collimation(
                aperture1=Aperture(type="slit", x_gap=0.01, y_gap=0.02),
                aperture2=Aperture(type="pinhole", diameter=0.005),
                collimation_distance=8.5,
                last_aperture_to_sample_distance=2.1,
            ),
            config_id="config_2",
            notes="test config",
        )

        table = ctx.configurations_table()

        self.assertEqual(
            table.columns,
            (
                "config_id",
                "wavelength",
                "sample_detector_distance",
                "collimation_distance",
                "last_aperture_to_sample_distance",
                "aperture1",
                "aperture2",
                "notes",
            ),
        )
        self.assertEqual(len(table.rows), 1)
        self.assertEqual(table.rows[0]["config_id"], "config_2")
        self.assertEqual(table.rows[0]["wavelength"], "6 A")
        self.assertEqual(table.rows[0]["sample_detector_distance"], "detector0=4.2 m; detector1=1.8 m")
        self.assertEqual(table.rows[0]["collimation_distance"], "8.5 m")
        self.assertEqual(table.rows[0]["last_aperture_to_sample_distance"], "2.1 m")
        self.assertEqual(table.rows[0]["aperture1"], "slit x=0.01 m y=0.02 m")
        self.assertEqual(table.rows[0]["aperture2"], "pinhole d=0.005 m")
        self.assertEqual(table.rows[0]["notes"], "test config")
        self.assertIn("<table>", table._repr_html_())

    def test_update_from_runs_table_csv_applies_manual_edits(self) -> None:
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
            ctx.set_transmission("sample_a", "config_1", 0.9)

            csv_path = ctx.write_runs_table_csv(root / "tables" / "runs.csv")
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["sample_name"] = "sample_b"
            rows[0]["config_id"] = "config_2"
            rows[0]["thickness"] = "1.5"
            rows[0]["transmission"] = "0.5"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            out = ctx.update_from_runs_table_csv(csv_path)

            self.assertIs(out, ctx)
            self.assertIn(
                RunKey(config_id="config_2", entity="sample", mode="scattering", sample_name="sample_b"),
                ctx.runs,
            )
            self.assertEqual(
                ctx.get_run_path(
                    RunKey(config_id="config_2", entity="sample", mode="scattering", sample_name="sample_b")
                ),
                sample_path.resolve(),
            )
            self.assertEqual(ctx.transmissions, {("sample_b", "config_2"): 0.5})
            self.assertEqual(ctx.sample_thicknesses, {("sample_b", "config_2"): 1.5})
            self.assertIn("config_2", ctx.configurations)
            self.assertEqual(ctx.get("runs_table_csv"), csv_path.resolve())

    def test_runs_table_displays_sample_thickness_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample.nxs"
            _write_minimal_raw_nexus_file(sample_path, sample_name="sample_a", count_time_s=10.0)

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )
            ctx.set_sample_thickness("sample_a", "config_1", 2.0)

            table = ctx.runs_table()

            self.assertEqual(table.rows[0]["thickness"], "2")

    def test_compute_transmissions_includes_empty_cell_transmission_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            empty_beam_path = root / "empty_beam_transmission.nxs"
            sample_path = root / "sample_transmission.nxs"
            empty_cell_path = root / "empty_cell_transmission.nxs"

            empty_beam_data = np.zeros((4, 4), dtype=np.float64)
            empty_beam_data[1:3, 1:3] = 100.0
            sample_data = np.zeros((4, 4), dtype=np.float64)
            sample_data[1:3, 1:3] = 40.0
            empty_cell_data = np.zeros((4, 4), dtype=np.float64)
            empty_cell_data[1:3, 1:3] = 50.0

            _write_transmission_file(empty_beam_path, data=empty_beam_data, monitor_integral=10.0)
            _write_transmission_file(sample_path, data=sample_data, monitor_integral=20.0)
            _write_transmission_file(empty_cell_path, data=empty_cell_data, monitor_integral=10.0)

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.set_empty_beam("config_1", "transmission", empty_beam_path)
            ctx.set_roi("config_1", (1, 3, 1, 3))
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="transmission", sample_name="sample_a"),
                sample_path,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="empty_cell", mode="transmission", sample_name="empty_cell"),
                empty_cell_path,
            )

            transmissions = ctx.compute_transmissions()

            self.assertEqual(transmissions, {("sample_a", "config_1"): 0.2, ("empty_cell", "config_1"): 0.5})
            self.assertAlmostEqual(ctx.get_transmission("sample_a", "config_1"), 0.2)
            self.assertAlmostEqual(ctx.get_transmission("empty_cell", "config_1"), 0.5)
            self.assertAlmostEqual(ctx.get_empty_cell_transmission("config_1"), 0.5)

    def test_compute_transmissions_skips_when_empty_beam_or_roi_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample_transmission.nxs"
            sample_data = np.zeros((4, 4), dtype=np.float64)
            sample_data[1:3, 1:3] = 40.0
            _write_transmission_file(sample_path, data=sample_data, monitor_integral=20.0)

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="transmission", sample_name="sample_a"),
                sample_path,
            )

            transmissions = ctx.compute_transmissions()

            self.assertEqual(transmissions, {})
            self.assertIsNone(ctx.get_transmission("sample_a", "config_1"))
            self.assertTrue(any(issue.where == "compute_transmissions" for issue in ctx.issues))

    def test_compute_transmissions_can_use_scattering_runs_with_semi_transparent_beamstop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            empty_beam_path = root / "empty_beam_scattering.nxs"
            sample_path = root / "sample_scattering.nxs"
            empty_cell_path = root / "empty_cell_scattering.nxs"

            empty_beam_data = np.zeros((4, 4), dtype=np.float64)
            empty_beam_data[1:3, 1:3] = 100.0
            sample_data = np.zeros((4, 4), dtype=np.float64)
            sample_data[1:3, 1:3] = 40.0
            empty_cell_data = np.zeros((4, 4), dtype=np.float64)
            empty_cell_data[1:3, 1:3] = 50.0

            _write_transmission_file(empty_beam_path, data=empty_beam_data, monitor_integral=10.0)
            _write_transmission_file(sample_path, data=sample_data, monitor_integral=20.0)
            _write_transmission_file(empty_cell_path, data=empty_cell_data, monitor_integral=10.0)

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.set_transmission_strategy("semi_transparent_beamstop")
            ctx.set_empty_beam("config_1", "scattering", empty_beam_path)
            ctx.set_roi("config_1", (1, 3, 1, 3))
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="empty_cell", mode="scattering", sample_name="empty_cell"),
                empty_cell_path,
            )

            transmissions = ctx.compute_transmissions()

            self.assertEqual(transmissions, {("sample_a", "config_1"): 0.2, ("empty_cell", "config_1"): 0.5})
            self.assertAlmostEqual(ctx.get_transmission("sample_a", "config_1"), 0.2)
            self.assertAlmostEqual(ctx.get_empty_cell_transmission("config_1"), 0.5)

    def test_initialize_transmission_geometry_uses_scattering_roi_and_transmission_center_for_semi_transparent_beamstop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            empty_beam_scattering = root / "empty_beam_scattering.nxs"
            empty_beam_transmission = root / "empty_beam_transmission.nxs"

            scattering_data = np.zeros((4, 4), dtype=np.float64)
            scattering_data[1:3, 1:3] = 100.0
            transmission_data = np.zeros((4, 4), dtype=np.float64)
            transmission_data[0, 3] = 200.0

            _write_transmission_file(empty_beam_scattering, data=scattering_data, monitor_integral=1.0)
            _write_transmission_file(empty_beam_transmission, data=transmission_data, monitor_integral=1.0)

            ctx = WorkflowContext(output_dir=root / "out")
            ctx.set_transmission_strategy("semi_transparent_beamstop")
            ctx.add_run(
                RunKey(config_id="config_1", entity="empty_beam", mode="scattering", sample_name="EB"),
                empty_beam_scattering,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="empty_beam", mode="transmission", sample_name="EB"),
                empty_beam_transmission,
            )

            _initialize_transmission_geometry(ctx, where="test")

            self.assertEqual(ctx.get_roi("config_1"), (0, 4, 0, 4))
            self.assertEqual(ctx.get_beam_center("config_1", 0), (3.0, 0.0))

    def test_initialize_workflow_context_detects_water_entity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_dir = root / "raw"
            output_dir = root / "out"
            raw_dir.mkdir()

            raw_path = raw_dir / "water_raw.h5"
            _write_minimal_raw_nexus_file(raw_path, sample_name="water", count_time_s=10.0)
            _write_sample_thickness(raw_path, 1.0)

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="cfg",
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
            ):
                ctx = initialize_workflow_context_from_raw_directory(
                    raw_dir,
                    output_dir=output_dir,
                    instrument_name="sansllb",
                )

            self.assertIn(
                RunKey(config_id="config_1", entity="water", mode="scattering", sample_name="water"),
                ctx.runs,
            )
            self.assertEqual(ctx.get_water("config_1", "scattering"), (output_dir / "water_raw.nxs").resolve())
            self.assertEqual(ctx.get_reference_file("water", "scattering", "config_1"), (output_dir / "water_raw.nxs").resolve())
            self.assertAlmostEqual(ctx.get_sample_thickness("water", "config_1"), 1.0)

    def test_initialize_workflow_context_converts_meter_sample_thickness_to_mm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_dir = root / "raw"
            output_dir = root / "out"
            raw_dir.mkdir()

            raw_path = raw_dir / "sample_raw.h5"
            _write_minimal_raw_nexus_file(raw_path, sample_name="sample_a", count_time_s=10.0)
            _write_sample_thickness(raw_path, 0.002, units="m")

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="cfg",
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
            ):
                ctx = initialize_workflow_context_from_raw_directory(
                    raw_dir,
                    output_dir=output_dir,
                    instrument_name="sansllb",
                )

            self.assertAlmostEqual(ctx.get_sample_thickness("sample_a", "config_1"), 2.0)

    def test_initialize_workflow_context_forces_scattering_mode_with_semi_transparent_beamstop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_dir = root / "raw"
            output_dir = root / "out"
            raw_dir.mkdir()

            raw_path = raw_dir / "sample_raw.h5"
            _write_minimal_raw_nexus_file(raw_path, sample_name="sample_a", count_time_s=10.0)

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="cfg",
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
                    return_value=SimpleNamespace(mode="transmission"),
                ) as mode_guess_mock,
                mock.patch(
                    "scarlet.workflow.configuration.configuration_from_nexus",
                    return_value=(configuration, []),
                ),
            ):
                ctx = initialize_workflow_context_from_raw_directory(
                    raw_dir,
                    output_dir=output_dir,
                    instrument_name="sansllb",
                    transmission_strategy="semi_transparent_beamstop",
                )

            self.assertIn(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="sample_a"),
                ctx.runs,
            )
            mode_guess_mock.assert_not_called()

    def test_initialize_workflow_context_keeps_empty_beam_transmission_in_semi_transparent_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_dir = root / "raw"
            output_dir = root / "out"
            raw_dir.mkdir()

            raw_path = raw_dir / "empty_beam_raw.h5"
            _write_minimal_raw_nexus_file(raw_path, sample_name="EB", count_time_s=10.0)

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="cfg",
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
                    return_value=SimpleNamespace(mode="transmission"),
                ) as mode_guess_mock,
                mock.patch(
                    "scarlet.workflow.configuration.configuration_from_nexus",
                    return_value=(configuration, []),
                ),
            ):
                ctx = initialize_workflow_context_from_raw_directory(
                    raw_dir,
                    output_dir=output_dir,
                    instrument_name="sansllb",
                    transmission_strategy="semi_transparent_beamstop",
                )

            self.assertIn(
                RunKey(config_id="config_1", entity="empty_beam", mode="transmission", sample_name="EB"),
                ctx.runs,
            )
            mode_guess_mock.assert_called_once()
