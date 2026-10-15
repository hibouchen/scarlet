from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from scarlet.gui.jobs import JobController
from scarlet.gui.main_window import ScarletMainWindow
from scarlet.gui.mask_editor import MaskEditorSource
from scarlet.gui.models import (
    ConfigurationSettingsModel,
    ProcessingRunsModel,
    ReferenceTransmissionsModel,
    RunsTableModel,
    configuration_settings_rows,
)
from scarlet.gui.pages import (
    AcquisitionsPage,
    ProcessingPage,
    ResultsPage,
    SettingsPage,
    StitchingPage,
)
from scarlet.gui.readiness import assess_workflow_readiness
from scarlet.gui.settings_data import (
    SettingsImagePayload,
    detect_settings_geometry,
    load_center_from_empty_beam_file,
    load_settings_image,
)
from scarlet.gui.widgets import SettingsImageView, StitchingOptionsDialog
from scarlet.gui.workflow_controller import WorkflowController
from scarlet.workflow import RunKey, WorkflowContext
from scarlet.workflow.configuration import Configuration
from scarlet.workflow.pipeline import StitchingOptions


_APP: QtWidgets.QApplication | None = None


def _application() -> QtWidgets.QApplication:
    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


def _write_settings_image_file(path: Path, image: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("raw_data")
        entry.attrs["NX_class"] = b"NXentry"
        control = entry.create_group("control")
        control.create_dataset("integral", data=100.0)
        detector = entry.create_group("instrument").create_group("detector0")
        detector.attrs["NX_class"] = b"NXdetector"
        detector.create_dataset("data", data=np.asarray(image, dtype=np.float64))
        detector.create_dataset("deadtime_corrected", data=True)
        detector.create_dataset("beam_center_x", data=2.25)
        detector.create_dataset("beam_center_y", data=1.75)


class TestRunsTableModel(unittest.TestCase):
    def test_model_exposes_workflow_runs_as_read_only_rows(self) -> None:
        _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "sample.hdf"
            raw_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            workflow.add_run(
                RunKey(
                    config_id="config_1",
                    entity="sample",
                    mode="scattering",
                    sample_name="sample-a",
                ),
                raw_path,
            )

            model = RunsTableModel()
            model.set_workflow(workflow)

            self.assertEqual(model.rowCount(), 1)
            self.assertEqual(model.columnCount(), 8)
            self.assertEqual(model.data(model.index(0, 0)), "sample-a")
            self.assertEqual(
                model.data(model.index(0, model.column_index("status"))),
                "missing",
            )
            self.assertTrue(model.flags(model.index(0, 0)) & QtCore.Qt.ItemFlag.ItemIsEditable)
            self.assertTrue(model.flags(model.index(0, 2)) & QtCore.Qt.ItemFlag.ItemIsEditable)
            self.assertTrue(
                model.flags(model.index(0, model.column_index("thickness")))
                & QtCore.Qt.ItemFlag.ItemIsEditable
            )

            changes: list[tuple[RunKey, str]] = []
            model.mode_change_requested.connect(
                lambda run_key, mode: changes.append((run_key, mode))
            )
            changed = model.setData(
                model.index(0, 2),
                "transmission",
                QtCore.Qt.ItemDataRole.EditRole,
            )

            self.assertTrue(changed)
            self.assertEqual(changes, [(next(iter(workflow.runs)), "transmission")])
            self.assertEqual(model.data(model.index(0, 2)), "transmission")

            entity_changes: list[tuple[RunKey, str]] = []
            model.entity_change_requested.connect(
                lambda run_key, entity: entity_changes.append((run_key, entity))
            )
            entity_index = model.index(0, model.column_index("entity"))
            self.assertTrue(
                model.setData(
                    entity_index,
                    "water",
                    QtCore.Qt.ItemDataRole.EditRole,
                )
            )
            self.assertEqual(entity_changes[0][1], "water")

    def test_sample_name_edit_requires_non_empty_name(self) -> None:
        _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "sample.hdf"
            raw_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            key = RunKey("config_1", "sample", "scattering", "old-name")
            workflow.add_run(key, raw_path)
            model = RunsTableModel()
            model.set_workflow(workflow)
            index = model.index(0, model.column_index("sample_name"))
            changes: list[tuple[RunKey, str]] = []
            model.sample_name_change_requested.connect(
                lambda run_key, value: changes.append((run_key, value))
            )

            self.assertFalse(model.setData(index, "  ", QtCore.Qt.ItemDataRole.EditRole))
            self.assertTrue(model.setData(index, "new-name", QtCore.Qt.ItemDataRole.EditRole))
            self.assertEqual(model.data(index), "new-name")
            self.assertEqual(changes, [(key, "new-name")])

    def test_thickness_edit_accepts_positive_or_empty_and_rejects_other_values(self) -> None:
        _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "sample.hdf"
            raw_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            key = RunKey("config_1", "sample", "scattering", "sample-a")
            workflow.add_run(key, raw_path)
            workflow.set_beam_center("config_1", 0, (2.0, 1.5))
            model = RunsTableModel()
            model.set_workflow(workflow)
            index = model.index(0, model.column_index("thickness"))
            changes: list[tuple[RunKey, float | None]] = []
            model.thickness_change_requested.connect(
                lambda run_key, value: changes.append((run_key, value))
            )

            self.assertTrue(model.setData(index, "1,25", QtCore.Qt.ItemDataRole.EditRole))
            self.assertEqual(model.data(index), "1.25")
            self.assertEqual(changes[-1], (key, 1.25))

            for invalid in ("0", "-1", "not-a-number", "nan", "inf"):
                with self.subTest(invalid=invalid):
                    self.assertFalse(
                        model.setData(index, invalid, QtCore.Qt.ItemDataRole.EditRole)
                    )
                    self.assertEqual(model.data(index), "1.25")

            self.assertTrue(model.setData(index, "", QtCore.Qt.ItemDataRole.EditRole))
            self.assertEqual(model.data(index), "")
            self.assertEqual(changes[-1], (key, None))

    def test_thickness_is_not_editable_for_reference_rows(self) -> None:
        _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "empty-beam.hdf"
            raw_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            workflow.add_run(
                RunKey("config_1", "empty_beam", "scattering", "EB"),
                raw_path,
            )
            model = RunsTableModel()
            model.set_workflow(workflow)
            index = model.index(0, model.column_index("thickness"))

            self.assertFalse(model.flags(index) & QtCore.Qt.ItemFlag.ItemIsEditable)
            self.assertFalse(model.setData(index, "1", QtCore.Qt.ItemDataRole.EditRole))

    def test_uncertain_mode_is_marked_for_review_until_corrected(self) -> None:
        _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "uncertain.hdf"
            raw_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            key = RunKey("config_1", "sample", "scattering", "sample-a")
            workflow.add_run(key, raw_path)
            workflow.warn(
                "Could not confidently infer measurement mode; using heuristic fallback",
                where="test",
                key=str(raw_path),
                guessed_mode="scattering",
            )
            model = RunsTableModel()
            model.set_workflow(workflow)
            review_index = model.index(0, model.column_index("review_warning"))

            self.assertEqual(model.data(review_index), "⚠")
            self.assertEqual(model.warning_count, 1)
            self.assertIn("heuristic fallback", model.data(review_index, QtCore.Qt.ItemDataRole.ToolTipRole))

            workflow.replace_run(key, RunKey("config_1", "sample", "transmission", "sample-a"))
            model.set_workflow(workflow)
            review_index = model.index(0, model.column_index("review_warning"))

            self.assertEqual(model.data(review_index), "")
            self.assertEqual(model.warning_count, 0)

    def test_single_click_opens_mode_combo_box(self) -> None:
        app = _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "sample.hdf"
            raw_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            workflow.add_run(
                RunKey("config_1", "sample", "scattering", "sample-a"),
                raw_path,
            )
            page = AcquisitionsPage()
            page.set_workflow(workflow)
            page.resize(900, 400)
            page.show()
            app.processEvents()

            index = page.sort_model.index(0, page.model.column_index("mode"))
            QtTest.QTest.mouseClick(
                page.table.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                pos=page.table.visualRect(index).center(),
            )
            app.processEvents()

            editor = page.table.findChild(QtWidgets.QComboBox)
            self.assertIsNotNone(editor)
            self.assertTrue(editor.isVisible())
            page.close()

    def test_clicking_column_header_toggles_row_sorting(self) -> None:
        app = _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            for sample_name in ("sample-10", "sample-2"):
                raw_path = root / f"{sample_name}.hdf"
                raw_path.write_bytes(b"raw")
                workflow.add_run(
                    RunKey("config_1", "sample", "scattering", sample_name),
                    raw_path,
                )
            page = AcquisitionsPage()
            page.set_workflow(workflow)
            page.resize(900, 400)
            page.show()
            app.processEvents()

            sample_column = page.model.column_index("sample_name")
            header = page.table.horizontalHeader()
            self.assertEqual(
                header.sectionResizeMode(sample_column),
                QtWidgets.QHeaderView.ResizeMode.Interactive,
            )
            previous_width = header.sectionSize(sample_column)
            header.resizeSection(sample_column, previous_width + 40)
            self.assertEqual(header.sectionSize(sample_column), previous_width + 40)
            click_position = QtCore.QPoint(
                header.sectionViewportPosition(sample_column)
                + header.sectionSize(sample_column) // 2,
                header.height() // 2,
            )
            QtTest.QTest.mouseClick(
                header.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                pos=click_position,
            )
            app.processEvents()
            first_order = [
                page.sort_model.data(page.sort_model.index(row, sample_column))
                for row in range(page.sort_model.rowCount())
            ]

            QtTest.QTest.mouseClick(
                header.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                pos=click_position,
            )
            app.processEvents()
            second_order = [
                page.sort_model.data(page.sort_model.index(row, sample_column))
                for row in range(page.sort_model.rowCount())
            ]

            self.assertIn(first_order, [["sample-2", "sample-10"], ["sample-10", "sample-2"]])
            self.assertEqual(second_order, list(reversed(first_order)))
            self.assertTrue(header.isSortIndicatorShown())
            page.close()

    def test_shift_selection_excludes_all_selected_sorted_rows(self) -> None:
        app = _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            for sample_name in ("sample-3", "sample-1", "sample-2"):
                raw_path = root / f"{sample_name}.hdf"
                raw_path.write_bytes(b"raw")
                workflow.add_run(
                    RunKey("config_1", "sample", "scattering", sample_name),
                    raw_path,
                )
            page = AcquisitionsPage()
            page.set_workflow(workflow)
            page.resize(900, 400)
            page.show()
            sample_column = page.model.column_index("sample_name")
            page.table.sortByColumn(sample_column, QtCore.Qt.SortOrder.AscendingOrder)
            app.processEvents()

            first = page.sort_model.index(0, sample_column)
            last = page.sort_model.index(2, sample_column)
            QtTest.QTest.mouseClick(
                page.table.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                pos=page.table.visualRect(first).center(),
            )
            QtTest.QTest.mouseClick(
                page.table.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                pos=page.table.visualRect(last).center(),
            )
            app.processEvents()

            emitted: list[list[RunKey]] = []
            page.exclude_requested.connect(lambda keys: emitted.append(keys))
            self.assertEqual(len(page.table.selectionModel().selectedRows()), 3)
            self.assertEqual(page.exclude_button.text(), "Exclude selected (3)")
            page.exclude_button.click()

            self.assertEqual(len(emitted), 1)
            self.assertEqual(set(emitted[0]), set(workflow.runs))
            page.close()

    def test_search_and_field_filters_reduce_visible_rows(self) -> None:
        app = _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            rows = (
                ("alpha", "config_1", "scattering"),
                ("beta", "config_2", "transmission"),
            )
            for sample_name, config_id, mode in rows:
                raw_path = root / f"{sample_name}.hdf"
                raw_path.write_bytes(b"raw")
                workflow.add_run(RunKey(config_id, "sample", mode, sample_name), raw_path)
            alpha_path = workflow.get_run_path(next(key for key in workflow.runs if key.sample_name == "alpha"))
            workflow.warn(
                "Could not confidently infer measurement mode; using heuristic fallback",
                where="test",
                key=str(alpha_path),
                guessed_mode="scattering",
            )
            page = AcquisitionsPage()
            page.set_workflow(workflow)
            page.show()
            app.processEvents()

            page.search_edit.setText("beta")
            self.assertEqual(page.sort_model.rowCount(), 1)
            page.search_edit.clear()

            page.mode_filter.setCurrentIndex(page.mode_filter.findData("transmission"))
            self.assertEqual(page.sort_model.rowCount(), 1)
            page.mode_filter.setCurrentIndex(0)

            page.config_filter.setCurrentIndex(page.config_filter.findData("config_1"))
            self.assertEqual(page.sort_model.rowCount(), 1)
            page.config_filter.setCurrentIndex(0)

            page.warning_only_check.setChecked(True)
            self.assertEqual(page.sort_model.rowCount(), 1)
            page.warning_only_check.setChecked(False)
            self.assertEqual(page.sort_model.rowCount(), 2)
            page.close()

    def test_mask_save_uses_preview_run_after_table_selection_is_reset(self) -> None:
        app = _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "sample.hdf"
            raw_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            key = RunKey("config_1", "sample", "scattering", "sample-a")
            workflow.add_run(key, raw_path)
            source = MaskEditorSource(
                file_path=raw_path,
                entry_path="/raw_data",
                detector_data={0: np.zeros((4, 5), dtype=np.float64)},
                configuration=Configuration(wavelength=6.0, sample_detector_distance=4.0),
                configuration_issues=[],
            )
            page = AcquisitionsPage()
            page.set_workflow(workflow)
            page.show()
            app.processEvents()

            page.set_preview(
                SimpleNamespace(
                    run_key=key,
                    source=source,
                    masks={},
                    beam_centers={0: (2.0, 1.5)},
                )
            )
            self.assertEqual(page.mask_editor._detector_tabs[0].beam_center, (2.0, 1.5))
            page.table.clearSelection()  # Happens after raw-file conversion rebuilds the model.
            emitted: list[tuple[str, MaskEditorSource, dict[int, np.ndarray]]] = []
            page.mask_save_requested.connect(
                lambda config_id, saved_source, masks: emitted.append(
                    (config_id, saved_source, masks)
                )
            )
            page.save_mask_button.click()

            self.assertEqual(len(emitted), 1)
            self.assertEqual(emitted[0][0], "config_1")
            self.assertIs(emitted[0][1], source)
            page.close()

    def test_table_selection_stays_enabled_while_loading_preview(self) -> None:
        _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "sample.hdf"
            raw_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            workflow.add_run(
                RunKey("config_1", "sample", "scattering", "sample-a"), raw_path
            )
            page = AcquisitionsPage()
            page.set_workflow(workflow)

            page.set_busy(True, "Loading acquisition detector image…")

            self.assertTrue(page.table.isEnabled())
            self.assertFalse(page.exclude_button.isEnabled())
            page.close()

    def test_selected_row_is_restored_when_workflow_model_refreshes(self) -> None:
        _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "sample.hdf"
            raw_path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            key = RunKey("config_1", "sample", "scattering", "sample-a")
            workflow.add_run(key, raw_path)
            page = AcquisitionsPage()
            page.set_workflow(workflow)
            page.table.selectRow(0)

            page.set_workflow(workflow)

            selected_rows = page.table.selectionModel().selectedRows()
            self.assertEqual(len(selected_rows), 1)
            source_row = page.sort_model.mapToSource(selected_rows[0]).row()
            self.assertEqual(page.model.run_key_at(source_row), key)
            page.close()


class TestProcessingPage(unittest.TestCase):
    def test_page_adapts_azimuthal_bins_to_the_instrument_detectors(self) -> None:
        _application()
        page = ProcessingPage()

        self.assertEqual(
            page.reduction_options(),
            ({0: 120}, "linear"),
        )

        page.n_bins_spinboxes[0].setValue(40)
        sansllb_workflow = WorkflowContext(instrument_name="SANS-LLB")
        sansllb_workflow.configurations["config_1"] = Configuration(
            wavelength=6.0,
            sample_detector_distance=[4.0, 1.5, 1.5],
            config_id="config_1",
        )
        page.set_workflow(sansllb_workflow)

        self.assertEqual(tuple(page.n_bins_spinboxes), (0, 1, 2))
        self.assertEqual(
            page.reduction_options(),
            ({0: 40, 1: 30, 2: 30}, "linear"),
        )

        page.n_bins_spinboxes[1].setValue(25)
        page.n_bins_spinboxes[2].setValue(20)
        page.q_scale_combo.setCurrentIndex(page.q_scale_combo.findData("log"))

        self.assertEqual(
            page.reduction_options(),
            ({0: 40, 1: 25, 2: 20}, "log"),
        )

        sam_workflow = WorkflowContext(instrument_name="SAM")
        sam_workflow.configurations["config_1"] = Configuration(
            wavelength=6.0,
            sample_detector_distance=4.0,
            config_id="config_1",
        )
        page.set_workflow(sam_workflow)

        self.assertEqual(tuple(page.n_bins_spinboxes), (0,))
        self.assertEqual(page.reduction_options(), ({0: 40}, "log"))
        page.set_busy(True)
        self.assertFalse(page.integration_group.isEnabled())
        page.close()

    def test_reference_transmissions_model_reports_water_and_empty_cell(self) -> None:
        workflow = WorkflowContext()
        workflow.add_run(
            RunKey("config_1", "sample", "scattering", "sample-a"),
            Path("/tmp/sample-a.nxs"),
        )
        workflow.add_run(
            RunKey("config_1", "water", "scattering", "H2O"),
            Path("/tmp/water.nxs"),
        )
        workflow.set_transmission("H2O", "config_1", 0.94)
        workflow.set_empty_cell_transmission("config_1", 0.97)

        model = ReferenceTransmissionsModel()
        model.set_workflow(workflow)

        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.data(model.index(0, 1)), "0.94")
        self.assertEqual(model.data(model.index(0, 2)), "0.97")

    def test_model_only_exposes_sample_scattering_candidates(self) -> None:
        _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            candidates = (
                RunKey("config_1", "sample", "scattering", "sample-a"),
                RunKey("config_1", "sample", "transmission", "sample-a"),
                RunKey("config_1", "water", "scattering", "H2O"),
            )
            for index, key in enumerate(candidates):
                path = root / f"run-{index}.hdf"
                path.write_bytes(b"raw")
                workflow.add_run(key, path)
            workflow.set_transmission("sample-a", "config_1", 0.875)

            model = ProcessingRunsModel()
            model.set_workflow(workflow)

            self.assertEqual(model.rowCount(), 1)
            self.assertEqual(model.run_key_at(0), candidates[0])
            status_index = model.index(0, model.column_index("status"))
            self.assertEqual(model.data(status_index), "pending")
            transmission_index = model.index(0, model.column_index("transmission"))
            self.assertEqual(model.data(transmission_index), "0.875")

    def test_page_supports_multiple_candidate_selection(self) -> None:
        app = _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            workflow.set_transmission_strategy("semi_transparent_beamstop")
            expected: set[RunKey] = set()
            for index, sample_name in enumerate(("sample-a", "sample-b")):
                path = root / f"sample-{index}.hdf"
                path.write_bytes(b"raw")
                key = RunKey("config_1", "sample", "scattering", sample_name)
                workflow.add_run(key, path)
                expected.add(key)
            for entity in ("water", "empty_beam"):
                path = root / f"{entity}.hdf"
                path.write_bytes(b"raw")
                workflow.add_run(
                    RunKey("config_1", entity, "scattering", entity), path
                )

            page = ProcessingPage()
            page.set_workflow(workflow)
            page.show()
            app.processEvents()
            selection = page.table.selectionModel()
            flags = (
                QtCore.QItemSelectionModel.SelectionFlag.Select
                | QtCore.QItemSelectionModel.SelectionFlag.Rows
            )
            selection.select(page.sort_model.index(0, 0), flags)
            selection.select(page.sort_model.index(1, 0), flags)
            app.processEvents()

            self.assertEqual(set(page.selected_run_keys()), expected)
            self.assertEqual(page.selection_label.text(), "2 acquisition(s) selected")
            self.assertEqual(page.process_selected_button.text(), "Process selected (2)")
            self.assertTrue(page.process_selected_button.isEnabled())
            self.assertTrue(page.process_pending_button.isEnabled())
            selected_requests: list[list[RunKey]] = []
            pending_requests: list[bool] = []
            page.process_selected_requested.connect(selected_requests.append)
            page.process_pending_requested.connect(lambda: pending_requests.append(True))
            page.process_selected_button.click()
            page.process_pending_button.click()
            self.assertEqual(set(selected_requests[0]), expected)
            self.assertEqual(pending_requests, [True])
            page.close()

    def test_stitching_page_requests_a_processed_sample(self) -> None:
        app = _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            for config_id in ("config_1", "config_2"):
                path = root / f"{config_id}.nxs"
                with h5py.File(path, "w") as handle:
                    entry = handle.create_group("entry")
                    entry.create_group("control")
                    entry.create_group("instrument").create_group("detector0").create_dataset(
                        "data", data=np.ones((2, 2), dtype=np.float64)
                    )
                    handle.create_group("processed")
                workflow.add_run(
                    RunKey(config_id, "sample", "scattering", "sample-a"), path
                )
            workflow.add_run(
                RunKey("config_1", "sample", "scattering", "sample-b"),
                root / "config_1.nxs",
            )

            page = StitchingPage()
            page.set_workflow(workflow)
            page.show()
            app.processEvents()

            self.assertEqual(page.model.rowCount(), 2)
            self.assertEqual(
                page.model.data(page.model.index(0, page.model.column_index("status"))),
                "ready",
            )
            page.table.selectRow(0)
            self.assertTrue(page.stitch_selected_button.isEnabled())
            self.assertEqual(page.reference_combo.count(), 3)
            page.reference_combo.setCurrentIndex(
                page.reference_combo.findData("config_2")
            )
            requests: list[tuple[list[str], object]] = []
            page.stitch_selected_requested.connect(
                lambda sample_names, reference: requests.append((sample_names, reference))
            )
            page.stitch_selected_button.click()

            self.assertEqual(requests, [(["sample-a"], "config_2")])
            page.table.selectionModel().select(
                page.model.index(1, 0),
                QtCore.QItemSelectionModel.SelectionFlag.Select
                | QtCore.QItemSelectionModel.SelectionFlag.Rows,
            )
            self.assertTrue(page.reference_combo.isEnabled())
            self.assertEqual(page.reference_combo.count(), 2)
            self.assertEqual(page.reference_combo.itemData(1), "config_1")
            page.reference_combo.setCurrentIndex(
                page.reference_combo.findData("config_1")
            )
            page.stitch_selected_button.click()
            self.assertEqual(requests[-1], (["sample-a", "sample-b"], "config_1"))
            output_path = workflow.output_dir / "sample-a_merged.txt"
            output_path.parent.mkdir(parents=True)
            output_path.write_text("merged", encoding="utf-8")
            page.set_workflow(workflow)
            self.assertEqual(
                page.model.data(page.model.index(0, page.model.column_index("status"))),
                "stitched",
            )
            page.close()


class TestResultsPage(unittest.TestCase):
    def test_multiple_selected_text_curves_are_plotted_with_silx(self) -> None:
        app = _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()
            for name, multiplier in (
                ("sample-a_config_config_1.txt", 1.0),
                ("sample-a_merged.txt", 2.0),
            ):
                np.savetxt(
                    output_dir / name,
                    np.asarray(
                        [[0.01, multiplier, 0.1, 0.001], [0.02, 2 * multiplier, 0.2, 0.002]]
                    ),
                    header="q I I_error q_error",
                )
            workflow = WorkflowContext(root_dir=root, output_dir=output_dir)
            page = ResultsPage()
            page.set_workflow(workflow)
            page.show()
            app.processEvents()

            self.assertEqual(page.model.rowCount(), 2)
            selection = page.table.selectionModel()
            flags = (
                QtCore.QItemSelectionModel.SelectionFlag.Select
                | QtCore.QItemSelectionModel.SelectionFlag.Rows
            )
            selection.select(page.sort_model.index(0, 0), flags)
            selection.select(page.sort_model.index(1, 0), flags)
            app.processEvents()

            self.assertEqual(len(page.plot.getAllCurves()), 2)
            self.assertTrue(page.legend_dock.isVisible())
            legend_view = page.legend_dock.findChild(QtWidgets.QAbstractItemView)
            self.assertIsNotNone(legend_view)
            self.assertEqual(
                legend_view.palette().color(QtGui.QPalette.ColorRole.Text).name(),
                "#202124",
            )
            self.assertEqual(page.error_label.text(), "")
            page.close()

    def test_search_filters_result_files_by_name_and_type(self) -> None:
        app = _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()
            for name in ("alpha_config_1.txt", "beta_merged.txt"):
                np.savetxt(
                    output_dir / name,
                    np.asarray([[0.01, 1.0, 0.1, 0.001]]),
                    header="q I I_error q_error",
                )
            workflow = WorkflowContext(root_dir=root, output_dir=output_dir)
            page = ResultsPage()
            page.set_workflow(workflow)
            page.show()
            app.processEvents()

            self.assertEqual(page.sort_model.rowCount(), 2)
            page.search_edit.setText("BETA")
            self.assertEqual(page.sort_model.rowCount(), 1)
            self.assertEqual(
                page.sort_model.data(page.sort_model.index(0, 0)),
                "beta_merged.txt",
            )

            page.search_edit.setText("reduced")
            self.assertEqual(page.sort_model.rowCount(), 1)
            self.assertEqual(
                page.sort_model.data(page.sort_model.index(0, 0)),
                "alpha_config_1.txt",
            )

            page.search_edit.clear()
            self.assertEqual(page.sort_model.rowCount(), 2)
            page.close()


class TestSettingsPage(unittest.TestCase):
    def test_settings_image_loader_uses_empty_beam_and_existing_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "empty-beam.nxs"
            transmission_path = root / "empty-beam-transmission.nxs"
            image = np.arange(20, dtype=np.float64).reshape(4, 5)
            _write_settings_image_file(image_path, image)
            _write_settings_image_file(transmission_path, image + 10.0)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            workflow.set_transmission_strategy("semi_transparent_beamstop")
            workflow.add_run(
                RunKey("config_1", "empty_beam", "scattering", "EB"),
                image_path,
            )
            workflow.add_run(
                RunKey("config_1", "empty_beam", "transmission", "EB"),
                transmission_path,
            )
            workflow.set_beam_center("config_1", 0, (2.0, 1.5))
            workflow.set_roi("config_1", (1, 4, 1, 3))

            payload = load_settings_image(
                workflow,
                config_id="config_1",
                purpose="center",
                detector_number=0,
            )

            self.assertEqual(payload.source_path, transmission_path.resolve())
            self.assertEqual(payload.center, (2.0, 1.5))
            self.assertEqual(payload.roi, (1, 4, 1, 3))
            np.testing.assert_allclose(payload.image, (image + 10.0) / 100.0)

            detected_center = detect_settings_geometry(
                workflow,
                config_id="config_1",
                purpose="center",
                detector_number=0,
            )
            detected_roi = detect_settings_geometry(
                workflow,
                config_id="config_1",
                purpose="roi",
                detector_number=0,
            )
            self.assertIsNotNone(detected_center.center)
            self.assertNotEqual(detected_center.center, payload.center)
            self.assertIsNotNone(detected_roi.roi)
            x0, x1, y0, y1 = detected_roi.roi
            self.assertLess(x0, x1)
            self.assertLess(y0, y1)

            stored_center = load_center_from_empty_beam_file(
                workflow,
                config_id="config_1",
                detector_number=0,
            )
            self.assertEqual(stored_center.center, (2.25, 1.75))

    def test_image_view_emits_center_and_roi_graphical_edits(self) -> None:
        app = _application()
        view = SettingsImageView()
        view.resize(400, 300)
        view.set_image(np.arange(100, dtype=np.float64).reshape(10, 10))
        view.show()
        app.processEvents()
        centers: list[tuple[float, float]] = []
        rois: list[tuple[int, int, int, int]] = []
        view.center_edited.connect(lambda x, y: centers.append((x, y)))
        view.roi_edited.connect(lambda roi: rois.append(roi))

        QtTest.QTest.mouseClick(
            view,
            QtCore.Qt.MouseButton.LeftButton,
            pos=view.rect().center(),
        )
        self.assertEqual(len(centers), 1)
        self.assertAlmostEqual(centers[0][0], 5.0, delta=0.2)
        self.assertAlmostEqual(centers[0][1], 5.0, delta=0.2)

        view.set_edit_mode("roi")
        QtTest.QTest.mousePress(
            view,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(120, 100),
        )
        QtTest.QTest.mouseMove(view, QtCore.QPoint(280, 220))
        QtTest.QTest.mouseRelease(
            view,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(280, 220),
        )
        self.assertEqual(len(rois), 1)
        self.assertLess(rois[0][0], rois[0][1])
        self.assertLess(rois[0][2], rois[0][3])
        view.close()

    def test_configuration_overview_reports_ready_and_incomplete_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            for index, config_id in enumerate(("config_1", "config_2")):
                path = root / f"sample-{index}.hdf"
                path.write_bytes(b"raw")
                workflow.add_run(
                    RunKey(config_id, "sample", "scattering", f"sample-{index}"),
                    path,
                )
            workflow.set_beam_center("config_1", 0, (12.5, 14.5))
            workflow.set_roi("config_1", (10, 20, 11, 21))
            workflow.set_mask("config_1", 0, np.zeros((2, 2), dtype=np.uint8))
            flatfield = root / "flatfield.nxs"
            flatfield.write_bytes(b"flatfield")
            workflow.set_flatfield("config_1", flatfield)

            rows = configuration_settings_rows(workflow)

            self.assertEqual([row.config_id for row in rows], ["config_1", "config_2"])
            self.assertEqual(rows[0].status, "ready")
            self.assertEqual(rows[0].centers_status, "1 defined")
            self.assertEqual(rows[0].roi_status, "defined")
            self.assertEqual(rows[0].masks_status, "1 defined")
            self.assertEqual(rows[0].flatfield_status, "ready")
            self.assertEqual(rows[1].status, "incomplete")
            self.assertEqual(rows[1].centers_status, "missing")

    def test_configuration_with_only_excluded_runs_is_not_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            active_path = root / "active.hdf"
            excluded_path = root / "excluded.hdf"
            active_path.write_bytes(b"raw")
            excluded_path.write_bytes(b"raw")
            workflow.add_run(
                RunKey("config_active", "sample", "scattering", "active"),
                active_path,
            )
            excluded_key = RunKey(
                "config_excluded",
                "sample",
                "scattering",
                "excluded",
            )
            workflow.add_run(excluded_key, excluded_path)
            workflow.set_beam_center("config_excluded", 0, (1.0, 1.0))
            workflow.set_roi("config_excluded", (0, 2, 0, 2))

            workflow.exclude_run(excluded_key)
            rows = configuration_settings_rows(workflow)

            self.assertEqual([row.config_id for row in rows], ["config_active"])

    def test_settings_page_selects_a_configuration_and_shows_details(self) -> None:
        app = _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.hdf"
            path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            workflow.add_run(
                RunKey("config_1", "sample", "scattering", "sample-a"),
                path,
            )
            workflow.configurations["config_1"] = Configuration(
                wavelength=6.0,
                sample_detector_distance=[4.0, 1.5],
                config_id="config_1",
                notes="test configuration",
            )
            workflow.set_beam_center("config_1", 0, (2.5, 3.5))
            workflow.set_roi("config_1", (1, 4, 2, 5))

            page = SettingsPage()
            page.set_workflow(workflow)
            page.show()
            app.processEvents()

            self.assertIsInstance(page.model, ConfigurationSettingsModel)
            self.assertEqual(page.model.rowCount(), 1)
            self.assertEqual(page.configuration_label.text(), "config_1")
            self.assertIn("Wavelength: 6 Å", page.characteristics_label.text())
            self.assertIn("detector1: 1.5 m", page.characteristics_label.text())
            self.assertIn("test configuration", page.characteristics_label.text())
            self.assertIn("detector0", page.centers_label.text())
            self.assertEqual(page.roi_label.text(), "x=1:4, y=2:5")
            self.assertIn("1 configuration(s)", page.summary_label.text())
            self.assertEqual(
                [page.settings_tabs.tabText(index) for index in range(page.settings_tabs.count())],
                ["Beam center and ROI", "Transmission", "Mask", "Flatfield"],
            )
            header = page.table.horizontalHeader()
            self.assertEqual(
                header.sectionResizeMode(0),
                QtWidgets.QHeaderView.ResizeMode.Interactive,
            )
            previous_width = header.sectionSize(0)
            header.resizeSection(0, previous_width + 40)
            self.assertEqual(header.sectionSize(0), previous_width + 40)
            self.assertIn("No detector mask", page.mask_tab_label.text())

            center_changes: list[tuple[str, int, tuple[float, float]]] = []
            page.beam_center_change_requested.connect(
                lambda config, detector, center: center_changes.append(
                    (config, detector, center)
                )
            )
            page.center_x_spin.setValue(3.0)
            page.save_center_button.click()
            self.assertEqual(center_changes, [("config_1", 0, (3.0, 3.5))])

            flatfield_requests: list[str] = []
            page.flatfield_build_requested.connect(flatfield_requests.append)
            self.assertEqual(page.build_flatfield_button.text(), "Create flatfield")
            page.build_flatfield_button.click()
            self.assertEqual(flatfield_requests, ["config_1"])

            detections: list[tuple[str, str, int]] = []
            page.auto_detect_requested.connect(
                lambda config, purpose, detector: detections.append(
                    (config, purpose, detector)
                )
            )
            page.detect_center_button.click()
            page.purpose_combo.setCurrentIndex(page.purpose_combo.findData("roi"))
            page.detect_roi_button.click()
            self.assertEqual(
                detections,
                [
                    ("config_1", "center", 0),
                    ("config_1", "roi", 0),
                ],
            )

            file_requests: list[tuple[str, int]] = []
            page.center_file_requested.connect(
                lambda config, detector: file_requests.append((config, detector))
            )
            page.load_center_button.click()
            self.assertEqual(file_requests, [("config_1", 0)])

            payload = SettingsImagePayload(
                config_id="config_1",
                source_config_id="config_1",
                purpose="center",
                detector_number=0,
                source_path=path,
                image=np.ones((8, 10), dtype=np.float64),
                center=(3.0, 3.5),
                roi=(1, 4, 2, 5),
            )
            page.purpose_combo.setCurrentIndex(page.purpose_combo.findData("center"))
            page.set_image(payload)
            self.assertEqual(page.image_view.image_shape, (8, 10))
            self.assertIn("10 × 8", page.image_source_label.text())
            page.center_x_spin.setValue(-12.5)
            page.center_y_spin.setValue(25.0)
            self.assertEqual(page.center_x_spin.value(), -12.5)
            self.assertEqual(page.center_y_spin.value(), 25.0)
            page.close()

    def test_settings_page_requests_flatfield_file_or_source_configuration(self) -> None:
        app = _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            for config_id in ("config_1", "config_2"):
                path = root / f"{config_id}.hdf"
                path.write_bytes(b"raw")
                workflow.add_run(
                    RunKey(config_id, "sample", "scattering", config_id), path
                )
            page = SettingsPage()
            page.set_workflow(workflow)
            page.show()
            app.processEvents()

            file_requests: list[str] = []
            source_changes: list[tuple[str, str]] = []
            transmission_source_changes: list[tuple[str, str]] = []
            mask_file_requests: list[str] = []
            mask_find_requests: list[str] = []
            page.flatfield_file_requested.connect(file_requests.append)
            page.flatfield_source_change_requested.connect(
                lambda config_id, source_config_id: source_changes.append(
                    (config_id, source_config_id)
                )
            )
            page.transmission_source_change_requested.connect(
                lambda config_id, source_config_id: transmission_source_changes.append(
                    (config_id, source_config_id)
                )
            )
            page.mask_file_requested.connect(mask_file_requests.append)
            page.mask_auto_find_requested.connect(mask_find_requests.append)
            page.choose_flatfield_button.click()
            page.flatfield_source_combo.setCurrentIndex(
                page.flatfield_source_combo.findData("config_2")
            )
            page.use_flatfield_source_button.click()
            page.transmission_source_combo.setCurrentIndex(
                page.transmission_source_combo.findData("config_2")
            )
            page.use_transmission_source_button.click()
            page.choose_mask_button.click()
            page.find_mask_button.click()

            self.assertEqual(file_requests, ["config_1"])
            self.assertEqual(source_changes, [("config_1", "config_2")])
            self.assertEqual(transmission_source_changes, [("config_1", "config_2")])
            self.assertEqual(mask_file_requests, ["config_1"])
            self.assertEqual(mask_find_requests, ["config_1"])
            page.close()


class TestJobController(unittest.TestCase):
    def test_operation_runs_and_returns_result(self) -> None:
        _application()
        jobs = JobController()
        event_loop = QtCore.QEventLoop()
        received: list[tuple[str, object]] = []
        jobs.succeeded.connect(lambda kind, result: received.append((kind, result)))
        jobs.finished.connect(lambda _kind: event_loop.quit())

        jobs.start("test", lambda: 42)
        QtCore.QTimer.singleShot(5000, event_loop.quit)
        event_loop.exec()

        self.assertEqual(received, [("test", 42)])
        self.assertFalse(jobs.is_running)


class TestWorkflowController(unittest.TestCase):
    def test_stitching_reference_is_forwarded_for_multiple_samples(self) -> None:
        controller = WorkflowController()
        workflow = WorkflowContext()
        controller._workflow = workflow

        with mock.patch.object(controller, "_start_stitching") as start_stitching:
            controller.stitch_selected(["sample-a", "sample-b"], "config_1")

        start_stitching.assert_called_once_with(
            workflow,
            ("sample-a", "sample-b"),
            "config_1",
            kind="stitch_selected",
        )

    def test_processing_options_are_forwarded_to_reduction(self) -> None:
        controller = WorkflowController()
        workflow = WorkflowContext()
        run_key = RunKey("config_1", "sample", "scattering", "sample-a")
        workflow.add_run(run_key, Path("/tmp/sample-a.nxs"))
        controller._workflow = workflow

        with mock.patch.object(controller, "_start_reduction") as start_reduction:
            controller.process_selected(
                [run_key],
                n_bins={0: 40, 1: 25, 2: 20},
                q_scale="log",
            )

        start_reduction.assert_called_once_with(
            workflow,
            (run_key,),
            kind="reduce_selected",
            n_bins={0: 40, 1: 25, 2: 20},
            q_scale="log",
        )

    def test_stitching_options_are_persisted_in_the_workflow(self) -> None:
        controller = WorkflowController()
        controller._workflow = WorkflowContext()
        options = StitchingOptions(min_points=12, max_chi2_red=4.5)

        controller.set_stitching_options(options)

        self.assertEqual(controller.stitching_options(), options)
        self.assertTrue(controller.is_dirty)

    def test_stitching_options_dialog_exposes_algorithm_parameters(self) -> None:
        _application()
        options = StitchingOptions(
            start_policy="best_low_q_quality",
            grid="b",
            n_common=20,
            min_points=11,
            keep_fraction=0.4,
        )
        dialog = StitchingOptionsDialog(options)

        self.assertEqual(dialog.options(), options)
        dialog.restore_defaults_button.click()
        self.assertEqual(dialog.options(), StitchingOptions())
        dialog.close()

    def test_controller_sets_transmission_source_and_notifies_settings(self) -> None:
        controller = WorkflowController()
        workflow = WorkflowContext()
        controller._workflow = workflow
        changed: list[str] = []
        controller.transmission_source_changed.connect(changed.append)

        controller.set_transmission_source("config_target", "config_source")

        self.assertEqual(
            workflow.get_transmission_source("config_target"), "config_source"
        )
        self.assertEqual(changed, ["config_target"])
        self.assertTrue(controller.is_dirty)

    def test_controller_scans_and_saves_workflow_in_background(self) -> None:
        _application()
        with tempfile.TemporaryDirectory() as tmp:
            controller = WorkflowController()
            failures: list[str] = []
            controller.failed.connect(lambda message, _details: failures.append(message))

            scan_loop = QtCore.QEventLoop()
            controller.busy_changed.connect(
                lambda busy, _message: scan_loop.quit() if not busy else None
            )
            controller.create_workflow(
                input_dir=Path("tests/data/sam/raw_data"),
                output_dir=Path(tmp) / "processed",
                experiment_id="gui-test",
                instrument_name="sam",
                transmission_strategy="opaque_beamstop",
            )
            QtCore.QTimer.singleShot(5000, scan_loop.quit)
            scan_loop.exec()

            self.assertEqual(failures, [])
            self.assertIsNotNone(controller.workflow)
            self.assertGreater(len(controller.workflow.runs), 0)
            self.assertTrue(controller.is_dirty)

            old_key = next(iter(controller.workflow.runs))
            old_path = controller.workflow.runs[old_key]
            new_mode = "transmission" if old_key.mode == "scattering" else "scattering"
            controller.update_run_mode(old_key, new_mode)
            updated_key = next(
                key
                for key, path in controller.workflow.runs.items()
                if path == old_path and key.mode == new_mode
            )
            self.assertEqual(updated_key.mode, new_mode)
            controller.rename_sample(updated_key, "renamed-sample")
            updated_key = next(
                key
                for key, path in controller.workflow.runs.items()
                if path == old_path and key.sample_name == "renamed-sample"
            )
            controller.update_run_entity(updated_key, "water")
            updated_key = next(
                key
                for key, path in controller.workflow.runs.items()
                if path == old_path and key.entity == "water"
            )
            self.assertEqual(updated_key.entity, "water")
            controller.update_run_thickness(updated_key, 2.5)
            self.assertEqual(
                controller.workflow.get_sample_thickness(
                    updated_key.sample_name,
                    updated_key.config_id,
                ),
                2.5,
            )

            csv_path = Path(tmp) / "runs.csv"
            export_loop = QtCore.QEventLoop()
            controller.busy_changed.connect(
                lambda busy, _message: export_loop.quit() if not busy else None
            )
            controller.export_runs_table(csv_path)
            QtCore.QTimer.singleShot(5000, export_loop.quit)
            export_loop.exec()
            self.assertEqual(failures, [])
            self.assertTrue(csv_path.exists())

            import_loop = QtCore.QEventLoop()
            controller.busy_changed.connect(
                lambda busy, _message: import_loop.quit() if not busy else None
            )
            controller.import_runs_table(csv_path)
            QtCore.QTimer.singleShot(5000, import_loop.quit)
            import_loop.exec()
            self.assertEqual(failures, [])
            self.assertTrue(controller.is_dirty)

            save_loop = QtCore.QEventLoop()
            controller.busy_changed.connect(
                lambda busy, _message: save_loop.quit() if not busy else None
            )
            workflow_path = Path(tmp) / "workflow.nxs"
            controller.save_workflow(workflow_path)
            QtCore.QTimer.singleShot(5000, save_loop.quit)
            save_loop.exec()

            self.assertEqual(failures, [])
            self.assertTrue(workflow_path.exists())
            self.assertFalse(controller.is_dirty)
            loaded = WorkflowContext.load(workflow_path)
            self.assertIn(updated_key, loaded.runs)
            self.assertEqual(
                loaded.get_sample_thickness(updated_key.sample_name, updated_key.config_id),
                2.5,
            )


class TestWorkflowReadiness(unittest.TestCase):
    def test_empty_workflow_is_not_ready(self) -> None:
        workflow = WorkflowContext()

        report = assess_workflow_readiness(workflow)

        self.assertFalse(report.ready)
        self.assertIn("No sample scattering", report.errors[0])

    def test_complete_semi_transparent_inputs_are_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            workflow.set_transmission_strategy("semi_transparent_beamstop")
            for entity, sample_name in (
                ("sample", "sample-a"),
                ("water", "H2O"),
                ("empty_beam", "EB"),
            ):
                path = root / f"{entity}.hdf"
                path.write_bytes(b"raw")
                workflow.add_run(
                    RunKey("config_1", entity, "scattering", sample_name),
                    path,
                )
            workflow.set_sample_thickness("sample-a", "config_1", 1.5)

            report = assess_workflow_readiness(workflow)

            self.assertTrue(report.ready)
            self.assertEqual(report.errors, ())
            self.assertEqual(report.warnings, ())

    def test_missing_thickness_is_a_warning_but_missing_references_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.hdf"
            path.write_bytes(b"raw")
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            workflow.set_transmission_strategy("semi_transparent_beamstop")
            workflow.add_run(
                RunKey("config_1", "sample", "scattering", "sample-a"),
                path,
            )

            report = assess_workflow_readiness(workflow)

            self.assertFalse(report.ready)
            self.assertTrue(any("thickness" in warning for warning in report.warnings))
            self.assertTrue(any("water scattering" in error for error in report.errors))
            self.assertTrue(any("empty-beam scattering" in error for error in report.errors))


class TestScarletMainWindow(unittest.TestCase):
    def test_window_can_be_created_without_a_workflow(self) -> None:
        _application()
        window = ScarletMainWindow()

        self.assertEqual(window.windowTitle(), "SCARLET")
        self.assertEqual(window.pages.count(), 6)
        self.assertFalse(window.save_action.isEnabled())
        self.assertIsInstance(window.settings_page, SettingsPage)
        self.assertIsInstance(window.processing_page, ProcessingPage)
        self.assertIsInstance(window.stitching_page, StitchingPage)
        self.assertIsInstance(window.results_page, ResultsPage)

        window.close()

    def test_workflow_refresh_keeps_the_current_page(self) -> None:
        _application()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowContext(root_dir=root, output_dir=root / "processed")
            path = root / "sample.hdf"
            path.write_bytes(b"raw")
            workflow.add_run(
                RunKey("config_1", "sample", "scattering", "sample-a"), path
            )
            window = ScarletMainWindow()
            window.navigation.setCurrentRow(2)

            window._set_workflow(workflow)

            self.assertEqual(window.navigation.currentRow(), 2)
            window.close()


if __name__ == "__main__":
    unittest.main()
