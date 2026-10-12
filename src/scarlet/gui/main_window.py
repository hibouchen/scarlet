from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from scarlet.workflow import WorkflowContext

from .pages import (
    AcquisitionsPage,
    ExperimentPage,
    ProcessingPage,
    ResultsPage,
    SettingsPage,
    StitchingPage,
)
from .widgets import StitchingOptionsDialog
from .workflow_controller import WorkflowController


class ScarletMainWindow(QtWidgets.QMainWindow):
    """Main window for the workflow-oriented SCARLET desktop application."""

    def __init__(
        self,
        *,
        controller: WorkflowController | None = None,
        initial_raw_directory: str | Path | None = None,
        initial_instrument: str = "sansllb",
    ) -> None:
        super().__init__()
        self.controller = controller or WorkflowController(self)
        self._close_after_save = False
        self.setMinimumSize(1000, 650)
        self.resize(1350, 820)

        self.experiment_page = ExperimentPage()
        self.acquisitions_page = AcquisitionsPage()
        self.settings_page = SettingsPage()
        self.processing_page = ProcessingPage()
        self.stitching_page = StitchingPage()
        self.results_page = ResultsPage()
        if initial_raw_directory is not None:
            self.experiment_page.set_initial_raw_directory(initial_raw_directory)
        self.experiment_page.set_initial_instrument(initial_instrument)

        self._build_actions()
        self._build_central_widget()
        self._build_activity_dock()
        self._connect_signals()
        self._set_workflow(None)
        self._update_title()
        self.statusBar().showMessage("Ready")

    def _build_actions(self) -> None:
        self.open_action = QtGui.QAction("Open workflow…", self)
        self.open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self._choose_workflow)

        self.save_action = QtGui.QAction("Save workflow", self)
        self.save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self._save_workflow)

        self.save_as_action = QtGui.QAction("Save workflow as…", self)
        self.save_as_action.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(lambda: self._save_workflow(save_as=True))

        self.quit_action = QtGui.QAction("Quit", self)
        self.quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        self.quit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        settings_menu = self.menuBar().addMenu("Settings")
        self.stitching_options_action = QtGui.QAction("Stitching options…", self)
        self.stitching_options_action.triggered.connect(self._edit_stitching_options)
        settings_menu.addAction(self.stitching_options_action)

    def _build_central_widget(self) -> None:
        sidebar = QtWidgets.QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)

        brand = QtWidgets.QLabel("SCARLET")
        brand.setObjectName("brand")
        sidebar_layout.addWidget(brand)

        self.navigation = QtWidgets.QListWidget()
        self.navigation.setObjectName("navigation")
        for label in (
            "Experiment",
            "Acquisitions",
            "Settings",
            "Processing",
            "Stitching",
            "Results",
        ):
            self.navigation.addItem(label)
        self.navigation.setCurrentRow(0)
        sidebar_layout.addWidget(self.navigation)

        workflow_group = QtWidgets.QGroupBox("Workflow")
        workflow_layout = QtWidgets.QVBoxLayout(workflow_group)
        self.workflow_summary = QtWidgets.QLabel("No workflow loaded")
        self.workflow_summary.setWordWrap(True)
        workflow_layout.addWidget(self.workflow_summary)
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.clicked.connect(self._save_workflow)
        workflow_layout.addWidget(self.save_button)
        sidebar_layout.addWidget(workflow_group)

        self.pages = QtWidgets.QStackedWidget()
        self.pages.addWidget(self.experiment_page)
        self.pages.addWidget(self.acquisitions_page)
        self.pages.addWidget(self.settings_page)
        self.pages.addWidget(self.processing_page)
        self.pages.addWidget(self.stitching_page)
        self.pages.addWidget(self.results_page)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(sidebar)
        layout.addWidget(self.pages, 1)
        self.setCentralWidget(central)

    def _build_activity_dock(self) -> None:
        self.activity_log = QtWidgets.QPlainTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setMaximumBlockCount(1000)
        self.activity_log.setPlaceholderText("Workflow activity will appear here.")
        dock = QtWidgets.QDockWidget("Activity", self)
        dock.setObjectName("activityDock")
        dock.setWidget(self.activity_log)
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.resize(dock.width(), 170)

    def _connect_signals(self) -> None:
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.experiment_page.create_requested.connect(self._create_workflow)
        self.experiment_page.open_requested.connect(self.open_workflow)
        self.acquisitions_page.refresh_requested.connect(self.controller.refresh_workflow)
        self.acquisitions_page.reconsider_excluded_requested.connect(
            lambda: self.controller.refresh_workflow(include_excluded=True)
        )
        self.acquisitions_page.mode_change_requested.connect(self.controller.update_run_mode)
        self.acquisitions_page.entity_change_requested.connect(self.controller.update_run_entity)
        self.acquisitions_page.thickness_change_requested.connect(
            self.controller.update_run_thickness
        )
        self.acquisitions_page.sample_name_change_requested.connect(
            self.controller.rename_sample
        )
        self.acquisitions_page.exclude_requested.connect(self.controller.exclude_runs)
        self.acquisitions_page.preview_requested.connect(
            self.controller.load_acquisition_preview
        )
        self.acquisitions_page.mask_save_requested.connect(self._save_acquisition_masks)
        self.acquisitions_page.import_csv_requested.connect(self._import_runs_csv)
        self.acquisitions_page.export_csv_requested.connect(self._export_runs_csv)
        self.processing_page.process_selected_requested.connect(
            self.controller.process_selected
        )
        self.processing_page.process_pending_requested.connect(self.controller.process_pending)
        self.stitching_page.stitch_selected_requested.connect(
            self.controller.stitch_selected
        )
        self.stitching_page.stitch_pending_requested.connect(self.controller.stitch_pending)
        self.settings_page.image_requested.connect(self.controller.load_settings_image)
        self.settings_page.auto_detect_requested.connect(
            self.controller.detect_settings_geometry
        )
        self.settings_page.center_file_requested.connect(
            self.controller.load_center_from_empty_beam_file
        )
        self.settings_page.beam_center_change_requested.connect(
            self.controller.update_beam_center
        )
        self.settings_page.roi_change_requested.connect(
            self.controller.update_transmission_roi
        )
        self.settings_page.flatfield_build_requested.connect(
            self.controller.build_water_flatfield
        )
        self.settings_page.flatfield_file_requested.connect(self._choose_flatfield_file)
        self.settings_page.flatfield_source_change_requested.connect(
            self.controller.set_flatfield_source
        )
        self.settings_page.transmission_source_change_requested.connect(
            self.controller.set_transmission_source
        )
        self.settings_page.mask_file_requested.connect(self._choose_mask_file)
        self.settings_page.mask_auto_find_requested.connect(
            self.controller.find_compatible_mask
        )
        self.controller.workflow_changed.connect(self._set_workflow)
        self.controller.workflow_saved.connect(self._finish_close_after_save)
        self.controller.dirty_changed.connect(self._update_title)
        self.controller.busy_changed.connect(self._set_busy)
        self.controller.activity.connect(self._append_activity)
        self.controller.failed.connect(self._cancel_close_after_failed_operation)
        self.controller.failed.connect(self._show_error)
        self.controller.settings_image_loaded.connect(self.settings_page.set_image)
        self.controller.acquisition_preview_loaded.connect(self.acquisitions_page.set_preview)
        self.controller.masks_changed.connect(self._refresh_preparation_state)
        self.controller.flatfield_changed.connect(self._refresh_preparation_state)
        self.controller.transmission_source_changed.connect(self._refresh_preparation_state)
        self.controller.processing_changed.connect(self._refresh_processing_state)

    @QtCore.Slot(object)
    def _create_workflow(self, parameters: object) -> None:
        if isinstance(parameters, dict) and self._confirm_replace_workflow():
            self.controller.create_workflow(**parameters)

    def open_workflow(self, file_path: str | Path) -> None:
        if self._confirm_replace_workflow():
            self.controller.open_workflow(file_path)

    def _confirm_replace_workflow(self) -> bool:
        if not self.controller.is_dirty:
            return True
        answer = QtWidgets.QMessageBox.question(
            self,
            "Unsaved workflow",
            "Discard the unsaved workflow and continue?",
            QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Discard

    @QtCore.Slot()
    def _choose_workflow(self) -> None:
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open SCARLET workflow",
            "",
            "NeXus workflow (*.nxs *.h5 *.hdf5);;All files (*)",
        )
        if selected:
            self.open_workflow(selected)

    @QtCore.Slot()
    def _edit_stitching_options(self) -> None:
        if self.controller.workflow is None:
            return
        dialog = StitchingOptionsDialog(self.controller.stitching_options(), self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.controller.set_stitching_options(dialog.options())

    @QtCore.Slot()
    def _save_workflow(self, *, save_as: bool = False) -> bool:
        if self.controller.workflow is None:
            return False
        path = None if save_as else self.controller.workflow_path
        if path is None:
            suggested = self.controller.workflow.output_dir / "workflow.nxs"
            selected, _filter = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save SCARLET workflow",
                str(suggested),
                "NeXus workflow (*.nxs);;All files (*)",
            )
            if not selected:
                return False
            path = Path(selected)
            if not path.suffix:
                path = path.with_suffix(".nxs")

        overwrite = path.exists()
        if overwrite and (save_as or path != self.controller.workflow_path):
            answer = QtWidgets.QMessageBox.question(
                self,
                "Replace workflow",
                f"Replace the existing file?\n{path}",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        self.controller.save_workflow(path, overwrite=overwrite)
        return True

    @QtCore.Slot()
    def _import_runs_csv(self) -> None:
        workflow = self.controller.workflow
        if workflow is None:
            return
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import acquisitions CSV",
            str(workflow.output_dir),
            "CSV files (*.csv);;All files (*)",
        )
        if selected:
            self.controller.import_runs_table(selected)

    @QtCore.Slot()
    def _export_runs_csv(self) -> None:
        workflow = self.controller.workflow
        if workflow is None:
            return
        selected, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export acquisitions CSV",
            str(workflow.output_dir / "runs_table.csv"),
            "CSV files (*.csv);;All files (*)",
        )
        if not selected:
            return
        path = Path(selected)
        if not path.suffix:
            path = path.with_suffix(".csv")
        self.controller.export_runs_table(path)

    @QtCore.Slot(str, object, object)
    def _save_acquisition_masks(self, config_id: str, source: object, masks: object) -> None:
        workflow = self.controller.workflow
        if workflow is None:
            return
        suggested = workflow.output_dir / f"{config_id}_masks.nxs"
        workflow.output_dir.mkdir(parents=True, exist_ok=True)
        dialog = QtWidgets.QFileDialog(self, "Save and attach mask bundle")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setNameFilters(["NeXus/HDF5 (*.nxs *.h5 *.hdf5)", "All files (*)"])
        dialog.setDefaultSuffix("nxs")
        dialog.setDirectory(str(workflow.output_dir))
        dialog.selectFile(suggested.name)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        selected_files = dialog.selectedFiles()
        if not selected_files:
            return
        path = Path(selected_files[0])
        if not path.suffix:
            path = path.with_suffix(".nxs")
        overwrite = path.exists()
        if overwrite:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Replace mask bundle",
                f"Replace the existing mask bundle?\n{path}",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self.controller.save_acquisition_masks(
            config_id,
            source,
            masks,
            path,
            overwrite=overwrite,
        )

    @QtCore.Slot(str)
    def _choose_flatfield_file(self, config_id: str) -> None:
        workflow = self.controller.workflow
        if workflow is None:
            return
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select SCARLET flatfield",
            str(workflow.output_dir),
            "SCARLET flatfield (*.nxs *.h5 *.hdf5);;All files (*)",
        )
        if selected:
            self.controller.attach_flatfield_file(config_id, selected)

    @QtCore.Slot(str)
    def _choose_mask_file(self, config_id: str) -> None:
        workflow = self.controller.workflow
        if workflow is None:
            return
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select SCARLET mask bundle",
            str(workflow.output_dir),
            "SCARLET masks (*.nxs *.h5 *.hdf5);;All files (*)",
        )
        if selected:
            self.controller.attach_mask_file(config_id, selected)

    @QtCore.Slot(str)
    def _refresh_preparation_state(self, _config_id: str) -> None:
        """Update preparation state while keeping the active silx editor alive."""
        workflow = self.controller.workflow
        self.acquisitions_page.refresh_readiness(workflow)
        self.settings_page.set_workflow(workflow)

    @QtCore.Slot()
    def _refresh_processing_state(self) -> None:
        """Refresh reduction statuses once a background processing job ends."""
        workflow = self.controller.workflow
        self.acquisitions_page.set_workflow(workflow)
        self.processing_page.set_workflow(workflow)
        self.stitching_page.set_workflow(workflow)
        self.results_page.set_workflow(workflow)

    @QtCore.Slot(object)
    def _set_workflow(self, workflow: WorkflowContext | None) -> None:
        self.acquisitions_page.set_workflow(workflow)
        self.settings_page.set_workflow(workflow)
        self.processing_page.set_workflow(workflow)
        self.stitching_page.set_workflow(workflow)
        self.results_page.set_workflow(workflow)
        has_workflow = workflow is not None
        self.save_action.setEnabled(has_workflow)
        self.save_as_action.setEnabled(has_workflow)
        self.save_button.setEnabled(has_workflow)
        self.stitching_options_action.setEnabled(has_workflow)
        for row in range(1, self.navigation.count()):
            item = self.navigation.item(row)
            item.setFlags(
                item.flags() | QtCore.Qt.ItemFlag.ItemIsEnabled
                if has_workflow
                else item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled
            )
        if workflow is None:
            self.workflow_summary.setText("No workflow loaded")
        else:
            self.workflow_summary.setText(
                f"{workflow.experiment_id}\n{workflow.instrument_name.upper()}\n"
                f"{len(workflow.runs)} acquisitions"
            )
            if self.navigation.currentRow() == 0:
                self.navigation.setCurrentRow(1)
            self._show_workflow_logs(workflow)
        self._update_title()

    def _show_workflow_logs(self, workflow: WorkflowContext) -> None:
        self.activity_log.clear()
        for entry in workflow.logs[-100:]:
            where = f" [{entry.where}]" if entry.where else ""
            self.activity_log.appendPlainText(f"{entry.level}{where} {entry.message}")

    @QtCore.Slot(bool, str)
    def _set_busy(self, busy: bool, message: str) -> None:
        self.experiment_page.set_busy(busy)
        self.acquisitions_page.set_busy(busy, message)
        self.settings_page.set_busy(busy)
        self.processing_page.set_busy(busy)
        self.stitching_page.set_busy(busy)
        self.results_page.set_busy(busy)
        self.open_action.setDisabled(busy)
        self.save_action.setDisabled(busy or self.controller.workflow is None)
        self.save_as_action.setDisabled(busy or self.controller.workflow is None)
        self.save_button.setDisabled(busy or self.controller.workflow is None)
        self.stitching_options_action.setDisabled(busy or self.controller.workflow is None)
        self.statusBar().showMessage(message if busy else "Ready")
        if not busy:
            self._finish_close_after_save()

    @QtCore.Slot()
    def _finish_close_after_save(self) -> None:
        if self._close_after_save and not self.controller.is_busy:
            self._close_after_save = False
            self.close()

    @QtCore.Slot(str, str)
    def _cancel_close_after_failed_operation(self, _message: str, _details: str) -> None:
        self._close_after_save = False

    @QtCore.Slot(str, str)
    def _append_activity(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_log.appendPlainText(f"{timestamp} {level} {message}")

    @QtCore.Slot(str, str)
    def _show_error(self, message: str, details: str) -> None:
        dialog = QtWidgets.QMessageBox(self)
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        dialog.setWindowTitle("SCARLET error")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()

    @QtCore.Slot()
    def _update_title(self, *_args: object) -> None:
        name = "SCARLET"
        if self.controller.workflow is not None:
            name = f"{self.controller.workflow.experiment_id} — {name}"
        if self.controller.is_dirty:
            name = f"*{name}"
        self.setWindowTitle(name)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.controller.is_busy:
            QtWidgets.QMessageBox.information(
                self,
                "SCARLET is busy",
                "Wait for the current operation to finish before closing the application.",
            )
            event.ignore()
            return
        if self.controller.is_dirty:
            dialog = QtWidgets.QMessageBox(self)
            dialog.setWindowTitle("Unsaved workflow")
            dialog.setText("Save the workflow before closing?")
            save_and_close = dialog.addButton(
                "Save and close",
                QtWidgets.QMessageBox.ButtonRole.AcceptRole,
            )
            discard = dialog.addButton(QtWidgets.QMessageBox.StandardButton.Discard)
            cancel = dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            dialog.setDefaultButton(save_and_close)
            dialog.exec()
            if dialog.clickedButton() is cancel:
                event.ignore()
                return
            if dialog.clickedButton() is save_and_close:
                self._close_after_save = True
                if not self._save_workflow():
                    self._close_after_save = False
                event.ignore()
                return
            if dialog.clickedButton() is not discard:
                event.ignore()
                return
        event.accept()
