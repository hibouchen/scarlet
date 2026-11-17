from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from scarlet.workflow import WorkflowContext
from scarlet.workflow.pipeline import StichingPipeline

from ..models import StitchingSamplesModel


class StitchingPage(QtWidgets.QWidget):
    """Review stitched outputs and merge selected ready sample groups."""

    stitch_selected_requested = QtCore.Signal(object, object, float, object)
    stitch_pending_requested = QtCore.Signal(float, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._workflow: WorkflowContext | None = None
        self._busy = False

        title = QtWidgets.QLabel("Stitching")
        title.setObjectName("pageTitle")
        intro = QtWidgets.QLabel(
            "The table records merged text files found in the workflow output folder. "
            "Select one or more ready samples to stitch them."
        )
        intro.setWordWrap(True)
        self.summary_label = QtWidgets.QLabel("No workflow loaded")

        self.model = StitchingSamplesModel(parent=self)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(48)
        header.setDefaultSectionSize(150)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)

        self.selection_label = QtWidgets.QLabel("No sample selected")
        self.curve_group_combo = QtWidgets.QComboBox()
        self.curve_group_combo.setObjectName("stitchingCurveGroup")
        self.curve_group_combo.addItem("Isotropic", None)
        self.curve_group_combo.setToolTip(
            "Stitch the isotropic curve, every common sector, or one named sector."
        )
        self.reference_combo = QtWidgets.QComboBox()
        self.normalization_factor_spin = QtWidgets.QDoubleSpinBox()
        self.normalization_factor_spin.setObjectName("stitchingNormalizationFactor")
        self.normalization_factor_spin.setRange(0.000000001, 1_000_000_000.0)
        self.normalization_factor_spin.setDecimals(9)
        self.normalization_factor_spin.setSingleStep(0.1)
        self.normalization_factor_spin.setValue(1.0)
        self.normalization_factor_spin.setToolTip(
            "Multiplicative factor applied to the stitched intensity and its error."
        )
        self.stitch_selected_button = QtWidgets.QPushButton("Stitch selected")
        self.stitch_pending_button = QtWidgets.QPushButton("Stitch pending")

        options = QtWidgets.QHBoxLayout()
        options.addWidget(QtWidgets.QLabel("Curves"))
        options.addWidget(self.curve_group_combo)
        options.addSpacing(16)
        options.addWidget(QtWidgets.QLabel("Reference"))
        options.addWidget(self.reference_combo)
        options.addSpacing(16)
        options.addWidget(QtWidgets.QLabel("Normalization factor"))
        options.addWidget(self.normalization_factor_spin)
        options.addStretch(1)

        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self.selection_label)
        actions.addStretch(1)
        actions.addWidget(self.stitch_selected_button)
        actions.addWidget(self.stitch_pending_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.table, 1)
        layout.addLayout(options)
        layout.addLayout(actions)

        self.curve_group_combo.currentIndexChanged.connect(
            self._curve_group_changed
        )
        self.stitch_selected_button.clicked.connect(self._request_selected_stitching)
        self.stitch_pending_button.clicked.connect(self._request_pending_stitching)
        self._update_controls()

    def set_workflow(self, workflow: WorkflowContext | None) -> None:
        selected_names = set(self.selected_sample_names())
        self._workflow = workflow
        selection_model = self.table.selectionModel()
        blocker = QtCore.QSignalBlocker(selection_model)
        self.model.set_workflow(workflow)
        self._restore_selected_samples(selected_names)
        del blocker

        self._populate_curve_groups(workflow)
        self.model.set_curve_group(self.curve_group())
        self._restore_selected_samples(selected_names)

        self._update_summary()
        self._selection_changed()

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.table.setDisabled(busy)
        self.curve_group_combo.setDisabled(busy or self._workflow is None)
        self._update_controls()

    def curve_group(self) -> str | None:
        value = self.curve_group_combo.currentData()
        return None if value is None else str(value)

    def _populate_curve_groups(self, workflow: WorkflowContext | None) -> None:
        current = self.curve_group()
        sector_names: set[str] = set()
        if workflow is not None:
            pipeline = StichingPipeline(workflow)
            sample_names = {
                run_key.sample_name
                for run_key in workflow.runs
                if run_key.entity == "sample"
                and run_key.mode == "scattering"
                and run_key.sample_name is not None
            }
            for sample_name in sample_names:
                sector_names.update(pipeline.available_sector_names(sample_name))

        blocker = QtCore.QSignalBlocker(self.curve_group_combo)
        self.curve_group_combo.clear()
        self.curve_group_combo.addItem("Isotropic", None)
        if sector_names:
            self.curve_group_combo.addItem("All common sectors", "*")
            for sector_name in sorted(sector_names):
                self.curve_group_combo.addItem(f"Sector: {sector_name}", sector_name)
        selected_index = self.curve_group_combo.findData(current)
        self.curve_group_combo.setCurrentIndex(max(0, selected_index))
        self.curve_group_combo.setEnabled(workflow is not None and not self._busy)
        del blocker

    def _restore_selected_samples(self, sample_names: set[str]) -> None:
        selection_model = self.table.selectionModel()
        selection_model.clearSelection()
        flags = (
            QtCore.QItemSelectionModel.SelectionFlag.Select
            | QtCore.QItemSelectionModel.SelectionFlag.Rows
        )
        for row in range(self.model.rowCount()):
            if self.model.sample_name_at(row) in sample_names:
                selection_model.select(self.model.index(row, 0), flags)

    def _update_summary(self) -> None:
        if self._workflow is None:
            self.summary_label.setText("No workflow loaded")
            return
        total = self.model.rowCount()
        stitched = self.model.status_count("stitched")
        ready = self.model.status_count("ready")
        waiting = self.model.status_count("waiting")
        waiting_text = f" · {waiting} waiting" if waiting else ""
        self.summary_label.setText(
            f"{total} sample(s) · {stitched} stitched · {ready} ready{waiting_text}"
        )

    @QtCore.Slot()
    def _curve_group_changed(self, *_args: object) -> None:
        selected_names = set(self.selected_sample_names())
        selection_model = self.table.selectionModel()
        blocker = QtCore.QSignalBlocker(selection_model)
        self.model.set_curve_group(self.curve_group())
        self._restore_selected_samples(selected_names)
        del blocker
        self._update_summary()
        self._selection_changed()

    def selected_sample_names(self) -> list[str]:
        return [
            sample_name
            for index in sorted(
                self.table.selectionModel().selectedRows(), key=lambda item: item.row()
            )
            if (sample_name := self.model.sample_name_at(index.row())) is not None
        ]

    @QtCore.Slot()
    def _selection_changed(self, *_args: object) -> None:
        selected = self.selected_sample_names()
        self.selection_label.setText(
            f"{len(selected)} sample(s) selected" if selected else "No sample selected"
        )
        self.stitch_selected_button.setText(
            f"Stitch selected ({len(selected)})" if selected else "Stitch selected"
        )
        self._populate_reference_configurations(selected)
        self._update_controls()

    def _populate_reference_configurations(self, selected_samples: list[str]) -> None:
        current_reference = self.reference_combo.currentData()
        configurations_by_sample: list[set[str]] = []
        if self._workflow is not None:
            for sample_name in selected_samples:
                configurations_by_sample.append(
                    {
                        run_key.config_id
                        for run_key in self._workflow.runs
                        if run_key.entity == "sample"
                        and run_key.mode == "scattering"
                        and run_key.sample_name == sample_name
                        and self._workflow.is_run_processed(run_key)
                    }
                )
        configurations = sorted(
            set.intersection(*configurations_by_sample)
            if configurations_by_sample
            else set()
        )
        blocker = QtCore.QSignalBlocker(self.reference_combo)
        self.reference_combo.clear()
        self.reference_combo.addItem("Automatic (lowest q)", None)
        for config_id in configurations:
            self.reference_combo.addItem(config_id, config_id)
        self.reference_combo.setCurrentIndex(
            max(0, self.reference_combo.findData(current_reference))
        )
        del blocker

    def _update_controls(self) -> None:
        selected = self.selected_sample_names()
        selected_ready = all(
            self.model.data(
                self.model.index(row, self.model.column_index("status")),
                QtCore.Qt.ItemDataRole.DisplayRole,
            )
            in {"ready", "stitched"}
            for row in (
                index.row() for index in self.table.selectionModel().selectedRows()
            )
        )
        can_stitch_selected = bool(selected) and selected_ready and not self._busy
        can_stitch_pending = self.model.status_count("ready") > 0 and not self._busy
        self.reference_combo.setEnabled(bool(selected) and not self._busy)
        self.normalization_factor_spin.setEnabled(
            self._workflow is not None and not self._busy
        )
        self.stitch_selected_button.setEnabled(can_stitch_selected)
        self.stitch_pending_button.setEnabled(can_stitch_pending)
        tooltip = "" if can_stitch_selected else "Select only samples ready for stitching."
        self.stitch_selected_button.setToolTip(tooltip)

    @QtCore.Slot()
    def _request_selected_stitching(self) -> None:
        samples = self.selected_sample_names()
        if samples:
            reference = self.reference_combo.currentData()
            self.stitch_selected_requested.emit(
                samples,
                reference,
                self.normalization_factor_spin.value(),
                self.curve_group(),
            )

    @QtCore.Slot()
    def _request_pending_stitching(self) -> None:
        self.stitch_pending_requested.emit(
            self.normalization_factor_spin.value(),
            self.curve_group(),
        )
