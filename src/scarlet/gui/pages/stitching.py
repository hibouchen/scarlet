from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from scarlet.workflow import WorkflowContext

from ..models import StitchingSamplesModel


class StitchingPage(QtWidgets.QWidget):
    """Review stitched outputs and merge selected ready sample groups."""

    stitch_selected_requested = QtCore.Signal(object, object, float)
    stitch_pending_requested = QtCore.Signal(float)

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

        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self.selection_label)
        actions.addStretch(1)
        actions.addWidget(QtWidgets.QLabel("Reference"))
        actions.addWidget(self.reference_combo)
        actions.addWidget(QtWidgets.QLabel("Normalization factor"))
        actions.addWidget(self.normalization_factor_spin)
        actions.addWidget(self.stitch_selected_button)
        actions.addWidget(self.stitch_pending_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.table, 1)
        layout.addLayout(actions)

        self.stitch_selected_button.clicked.connect(self._request_selected_stitching)
        self.stitch_pending_button.clicked.connect(self._request_pending_stitching)
        self._update_controls()

    def set_workflow(self, workflow: WorkflowContext | None) -> None:
        selected_names = set(self.selected_sample_names())
        self._workflow = workflow
        selection_model = self.table.selectionModel()
        blocker = QtCore.QSignalBlocker(selection_model)
        self.model.set_workflow(workflow)
        for row in range(self.model.rowCount()):
            if self.model.sample_name_at(row) in selected_names:
                self.table.selectRow(row)
        del blocker

        if workflow is None:
            self.summary_label.setText("No workflow loaded")
        else:
            total = self.model.rowCount()
            stitched = self.model.status_count("stitched")
            ready = self.model.status_count("ready")
            waiting = self.model.status_count("waiting")
            waiting_text = f" · {waiting} waiting" if waiting else ""
            self.summary_label.setText(
                f"{total} sample(s) · {stitched} stitched · {ready} ready{waiting_text}"
            )
        self._selection_changed()

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.table.setDisabled(busy)
        self._update_controls()

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
            )

    @QtCore.Slot()
    def _request_pending_stitching(self) -> None:
        self.stitch_pending_requested.emit(self.normalization_factor_spin.value())
