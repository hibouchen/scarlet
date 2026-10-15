from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from scarlet.workflow import RunKey, WorkflowContext
from scarlet.workflow.pipeline import (
    DEFAULT_AZIMUTHAL_N_BINS,
    DEFAULT_AZIMUTHAL_Q_SCALE,
)

from ..models import ProcessingRunsModel, ReferenceTransmissionsModel
from ..readiness import assess_workflow_readiness


class ProcessingPage(QtWidgets.QWidget):
    """Select and start reduction of sample-scattering acquisitions."""

    selection_changed = QtCore.Signal(object)
    process_selected_requested = QtCore.Signal(object)
    process_pending_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = ProcessingRunsModel(parent=self)
        self.sort_model = QtCore.QSortFilterProxyModel(self)
        self.sort_model.setSourceModel(self.model)
        self.sort_model.setDynamicSortFilter(True)

        title = QtWidgets.QLabel("Processing")
        title.setObjectName("pageTitle")
        intro = QtWidgets.QLabel(
            "Select the sample scattering acquisitions to reduce. "
            "Processing runs in the background and is available once the "
            "workflow inputs are ready."
        )
        intro.setWordWrap(True)

        self.summary_label = QtWidgets.QLabel("No workflow loaded")
        self.readiness_label = QtWidgets.QLabel()
        self.readiness_label.setWordWrap(True)

        self.table = QtWidgets.QTableView()
        self.table.setModel(self.sort_model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            self.model.column_index("file_path"),
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)

        self.reference_transmissions_model = ReferenceTransmissionsModel(parent=self)
        self.reference_transmissions_table = QtWidgets.QTableView()
        self.reference_transmissions_table.setModel(self.reference_transmissions_model)
        self.reference_transmissions_table.setAlternatingRowColors(True)
        self.reference_transmissions_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.reference_transmissions_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.reference_transmissions_table.verticalHeader().setVisible(False)
        self.reference_transmissions_table.setMaximumHeight(130)
        reference_header = self.reference_transmissions_table.horizontalHeader()
        reference_header.setStretchLastSection(True)
        reference_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        reference_title = QtWidgets.QLabel("Reference transmissions")
        reference_title.setObjectName("sectionTitle")

        self.integration_group = QtWidgets.QGroupBox("Azimuthal integration")
        integration_layout = QtWidgets.QFormLayout(self.integration_group)
        self.n_bins_widget = QtWidgets.QWidget()
        self.n_bins_layout = QtWidgets.QFormLayout(self.n_bins_widget)
        self.n_bins_layout.setContentsMargins(0, 0, 0, 0)
        self.n_bins_spinboxes: dict[int, QtWidgets.QSpinBox] = {}
        self._set_detector_numbers((0,))
        integration_layout.addRow(self.n_bins_widget)

        self.q_scale_combo = QtWidgets.QComboBox()
        self.q_scale_combo.setObjectName("azimuthalQScale")
        self.q_scale_combo.addItem("Linear", "linear")
        self.q_scale_combo.addItem("Logarithmic", "log")
        default_scale_index = self.q_scale_combo.findData(DEFAULT_AZIMUTHAL_Q_SCALE)
        self.q_scale_combo.setCurrentIndex(default_scale_index)
        self.q_scale_combo.setToolTip("Spacing of the radial q-bin boundaries.")
        integration_layout.addRow("Q scale", self.q_scale_combo)

        self.selection_label = QtWidgets.QLabel("No acquisition selected")
        self.process_selected_button = QtWidgets.QPushButton("Process selected")
        self.process_pending_button = QtWidgets.QPushButton("Process pending")
        self._workflow_ready = False
        self._has_pending = False
        self._busy = False

        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self.selection_label)
        actions.addStretch(1)
        actions.addWidget(self.process_selected_button)
        actions.addWidget(self.process_pending_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.readiness_label)
        layout.addWidget(self.table, 1)
        layout.addWidget(reference_title)
        layout.addWidget(self.reference_transmissions_table)
        layout.addWidget(self.integration_group)
        layout.addLayout(actions)

        self.process_selected_button.clicked.connect(self._request_selected_processing)
        self.process_pending_button.clicked.connect(self.process_pending_requested)

    def _set_detector_numbers(self, detector_numbers: tuple[int, ...]) -> None:
        previous_values = {
            detector_number: spinbox.value()
            for detector_number, spinbox in self.n_bins_spinboxes.items()
        }
        while self.n_bins_layout.rowCount() > 0:
            self.n_bins_layout.removeRow(0)
        self.n_bins_spinboxes.clear()

        for detector_number in detector_numbers:
            spinbox = QtWidgets.QSpinBox()
            spinbox.setObjectName(f"nBinsDetector{detector_number}")
            spinbox.setRange(1, 10000)
            spinbox.setValue(
                previous_values.get(
                    detector_number,
                    DEFAULT_AZIMUTHAL_N_BINS.get(detector_number, 30),
                )
            )
            spinbox.setToolTip(
                f"Number of radial q bins used for detector{detector_number}."
            )
            self.n_bins_spinboxes[detector_number] = spinbox
            self.n_bins_layout.addRow(f"Detector {detector_number} bins", spinbox)

    @staticmethod
    def _workflow_detector_numbers(workflow: WorkflowContext | None) -> tuple[int, ...]:
        if workflow is None:
            return (0,)
        detector_counts = []
        for configuration in workflow.configurations.values():
            distance = configuration.sample_detector_distance
            detector_counts.append(len(distance) if isinstance(distance, list) else 1)
        detector_count = max(detector_counts, default=1)
        return tuple(range(detector_count))

    def set_workflow(self, workflow: WorkflowContext | None) -> None:
        self._set_detector_numbers(self._workflow_detector_numbers(workflow))
        self.model.set_workflow(workflow)
        self.reference_transmissions_model.set_workflow(workflow)
        if workflow is None:
            self._workflow_ready = False
            self._has_pending = False
            self.summary_label.setText("No workflow loaded")
            self.readiness_label.clear()
            self._selection_changed()
            return

        total = self.model.rowCount()
        pending = self.model.status_count("pending")
        processed = self.model.status_count("processed")
        unavailable = self.model.status_count("unavailable")
        unavailable_text = f" · {unavailable} unavailable" if unavailable else ""
        self.summary_label.setText(
            f"{total} sample acquisition(s) · {pending} pending · "
            f"{processed} processed{unavailable_text}"
        )

        readiness = assess_workflow_readiness(workflow)
        self._workflow_ready = readiness.ready
        self._has_pending = pending > 0
        if not readiness.ready:
            self.readiness_label.setText(
                f"Processing blocked · {len(readiness.errors)} issue(s). "
                "Return to Acquisitions to review the inputs."
            )
            self.readiness_label.setToolTip("\n".join(readiness.errors))
            self.readiness_label.setStyleSheet("color: #b71c1c;")
        elif readiness.warnings:
            self.readiness_label.setText(
                f"Ready with {len(readiness.warnings)} warning(s)."
            )
            self.readiness_label.setToolTip("\n".join(readiness.warnings))
            self.readiness_label.setStyleSheet("color: #8a5a00;")
        else:
            self.readiness_label.setText("Ready for processing.")
            self.readiness_label.setToolTip("")
            self.readiness_label.setStyleSheet("color: #2e7d32;")
        self._selection_changed()

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.table.setDisabled(busy)
        self.reference_transmissions_table.setDisabled(busy)
        self.integration_group.setDisabled(busy)
        self._update_action_controls()

    def reduction_options(self) -> tuple[dict[int, int], str]:
        """Return the azimuthal-integration options selected in the page."""
        n_bins = {
            detector_number: spinbox.value()
            for detector_number, spinbox in self.n_bins_spinboxes.items()
        }
        q_scale = str(self.q_scale_combo.currentData())
        return n_bins, q_scale

    def selected_run_keys(self) -> list[RunKey]:
        selected: list[RunKey] = []
        for proxy_index in sorted(
            self.table.selectionModel().selectedRows(),
            key=lambda index: index.row(),
        ):
            source_index = self.sort_model.mapToSource(proxy_index)
            run_key = self.model.run_key_at(source_index.row())
            if run_key is not None:
                selected.append(run_key)
        return selected

    @QtCore.Slot()
    def _selection_changed(self, *_args: object) -> None:
        selected = self.selected_run_keys()
        count = len(selected)
        self.selection_label.setText(
            f"{count} acquisition(s) selected" if count else "No acquisition selected"
        )
        self.process_selected_button.setText(
            f"Process selected ({count})" if count else "Process selected"
        )
        self._update_action_controls()
        self.selection_changed.emit(selected)

    def _update_action_controls(self) -> None:
        selected = self.selected_run_keys()
        can_process = self._workflow_ready and not self._busy
        self.process_selected_button.setEnabled(can_process and bool(selected))
        self.process_pending_button.setEnabled(can_process and self._has_pending)
        if not self._workflow_ready:
            tooltip = "Complete the blocking items above before processing."
        elif self._busy:
            tooltip = "Processing is already running."
        else:
            tooltip = ""
        self.process_selected_button.setToolTip(tooltip)
        self.process_pending_button.setToolTip(tooltip)

    @QtCore.Slot()
    def _request_selected_processing(self) -> None:
        selected = self.selected_run_keys()
        if selected:
            self.process_selected_requested.emit(selected)
