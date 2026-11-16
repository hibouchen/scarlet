from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets

from scarlet.reduction import Sector
from scarlet.workflow import RunKey, WorkflowContext
from scarlet.workflow.pipeline import (
    DEFAULT_AZIMUTHAL_N_BINS,
    DEFAULT_AZIMUTHAL_Q_SCALE,
)

from ..models import ProcessingRunsModel, ReferenceTransmissionsModel
from ..readiness import assess_workflow_readiness
from ..widgets import SectorPreviewWidget


@dataclass(frozen=True)
class ReductionOptions:
    """Reduction parameters selected on the processing page."""

    n_bins: dict[int, int]
    q_scale: str
    sectors: tuple[Sector, ...] = ()


class ProcessingPage(QtWidgets.QWidget):
    """Select and start reduction of sample-scattering acquisitions."""

    selection_changed = QtCore.Signal(object)
    preview_requested = QtCore.Signal(object)
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

        self.anisotropic_check = QtWidgets.QCheckBox("Enable sector integration")
        self.anisotropic_check.setObjectName("anisotropicIntegration")
        self.anisotropic_check.setToolTip(
            "Compute directional I(q) curves in addition to the isotropic average."
        )
        integration_layout.addRow(self.anisotropic_check)

        self.sector_controls_widget = QtWidgets.QWidget()
        sector_layout = QtWidgets.QFormLayout(self.sector_controls_widget)
        sector_layout.setContentsMargins(18, 0, 0, 0)

        self.sector_preset_combo = QtWidgets.QComboBox()
        self.sector_preset_combo.setObjectName("sectorPreset")
        self.sector_preset_combo.addItem(
            "Parallel / perpendicular",
            "parallel_perpendicular",
        )
        self.sector_preset_combo.addItem("Single direction", "single")
        self.sector_preset_combo.setToolTip(
            "Create either two orthogonal sector curves or one directional curve."
        )
        sector_layout.addRow("Preset", self.sector_preset_combo)

        self.sector_direction_spin = QtWidgets.QDoubleSpinBox()
        self.sector_direction_spin.setObjectName("sectorDirection")
        self.sector_direction_spin.setRange(-180.0, 180.0)
        self.sector_direction_spin.setDecimals(1)
        self.sector_direction_spin.setSingleStep(1.0)
        self.sector_direction_spin.setSuffix("°")
        self.sector_direction_spin.setWrapping(True)
        self.sector_direction_spin.setToolTip(
            "Main direction in detector coordinates: 0° is +X and 90° is +Y."
        )
        sector_layout.addRow("Main direction", self.sector_direction_spin)

        self.sector_width_spin = QtWidgets.QDoubleSpinBox()
        self.sector_width_spin.setObjectName("sectorWidth")
        self.sector_width_spin.setRange(1.0, 180.0)
        self.sector_width_spin.setDecimals(1)
        self.sector_width_spin.setSingleStep(1.0)
        self.sector_width_spin.setValue(20.0)
        self.sector_width_spin.setSuffix("°")
        self.sector_width_spin.setToolTip("Full angular width of each sector.")
        sector_layout.addRow("Sector width", self.sector_width_spin)

        self.sector_opposite_check = QtWidgets.QCheckBox(
            "Combine the opposite direction"
        )
        self.sector_opposite_check.setObjectName("sectorIncludeOpposite")
        self.sector_opposite_check.setChecked(True)
        self.sector_opposite_check.setToolTip(
            "Average the sector centred 180° away into the same I(q) curve."
        )
        sector_layout.addRow(self.sector_opposite_check)
        integration_layout.addRow(self.sector_controls_widget)

        self.sector_summary_label = QtWidgets.QLabel()
        self.sector_summary_label.setObjectName("sectorSummary")
        self.sector_summary_label.setWordWrap(True)
        self.sector_summary_label.setStyleSheet("color: #5f6368;")
        integration_layout.addRow("Output", self.sector_summary_label)

        self.sector_preview = SectorPreviewWidget()
        self.sector_preview.setMinimumHeight(260)
        preview_title = QtWidgets.QLabel("Detector and sector preview")
        preview_title.setObjectName("sectionTitle")
        self.sector_legend_label = QtWidgets.QLabel(
            "Red: parallel or selected direction · Blue: perpendicular"
        )
        self.sector_legend_label.setWordWrap(True)
        self.sector_legend_label.setStyleSheet("color: #5f6368;")
        preview_panel = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(self.sector_preview, 1)
        preview_layout.addWidget(self.sector_legend_label)

        self.content_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.content_splitter.addWidget(self.table)
        self.content_splitter.addWidget(preview_panel)
        self.content_splitter.setStretchFactor(0, 3)
        self.content_splitter.setStretchFactor(1, 2)
        self.content_splitter.setSizes([760, 500])

        self.anisotropic_check.toggled.connect(self._update_sector_controls)
        self.sector_preset_combo.currentIndexChanged.connect(
            self._update_sector_summary
        )
        self.sector_direction_spin.valueChanged.connect(
            self._update_sector_summary
        )
        self.sector_width_spin.valueChanged.connect(self._update_sector_summary)
        self.sector_opposite_check.toggled.connect(self._update_sector_summary)
        self._update_sector_controls(False)

        self.selection_label = QtWidgets.QLabel("No acquisition selected")
        self.process_selected_button = QtWidgets.QPushButton("Process selected")
        self.process_pending_button = QtWidgets.QPushButton("Process pending")
        self._workflow_ready = False
        self._has_pending = False
        self._busy = False
        self._preview_run_key: RunKey | None = None
        self._requested_preview_run_key: RunKey | None = None

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
        layout.addWidget(self.content_splitter, 1)
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
        selected_run_keys = self.selected_run_keys()
        selection_model = self.table.selectionModel()
        blocker = QtCore.QSignalBlocker(selection_model)
        self._set_detector_numbers(self._workflow_detector_numbers(workflow))
        self.model.set_workflow(workflow)
        self._restore_selected_run_keys(selected_run_keys)
        del blocker
        self.reference_transmissions_model.set_workflow(workflow)
        if workflow is None:
            self._workflow_ready = False
            self._has_pending = False
            self.summary_label.setText("No workflow loaded")
            self.readiness_label.clear()
            self.clear_preview()
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
        self._update_preview_beam_centers(workflow)
        self._selection_changed()

    def set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = bool(busy)
        preview_loading = busy and message == "Loading acquisition detector image…"
        self.table.setEnabled(not busy or preview_loading)
        self.reference_transmissions_table.setDisabled(busy)
        self.integration_group.setDisabled(busy and not preview_loading)
        self._update_action_controls()

    def _selected_sectors(self) -> tuple[Sector, ...]:
        if not self.anisotropic_check.isChecked():
            return ()

        direction = self.sector_direction_spin.value()
        width = self.sector_width_spin.value()
        include_opposite = self.sector_opposite_check.isChecked()
        parallel_preset = (
            self.sector_preset_combo.currentData() == "parallel_perpendicular"
        )
        primary = Sector(
            "parallel" if parallel_preset else "direction",
            center_deg=direction,
            width_deg=width,
            include_opposite=include_opposite,
        )
        if self.sector_preset_combo.currentData() == "single":
            return (primary,)
        return (
            primary,
            Sector(
                "perpendicular",
                center_deg=direction + 90.0,
                width_deg=width,
                include_opposite=include_opposite,
            ),
        )

    @QtCore.Slot()
    def _update_sector_summary(self, *_args: object) -> None:
        sectors = self._selected_sectors()
        self.sector_preview.set_sectors(sectors)
        if not sectors:
            self.sector_summary_label.setText("Isotropic I(q) only.")
            return
        descriptions = [
            f"{sector.name}: {sector.center_deg:g}° ± {sector.width_deg / 2:g}°"
            + (" and opposite" if sector.include_opposite else "")
            for sector in sectors
        ]
        self.sector_summary_label.setText("; ".join(descriptions) + ".")

    @QtCore.Slot(bool)
    def _update_sector_controls(self, enabled: bool) -> None:
        self.sector_controls_widget.setEnabled(bool(enabled))
        self._update_sector_summary()

    def reduction_options(self) -> ReductionOptions:
        """Return the integration options selected in the page."""
        n_bins = {
            detector_number: spinbox.value()
            for detector_number, spinbox in self.n_bins_spinboxes.items()
        }
        q_scale = str(self.q_scale_combo.currentData())
        return ReductionOptions(
            n_bins=n_bins,
            q_scale=q_scale,
            sectors=self._selected_sectors(),
        )

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

    def _restore_selected_run_keys(self, run_keys: list[RunKey]) -> None:
        selection_model = self.table.selectionModel()
        selection_model.clearSelection()
        wanted = set(run_keys)
        if not wanted:
            return
        flags = (
            QtCore.QItemSelectionModel.SelectionFlag.Select
            | QtCore.QItemSelectionModel.SelectionFlag.Rows
        )
        for source_row in range(self.model.rowCount()):
            if self.model.run_key_at(source_row) not in wanted:
                continue
            proxy_index = self.sort_model.mapFromSource(self.model.index(source_row, 0))
            if proxy_index.isValid():
                selection_model.select(proxy_index, flags)

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
        if count != 1:
            message = (
                "Select one acquisition to display its detector image."
                if count == 0
                else "Select a single acquisition to preview its integration sectors."
            )
            self.clear_preview(message)
            return
        run_key = selected[0]
        if run_key in {self._preview_run_key, self._requested_preview_run_key}:
            return
        self._preview_run_key = None
        self._requested_preview_run_key = run_key
        self.sector_preview.clear("Loading selected acquisition…")
        self.preview_requested.emit(run_key)

    @QtCore.Slot(object)
    def set_preview(self, payload: object) -> None:
        source = getattr(payload, "source", None)
        run_key = getattr(payload, "run_key", None)
        if (
            source is None
            or run_key is None
            or run_key != self._requested_preview_run_key
        ):
            return
        self._preview_run_key = run_key
        self.sector_preview.set_source(
            source,
            display_count_time=getattr(payload, "display_count_time", None),
            beam_centers=getattr(payload, "beam_centers", {}),
        )
        self.sector_preview.set_sectors(self._selected_sectors())

    def clear_preview(
        self,
        message: str = "Select one acquisition to display its detector image.",
    ) -> None:
        self._preview_run_key = None
        self._requested_preview_run_key = None
        self.sector_preview.clear(message)

    def _update_preview_beam_centers(self, workflow: WorkflowContext) -> None:
        run_key = self._preview_run_key
        if run_key is None:
            return
        self.sector_preview.set_beam_centers(
            workflow.beam_centers.get(run_key.config_id, {})
        )

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
