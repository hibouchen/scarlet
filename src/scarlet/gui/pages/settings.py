from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from scarlet.workflow import WorkflowContext

from ..models import ConfigurationSettingsModel, ConfigurationSettingsRow
from ..settings_data import SettingsImagePayload
from ..widgets import SettingsImageView


class SettingsPage(QtWidgets.QWidget):
    """Entry point for configuration-level preparation before reduction."""

    image_requested = QtCore.Signal(str, str, int)
    auto_detect_requested = QtCore.Signal(str, str, int)
    beam_center_change_requested = QtCore.Signal(str, int, object)
    roi_change_requested = QtCore.Signal(str, object)
    flatfield_build_requested = QtCore.Signal(str)
    flatfield_file_requested = QtCore.Signal(str)
    flatfield_source_change_requested = QtCore.Signal(str, str)
    transmission_source_change_requested = QtCore.Signal(str, str)
    mask_file_requested = QtCore.Signal(str)
    mask_auto_find_requested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        title = QtWidgets.QLabel("Settings")
        title.setObjectName("pageTitle")
        intro = QtWidgets.QLabel(
            "Review beam centers and transmission ROIs, assign detector masks, "
            "then create or assign flatfields before processing."
        )
        intro.setWordWrap(True)
        self.summary_label = QtWidgets.QLabel("No workflow loaded")
        self.summary_label.setWordWrap(True)
        self._workflow: WorkflowContext | None = None
        self._center_available = False
        self._roi_available = False

        self.model = ConfigurationSettingsModel(parent=self)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(48)
        header.setDefaultSectionSize(130)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)

        details = QtWidgets.QGroupBox("Selected configuration")
        details_form = QtWidgets.QFormLayout(details)
        self.configuration_label = QtWidgets.QLabel("—")
        self.characteristics_label = self._detail_label()
        self.centers_label = self._detail_label()
        self.roi_label = self._detail_label()
        self.masks_label = self._detail_label()
        self.flatfield_label = self._detail_label()
        details_form.addRow("Configuration", self.configuration_label)
        details_form.addRow("Characteristics", self.characteristics_label)
        details_form.addRow("Beam centers", self.centers_label)
        details_form.addRow("Transmission ROI", self.roi_label)
        details_form.addRow("Masks", self.masks_label)
        details_form.addRow("Flatfield", self.flatfield_label)

        image_controls = QtWidgets.QHBoxLayout()
        self.purpose_combo = QtWidgets.QComboBox()
        self.purpose_combo.addItem("Beam center", "center")
        self.purpose_combo.addItem("Transmission ROI", "roi")
        self.detector_combo = QtWidgets.QComboBox()
        self.load_image_button = QtWidgets.QPushButton("Load empty beam")
        image_controls.addWidget(QtWidgets.QLabel("Edit"))
        image_controls.addWidget(self.purpose_combo)
        image_controls.addWidget(QtWidgets.QLabel("Detector"))
        image_controls.addWidget(self.detector_combo)
        image_controls.addWidget(self.load_image_button)

        self.image_source_label = QtWidgets.QLabel("Select a configuration and load an image.")
        self.image_source_label.setWordWrap(True)
        self.image_view = SettingsImageView()

        self.center_x_spin = self._coordinate_spin()
        self.center_y_spin = self._coordinate_spin()
        center_form = QtWidgets.QFormLayout()
        center_form.addRow("Center X", self.center_x_spin)
        center_form.addRow("Center Y", self.center_y_spin)
        self.detect_center_button = QtWidgets.QPushButton("Detect center automatically")
        self.save_center_button = QtWidgets.QPushButton("Save beam center")
        center_form.addRow(self.detect_center_button)
        center_form.addRow(self.save_center_button)
        self.center_editor = QtWidgets.QWidget()
        self.center_editor.setLayout(center_form)

        self.roi_x0_spin = self._pixel_spin()
        self.roi_x1_spin = self._pixel_spin()
        self.roi_y0_spin = self._pixel_spin()
        self.roi_y1_spin = self._pixel_spin()
        roi_form = QtWidgets.QFormLayout()
        roi_form.addRow("ROI X start", self.roi_x0_spin)
        roi_form.addRow("ROI X end", self.roi_x1_spin)
        roi_form.addRow("ROI Y start", self.roi_y0_spin)
        roi_form.addRow("ROI Y end", self.roi_y1_spin)
        self.detect_roi_button = QtWidgets.QPushButton("Detect ROI automatically")
        self.save_roi_button = QtWidgets.QPushButton("Save transmission ROI")
        roi_form.addRow(self.detect_roi_button)
        roi_form.addRow(self.save_roi_button)
        self.roi_editor = QtWidgets.QWidget()
        self.roi_editor.setLayout(roi_form)

        self.numeric_editors = QtWidgets.QStackedWidget()
        self.numeric_editors.addWidget(self.center_editor)
        self.numeric_editors.addWidget(self.roi_editor)

        geometry_tab = QtWidgets.QWidget()
        geometry_layout = QtWidgets.QVBoxLayout(geometry_tab)
        geometry_layout.setContentsMargins(8, 8, 8, 8)
        geometry_layout.addLayout(image_controls)
        geometry_layout.addWidget(self.image_source_label)
        geometry_layout.addWidget(self.image_view, 1)
        geometry_layout.addWidget(self.numeric_editors)

        mask_tab = QtWidgets.QWidget()
        mask_layout = QtWidgets.QVBoxLayout(mask_tab)
        mask_layout.setContentsMargins(12, 12, 12, 12)
        mask_intro = QtWidgets.QLabel(
            "Detector masks are edited from Acquisitions: select a file, draw the mask "
            "in the detector viewer, then save and attach it to this configuration."
        )
        mask_intro.setWordWrap(True)
        self.mask_tab_label = self._detail_label()
        self.choose_mask_button = QtWidgets.QPushButton("Choose mask file…")
        self.find_mask_button = QtWidgets.QPushButton("Find compatible mask")
        self.choose_mask_button.setEnabled(False)
        self.find_mask_button.setEnabled(False)
        mask_actions = QtWidgets.QHBoxLayout()
        mask_actions.addWidget(self.choose_mask_button)
        mask_actions.addWidget(self.find_mask_button)
        mask_layout.addWidget(mask_intro)
        mask_layout.addWidget(self.mask_tab_label)
        mask_layout.addLayout(mask_actions)
        mask_layout.addStretch(1)

        flatfield_tab = QtWidgets.QWidget()
        flatfield_layout = QtWidgets.QVBoxLayout(flatfield_tab)
        flatfield_layout.setContentsMargins(12, 12, 12, 12)
        self.flatfield_tab_label = self._detail_label()
        self.build_flatfield_button = QtWidgets.QPushButton("Create flatfield")
        self.choose_flatfield_button = QtWidgets.QPushButton("Choose flatfield file…")
        self.build_flatfield_button.setEnabled(False)
        self.choose_flatfield_button.setEnabled(False)
        flatfield_actions = QtWidgets.QHBoxLayout()
        flatfield_actions.addWidget(self.build_flatfield_button)
        flatfield_actions.addWidget(self.choose_flatfield_button)
        self.flatfield_source_combo = QtWidgets.QComboBox()
        self.use_flatfield_source_button = QtWidgets.QPushButton("Use configuration")
        self.use_flatfield_source_button.setEnabled(False)
        flatfield_source_form = QtWidgets.QFormLayout()
        flatfield_source_form.addRow("Reuse source", self.flatfield_source_combo)
        flatfield_source_form.addRow(self.use_flatfield_source_button)
        flatfield_layout.addWidget(self.flatfield_tab_label)
        flatfield_layout.addLayout(flatfield_actions)
        flatfield_layout.addLayout(flatfield_source_form)
        flatfield_layout.addStretch(1)

        transmission_tab = QtWidgets.QWidget()
        transmission_layout = QtWidgets.QVBoxLayout(transmission_tab)
        transmission_layout.setContentsMargins(12, 12, 12, 12)
        transmission_intro = QtWidgets.QLabel(
            "Use the transmissions measured for another configuration. This also "
            "uses that configuration's transmission ROI and invalidates any "
            "flatfield built for the selected configuration."
        )
        transmission_intro.setWordWrap(True)
        self.transmission_source_label = self._detail_label()
        self.transmission_source_combo = QtWidgets.QComboBox()
        self.use_transmission_source_button = QtWidgets.QPushButton("Use configuration")
        self.use_transmission_source_button.setEnabled(False)
        transmission_source_form = QtWidgets.QFormLayout()
        transmission_source_form.addRow("Transmission source", self.transmission_source_combo)
        transmission_source_form.addRow(self.use_transmission_source_button)
        transmission_layout.addWidget(transmission_intro)
        transmission_layout.addWidget(self.transmission_source_label)
        transmission_layout.addLayout(transmission_source_form)
        transmission_layout.addStretch(1)

        self.settings_tabs = QtWidgets.QTabWidget()
        self.settings_tabs.addTab(geometry_tab, "Beam center and ROI")
        self.settings_tabs.addTab(transmission_tab, "Transmission")
        self.settings_tabs.addTab(mask_tab, "Mask")
        self.settings_tabs.addTab(flatfield_tab, "Flatfield")

        editor = QtWidgets.QWidget()
        editor_layout = QtWidgets.QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.addWidget(details)
        editor_layout.addWidget(self.settings_tabs, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(editor)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addWidget(self.summary_label)
        layout.addWidget(splitter, 1)

        self.purpose_combo.currentIndexChanged.connect(self._purpose_changed)
        self.detector_combo.currentIndexChanged.connect(self._metadata_selection_changed)
        self.load_image_button.clicked.connect(self._request_image)
        self.detect_center_button.clicked.connect(self._request_auto_detection)
        self.detect_roi_button.clicked.connect(self._request_auto_detection)
        self.save_center_button.clicked.connect(self._save_center)
        self.save_roi_button.clicked.connect(self._save_roi)
        self.build_flatfield_button.clicked.connect(self._request_flatfield_build)
        self.choose_flatfield_button.clicked.connect(self._request_flatfield_file)
        self.use_flatfield_source_button.clicked.connect(self._set_flatfield_source)
        self.use_transmission_source_button.clicked.connect(self._set_transmission_source)
        self.choose_mask_button.clicked.connect(self._request_mask_file)
        self.find_mask_button.clicked.connect(self._request_compatible_mask)
        self.center_x_spin.valueChanged.connect(self._center_fields_changed)
        self.center_y_spin.valueChanged.connect(self._center_fields_changed)
        for spin in (
            self.roi_x0_spin,
            self.roi_x1_spin,
            self.roi_y0_spin,
            self.roi_y1_spin,
        ):
            spin.valueChanged.connect(self._roi_fields_changed)
        self.image_view.center_edited.connect(self._center_edited_on_image)
        self.image_view.roi_edited.connect(self._roi_edited_on_image)
        self._purpose_changed()

    def set_workflow(self, workflow: WorkflowContext | None) -> None:
        selected_config = self._selected_config_id()
        self._workflow = workflow
        self.model.set_workflow(workflow)
        if workflow is None:
            self.summary_label.setText("No workflow loaded")
            self._show_row(None)
            return
        count = self.model.rowCount()
        ready = self.model.status_count("ready")
        self.summary_label.setText(
            f"{count} configuration(s) · {ready} ready · {count - ready} incomplete"
        )
        if count:
            selected_row = next(
                (
                    index
                    for index in range(count)
                    if self.model.row_at(index).config_id == selected_config
                ),
                0,
            )
            self.table.selectRow(selected_row)
        else:
            self._show_row(None)

    def set_busy(self, busy: bool) -> None:
        self.table.setDisabled(busy)
        for widget in (
            self.purpose_combo,
            self.detector_combo,
            self.load_image_button,
            self.detect_center_button,
            self.detect_roi_button,
            self.save_center_button,
            self.save_roi_button,
            self.build_flatfield_button,
            self.choose_flatfield_button,
            self.flatfield_source_combo,
            self.use_flatfield_source_button,
            self.transmission_source_combo,
            self.use_transmission_source_button,
            self.choose_mask_button,
            self.find_mask_button,
        ):
            widget.setDisabled(busy)
        if not busy:
            self.detector_combo.setEnabled(self._purpose() == "center")
            self._update_flatfield_button()
            self._update_flatfield_source_controls()
            self._update_transmission_source_controls()
            self._update_mask_controls()

    @QtCore.Slot(object)
    def set_image(self, payload: object) -> None:
        if not isinstance(payload, SettingsImagePayload):
            return
        if (
            payload.config_id != self._selected_config_id()
            or payload.purpose != self._purpose()
            or payload.detector_number
            != (self._detector_number() if self._purpose() == "center" else 0)
        ):
            return
        self.image_view.set_image(payload.image)
        self.image_source_label.setText(
            f"{payload.source_path.name} · {payload.image.shape[1]} × {payload.image.shape[0]} pixels"
        )
        self._set_coordinate_bounds(payload.image.shape)
        if payload.center is not None:
            self._set_center_fields(payload.center)
        if payload.roi is not None:
            self._set_roi_fields(payload.roi)
        self._update_overlays_from_fields()

    @staticmethod
    def _detail_label() -> QtWidgets.QLabel:
        label = QtWidgets.QLabel("—")
        label.setWordWrap(True)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    @QtCore.Slot()
    def _selection_changed(self, *_args: object) -> None:
        rows = self.table.selectionModel().selectedRows()
        self._show_row(self.model.row_at(rows[0].row()) if rows else None)
        self._populate_detectors()
        self._center_available = False
        self._roi_available = False
        self._metadata_selection_changed()
        self.image_view.set_image(None)
        self.image_source_label.setText("Load an empty-beam image for this configuration.")

    def _show_row(self, row: ConfigurationSettingsRow | None) -> None:
        if row is None:
            for label in (
                self.configuration_label,
                self.characteristics_label,
                self.centers_label,
                self.roi_label,
                self.masks_label,
                self.flatfield_label,
                self.mask_tab_label,
                self.flatfield_tab_label,
                self.transmission_source_label,
            ):
                label.setText("—")
            self.build_flatfield_button.setEnabled(False)
            self.choose_flatfield_button.setEnabled(False)
            self.flatfield_source_combo.setEnabled(False)
            self.use_flatfield_source_button.setEnabled(False)
            self.transmission_source_combo.setEnabled(False)
            self.use_transmission_source_button.setEnabled(False)
            self.choose_mask_button.setEnabled(False)
            self.find_mask_button.setEnabled(False)
            return
        self.configuration_label.setText(row.config_id)
        self.characteristics_label.setText(self._configuration_characteristics(row.config_id))
        self.centers_label.setText(row.centers_detail)
        self.roi_label.setText(row.roi_detail)
        self.masks_label.setText(row.masks_detail)
        self.flatfield_label.setText(row.flatfield_detail)
        self.mask_tab_label.setText(row.masks_detail)
        self.flatfield_tab_label.setText(row.flatfield_detail)
        self._update_flatfield_button(row)
        self._populate_flatfield_sources(row.config_id)
        self._populate_transmission_sources(row.config_id)
        self._update_mask_controls()

    def _configuration_characteristics(self, config_id: str) -> str:
        workflow = self._workflow
        configuration = None if workflow is None else workflow.configurations.get(config_id)
        if configuration is None:
            return "No configuration metadata is available."

        parts: list[str] = []
        wavelength = float(configuration.wavelength)
        if math.isfinite(wavelength):
            parts.append(f"Wavelength: {wavelength:g} Å")

        distances = configuration.sample_detector_distance
        if isinstance(distances, list):
            detector_distances = ", ".join(
                f"detector{index}: {float(distance):g} m"
                for index, distance in enumerate(distances)
                if math.isfinite(float(distance))
            )
            if detector_distances:
                parts.append(f"Sample-detector: {detector_distances}")
        elif math.isfinite(float(distances)):
            parts.append(f"Sample-detector: {float(distances):g} m")

        collimation = configuration.collimation
        if collimation is not None:
            parts.append(
                "Collimation: "
                f"{collimation.collimation_distance:g} m; "
                f"last aperture to sample: {collimation.last_aperture_to_sample_distance:g} m"
            )
            parts.append(
                "Apertures: "
                f"{self._format_aperture(collimation.aperture1)}; "
                f"{self._format_aperture(collimation.aperture2)}"
            )
        if configuration.notes:
            parts.append(f"Notes: {configuration.notes}")
        return "\n".join(parts) if parts else "No configuration metadata is available."

    @staticmethod
    def _format_aperture(aperture: object) -> str:
        aperture_type = getattr(aperture, "type", "unknown")
        if aperture_type == "slit":
            x_gap = getattr(aperture, "x_gap", None)
            y_gap = getattr(aperture, "y_gap", None)
            if x_gap is not None and y_gap is not None:
                return f"slit {float(x_gap) * 1_000:g} × {float(y_gap) * 1_000:g} mm"
        if aperture_type == "pinhole":
            diameter = getattr(aperture, "diameter", None)
            if diameter is not None:
                return f"pinhole Ø {float(diameter) * 1_000:g} mm"
        return str(aperture_type)

    def _update_mask_controls(self) -> None:
        enabled = self._selected_config_id() is not None and self.table.isEnabled()
        self.choose_mask_button.setEnabled(enabled)
        self.find_mask_button.setEnabled(enabled)

    def _update_flatfield_button(
        self,
        row: ConfigurationSettingsRow | None = None,
    ) -> None:
        if row is None:
            config_id = self._selected_config_id()
            row = next(
                (
                    self.model.row_at(index)
                    for index in range(self.model.rowCount())
                    if self.model.row_at(index) is not None
                    and self.model.row_at(index).config_id == config_id
                ),
                None,
            )
        if row is None:
            self.build_flatfield_button.setEnabled(False)
            self.build_flatfield_button.setText("Create flatfield")
            return
        existing = row.flatfield_status != "not created"
        self.build_flatfield_button.setText(
            "Rebuild flatfield" if existing else "Create flatfield"
        )
        self.build_flatfield_button.setEnabled(self.table.isEnabled())
        self.choose_flatfield_button.setEnabled(self.table.isEnabled())

    def _populate_flatfield_sources(self, config_id: str) -> None:
        workflow = self._workflow
        if workflow is None:
            return
        configurations = sorted({key.config_id for key in workflow.runs})
        selected = workflow.get_flatfield_source(config_id) or config_id
        blocker = QtCore.QSignalBlocker(self.flatfield_source_combo)
        self.flatfield_source_combo.clear()
        self.flatfield_source_combo.addItem("This configuration", config_id)
        for candidate in configurations:
            if candidate != config_id:
                self.flatfield_source_combo.addItem(candidate, candidate)
        index = self.flatfield_source_combo.findData(selected)
        self.flatfield_source_combo.setCurrentIndex(max(0, index))
        del blocker
        self._update_flatfield_source_controls()

    def _update_flatfield_source_controls(self) -> None:
        enabled = self._selected_config_id() is not None and self.table.isEnabled()
        self.flatfield_source_combo.setEnabled(enabled)
        self.use_flatfield_source_button.setEnabled(enabled)

    def _populate_transmission_sources(self, config_id: str) -> None:
        workflow = self._workflow
        if workflow is None:
            return
        configurations = sorted({key.config_id for key in workflow.runs})
        selected = workflow.get_transmission_source(config_id) or config_id
        blocker = QtCore.QSignalBlocker(self.transmission_source_combo)
        self.transmission_source_combo.clear()
        self.transmission_source_combo.addItem("This configuration", config_id)
        for candidate in configurations:
            if candidate != config_id:
                self.transmission_source_combo.addItem(candidate, candidate)
        index = self.transmission_source_combo.findData(selected)
        self.transmission_source_combo.setCurrentIndex(max(0, index))
        del blocker
        if selected == config_id:
            self.transmission_source_label.setText(
                "Uses transmissions measured for this configuration."
            )
        else:
            self.transmission_source_label.setText(
                f"Uses transmissions from {selected}."
            )
        self._update_transmission_source_controls()

    def _update_transmission_source_controls(self) -> None:
        enabled = self._selected_config_id() is not None and self.table.isEnabled()
        self.transmission_source_combo.setEnabled(enabled)
        self.use_transmission_source_button.setEnabled(enabled)

    @staticmethod
    def _coordinate_spin() -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        # Off-axis detector panels can have a beam center outside their local
        # pixel frame, including at negative coordinates.
        spin.setRange(-1_000_000.0, 1_000_000.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        return spin

    @staticmethod
    def _pixel_spin() -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(0, 1_000_000)
        return spin

    def _selected_config_id(self) -> str | None:
        rows = self.table.selectionModel().selectedRows()
        row = self.model.row_at(rows[0].row()) if rows else None
        return None if row is None else row.config_id

    def _purpose(self) -> str:
        return str(self.purpose_combo.currentData())

    def _detector_number(self) -> int:
        value = self.detector_combo.currentData()
        return 0 if value is None else int(value)

    def _populate_detectors(self) -> None:
        config_id = self._selected_config_id()
        current = self._detector_number()
        detectors: set[int] = {0}
        if self._workflow is not None and config_id is not None:
            detectors.update(self._workflow.beam_centers.get(config_id, {}))
            configuration = self._workflow.configurations.get(config_id)
            distances = getattr(configuration, "sample_detector_distance", None)
            if isinstance(distances, list):
                detectors.update(range(len(distances)))
        blocker = QtCore.QSignalBlocker(self.detector_combo)
        self.detector_combo.clear()
        for detector in sorted(detectors):
            self.detector_combo.addItem(str(detector), detector)
        index = self.detector_combo.findData(current)
        self.detector_combo.setCurrentIndex(max(0, index))
        del blocker

    @QtCore.Slot()
    def _purpose_changed(self, *_args: object) -> None:
        is_center = self._purpose() == "center"
        if not is_center:
            detector_zero = self.detector_combo.findData(0)
            if detector_zero >= 0:
                self.detector_combo.setCurrentIndex(detector_zero)
        self.numeric_editors.setCurrentIndex(0 if is_center else 1)
        self.detector_combo.setEnabled(is_center)
        self.image_view.set_edit_mode("center" if is_center else "roi")
        self._metadata_selection_changed()

    @QtCore.Slot()
    def _metadata_selection_changed(self, *_args: object) -> None:
        workflow = self._workflow
        config_id = self._selected_config_id()
        if workflow is None or config_id is None:
            return
        center = workflow.get_beam_center(config_id, self._detector_number())
        if center is not None:
            self._set_center_fields(center)
        source_config = workflow.resolve_transmission_config(config_id)
        roi = workflow.get_roi(source_config)
        if roi is not None:
            self._set_roi_fields(roi)
        self._update_overlays_from_fields()

    @QtCore.Slot()
    def _request_image(self) -> None:
        config_id = self._selected_config_id()
        if config_id is not None:
            detector = self._detector_number() if self._purpose() == "center" else 0
            self.image_requested.emit(config_id, self._purpose(), detector)

    @QtCore.Slot()
    def _request_auto_detection(self) -> None:
        config_id = self._selected_config_id()
        if config_id is not None:
            detector = self._detector_number() if self._purpose() == "center" else 0
            self.auto_detect_requested.emit(config_id, self._purpose(), detector)

    @QtCore.Slot()
    def _save_center(self) -> None:
        config_id = self._selected_config_id()
        if config_id is not None:
            self.beam_center_change_requested.emit(
                config_id,
                self._detector_number(),
                (self.center_x_spin.value(), self.center_y_spin.value()),
            )

    @QtCore.Slot()
    def _save_roi(self) -> None:
        config_id = self._selected_config_id()
        roi = self._roi_from_fields()
        if config_id is None:
            return
        if not (roi[0] < roi[1] and roi[2] < roi[3]):
            self.image_source_label.setText("ROI end values must be greater than start values.")
            QtWidgets.QApplication.beep()
            return
        self.roi_change_requested.emit(config_id, roi)

    @QtCore.Slot()
    def _request_flatfield_build(self) -> None:
        config_id = self._selected_config_id()
        if config_id is not None:
            self.flatfield_build_requested.emit(config_id)

    @QtCore.Slot()
    def _request_flatfield_file(self) -> None:
        config_id = self._selected_config_id()
        if config_id is not None:
            self.flatfield_file_requested.emit(config_id)

    @QtCore.Slot()
    def _set_flatfield_source(self) -> None:
        config_id = self._selected_config_id()
        source_config_id = self.flatfield_source_combo.currentData()
        if config_id is not None and isinstance(source_config_id, str):
            self.flatfield_source_change_requested.emit(config_id, source_config_id)

    @QtCore.Slot()
    def _set_transmission_source(self) -> None:
        config_id = self._selected_config_id()
        source_config_id = self.transmission_source_combo.currentData()
        if config_id is not None and isinstance(source_config_id, str):
            self.transmission_source_change_requested.emit(config_id, source_config_id)

    @QtCore.Slot()
    def _request_mask_file(self) -> None:
        config_id = self._selected_config_id()
        if config_id is not None:
            self.mask_file_requested.emit(config_id)

    @QtCore.Slot()
    def _request_compatible_mask(self) -> None:
        config_id = self._selected_config_id()
        if config_id is not None:
            self.mask_auto_find_requested.emit(config_id)

    @QtCore.Slot(float, float)
    def _center_edited_on_image(self, x: float, y: float) -> None:
        self._set_center_fields((x, y))

    @QtCore.Slot(object)
    def _roi_edited_on_image(self, roi: object) -> None:
        if isinstance(roi, tuple) and len(roi) == 4:
            self._set_roi_fields(roi)

    def _set_center_fields(self, center: tuple[float, float]) -> None:
        self._center_available = True
        blockers = [QtCore.QSignalBlocker(self.center_x_spin), QtCore.QSignalBlocker(self.center_y_spin)]
        self.center_x_spin.setValue(float(center[0]))
        self.center_y_spin.setValue(float(center[1]))
        del blockers
        self._update_overlays_from_fields()

    def _set_roi_fields(self, roi: tuple[int, int, int, int]) -> None:
        self._roi_available = True
        spins = (self.roi_x0_spin, self.roi_x1_spin, self.roi_y0_spin, self.roi_y1_spin)
        blockers = [QtCore.QSignalBlocker(spin) for spin in spins]
        for spin, value in zip(spins, roi, strict=True):
            spin.setValue(int(value))
        del blockers
        self._update_overlays_from_fields()

    def _roi_from_fields(self) -> tuple[int, int, int, int]:
        return (
            self.roi_x0_spin.value(),
            self.roi_x1_spin.value(),
            self.roi_y0_spin.value(),
            self.roi_y1_spin.value(),
        )

    @QtCore.Slot()
    def _update_overlays_from_fields(self, *_args: object) -> None:
        self.image_view.set_overlays(
            center=(self.center_x_spin.value(), self.center_y_spin.value())
            if self._center_available
            else None,
            roi=self._roi_from_fields() if self._roi_available else None,
        )

    @QtCore.Slot()
    def _center_fields_changed(self, *_args: object) -> None:
        self._center_available = True
        self._update_overlays_from_fields()

    @QtCore.Slot()
    def _roi_fields_changed(self, *_args: object) -> None:
        self._roi_available = True
        self._update_overlays_from_fields()

    def _set_coordinate_bounds(self, shape: tuple[int, int]) -> None:
        ny, nx = shape
        # Beam centers are expressed in the local detector frame but can lie
        # outside it for offset detector panels. ROI coordinates, however,
        # must remain inside the displayed transmission image.
        for spin in (self.roi_x0_spin, self.roi_x1_spin):
            spin.setMaximum(nx)
        for spin in (self.roi_y0_spin, self.roi_y1_spin):
            spin.setMaximum(ny)
