from __future__ import annotations

import re

import math

from PySide6 import QtCore, QtGui, QtWidgets

from scarlet.workflow import WorkflowContext

from ..models import RunsTableModel
from ..readiness import WorkflowReadiness, assess_workflow_readiness
from ..widgets import DetectorMaskEditorWidget


def _natural_sort_key(value: object) -> tuple[tuple[int, object], ...]:
    """Return a case-insensitive key that keeps config_2 before config_10."""
    parts = re.split(r"(\d+)", str(value).casefold())
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in parts
        if part
    )


class _RunsSortProxyModel(QtCore.QSortFilterProxyModel):
    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._search_text = ""
        self._warning_only = False
        self._filters: dict[str, str] = {
            "status": "",
            "entity": "",
            "mode": "",
            "config_id": "",
        }

    def set_search_text(self, text: str) -> None:
        self.beginFilterChange()
        self._search_text = text.strip().casefold()
        self.endFilterChange(QtCore.QSortFilterProxyModel.Direction.Rows)

    def set_warning_only(self, enabled: bool) -> None:
        self.beginFilterChange()
        self._warning_only = bool(enabled)
        self.endFilterChange(QtCore.QSortFilterProxyModel.Direction.Rows)

    def set_field_filter(self, field: str, value: str) -> None:
        if field not in self._filters:
            raise KeyError(field)
        self.beginFilterChange()
        self._filters[field] = value
        self.endFilterChange(QtCore.QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(self, source_row: int, source_parent: QtCore.QModelIndex) -> bool:
        source = self.sourceModel()
        if self._warning_only and not source.warning_at(source_row):
            return False
        for field, expected in self._filters.items():
            if not expected:
                continue
            column = RunsTableModel.column_index(field)
            value = source.data(
                source.index(source_row, column, source_parent),
                QtCore.Qt.ItemDataRole.EditRole,
            )
            if str(value) != expected:
                return False
        if not self._search_text:
            return True
        searchable_fields = (
            "sample_name",
            "config_id",
            "mode",
            "entity",
            "status",
            "file_path",
        )
        return any(
            self._search_text
            in str(
                source.data(
                    source.index(source_row, RunsTableModel.column_index(field), source_parent),
                    QtCore.Qt.ItemDataRole.DisplayRole,
                )
            ).casefold()
            for field in searchable_fields
        )

    def lessThan(self, left: QtCore.QModelIndex, right: QtCore.QModelIndex) -> bool:
        source = self.sourceModel()
        left_value = source.data(left, QtCore.Qt.ItemDataRole.EditRole)
        right_value = source.data(right, QtCore.Qt.ItemDataRole.EditRole)
        if left.column() == RunsTableModel.column_index("thickness"):
            left_number = float(left_value) if str(left_value).strip() else float("-inf")
            right_number = float(right_value) if str(right_value).strip() else float("-inf")
            return left_number < right_number
        return _natural_sort_key(left_value) < _natural_sort_key(right_value)


class _ChoiceDelegate(QtWidgets.QStyledItemDelegate):
    """Use a constrained combo box for one-click table edits."""

    def __init__(self, choices: list[str], parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._choices = choices

    def createEditor(self, parent, option, index):
        editor = QtWidgets.QComboBox(parent)
        editor.addItems(self._choices)
        QtCore.QTimer.singleShot(0, editor.showPopup)
        return editor

    def setEditorData(self, editor, index) -> None:
        value = str(index.model().data(index, QtCore.Qt.ItemDataRole.EditRole))
        editor.setCurrentText(value)

    def setModelData(self, editor, model, index) -> None:
        model.setData(index, editor.currentText(), QtCore.Qt.ItemDataRole.EditRole)


class _PositiveOrEmptyValidator(QtGui.QValidator):
    def validate(self, text: str, position: int):
        normalized = text.strip().replace(",", ".")
        if normalized == "":
            return QtGui.QValidator.State.Acceptable, text, position
        if not re.fullmatch(r"\d*(?:\.\d*)?", normalized):
            return QtGui.QValidator.State.Invalid, text, position
        if normalized == ".":
            return QtGui.QValidator.State.Intermediate, text, position
        try:
            value = float(normalized)
        except ValueError:
            return QtGui.QValidator.State.Invalid, text, position
        if value == 0.0:
            return QtGui.QValidator.State.Intermediate, text, position
        state = QtGui.QValidator.State.Acceptable if math.isfinite(value) else QtGui.QValidator.State.Invalid
        return state, text, position


class _ThicknessDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = QtWidgets.QLineEdit(parent)
        editor.setValidator(_PositiveOrEmptyValidator(editor))
        editor.setPlaceholderText("positive value or empty")
        editor.setToolTip("Enter a strictly positive thickness in mm, or leave empty.")
        editor.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        return editor

    def setEditorData(self, editor, index) -> None:
        editor.setText(str(index.model().data(index, QtCore.Qt.ItemDataRole.EditRole)))
        editor.selectAll()

    def setModelData(self, editor, model, index) -> None:
        if not model.setData(index, editor.text(), QtCore.Qt.ItemDataRole.EditRole):
            QtWidgets.QApplication.beep()


class _SampleNameDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = QtWidgets.QLineEdit(parent)
        editor.setPlaceholderText("sample name")
        return editor

    def setEditorData(self, editor, index) -> None:
        editor.setText(str(index.model().data(index, QtCore.Qt.ItemDataRole.EditRole)))
        editor.selectAll()

    def setModelData(self, editor, model, index) -> None:
        if not model.setData(index, editor.text(), QtCore.Qt.ItemDataRole.EditRole):
            QtWidgets.QApplication.beep()


class AcquisitionsPage(QtWidgets.QWidget):
    refresh_requested = QtCore.Signal()
    reconsider_excluded_requested = QtCore.Signal()
    import_csv_requested = QtCore.Signal()
    export_csv_requested = QtCore.Signal()
    mode_change_requested = QtCore.Signal(object, str)
    entity_change_requested = QtCore.Signal(object, str)
    thickness_change_requested = QtCore.Signal(object, object)
    sample_name_change_requested = QtCore.Signal(object, str)
    exclude_requested = QtCore.Signal(object)
    preview_requested = QtCore.Signal(object)
    mask_save_requested = QtCore.Signal(str, object, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._has_workflow = False
        self._has_excluded = False
        self._busy = False
        self._preview_run_key: object | None = None
        self._requested_preview_run_key: object | None = None
        self.model = RunsTableModel(parent=self)
        self.model.mode_change_requested.connect(self.mode_change_requested.emit)
        self.model.entity_change_requested.connect(self.entity_change_requested.emit)
        self.model.thickness_change_requested.connect(self.thickness_change_requested.emit)
        self.model.sample_name_change_requested.connect(self.sample_name_change_requested.emit)
        self.sort_model = _RunsSortProxyModel(self)
        self.sort_model.setSourceModel(self.model)
        self.sort_model.setDynamicSortFilter(True)
        self.summary_label = QtWidgets.QLabel("No workflow loaded")
        self.refresh_button = QtWidgets.QPushButton("Refresh folder")
        self.refresh_button.clicked.connect(self.refresh_requested)
        self.reconsider_button = QtWidgets.QPushButton("Reconsider excluded")
        self.reconsider_button.clicked.connect(self.reconsider_excluded_requested)
        self.exclude_button = QtWidgets.QPushButton("Exclude selected")
        self.exclude_button.clicked.connect(self._exclude_selected)
        self.exclude_button.setEnabled(False)
        self.import_csv_button = QtWidgets.QPushButton("Import CSV…")
        self.import_csv_button.clicked.connect(self.import_csv_requested)
        self.export_csv_button = QtWidgets.QPushButton("Export CSV…")
        self.export_csv_button.clicked.connect(self.export_csv_requested)

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.addWidget(self.refresh_button)
        toolbar.addWidget(self.reconsider_button)
        toolbar.addWidget(self.exclude_button)
        toolbar.addWidget(self.import_csv_button)
        toolbar.addWidget(self.export_csv_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.summary_label)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Search acquisitions…")
        self.search_edit.setClearButtonEnabled(True)
        self.warning_only_check = QtWidgets.QCheckBox("Only detections to review")
        self.status_filter = self._filter_combo(
            "All statuses",
            ["processed", "unprocessed", "missing", "unreadable"],
        )
        self.mode_filter = self._filter_combo(
            "All modes",
            ["scattering", "transmission"],
        )
        self.entity_filter = self._filter_combo(
            "All entities",
            ["sample", "empty_beam", "empty_cell", "dark", "water"],
        )
        self.config_filter = self._filter_combo("All configurations", [])
        filters = QtWidgets.QHBoxLayout()
        filters.addWidget(self.search_edit, 1)
        filters.addWidget(self.status_filter)
        filters.addWidget(self.mode_filter)
        filters.addWidget(self.entity_filter)
        filters.addWidget(self.config_filter)
        filters.addWidget(self.warning_only_check)

        self.search_edit.textChanged.connect(self.sort_model.set_search_text)
        self.warning_only_check.toggled.connect(self.sort_model.set_warning_only)
        self.status_filter.currentIndexChanged.connect(
            lambda: self.sort_model.set_field_filter("status", str(self.status_filter.currentData()))
        )
        self.mode_filter.currentIndexChanged.connect(
            lambda: self.sort_model.set_field_filter("mode", str(self.mode_filter.currentData()))
        )
        self.entity_filter.currentIndexChanged.connect(
            lambda: self.sort_model.set_field_filter("entity", str(self.entity_filter.currentData()))
        )
        self.config_filter.currentIndexChanged.connect(
            lambda: self.sort_model.set_field_filter("config_id", str(self.config_filter.currentData()))
        )

        self.table = QtWidgets.QTableView()
        self.table.setModel(self.sort_model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        mode_column = self.model.column_index("mode")
        entity_column = self.model.column_index("entity")
        self.table.setItemDelegateForColumn(
            mode_column,
            _ChoiceDelegate(["scattering", "transmission"], self.table),
        )
        self.table.setItemDelegateForColumn(
            entity_column,
            _ChoiceDelegate(
                ["sample", "empty_beam", "empty_cell", "dark", "water"],
                self.table,
            ),
        )
        self.table.setItemDelegateForColumn(
            self.model.column_index("thickness"),
            _ThicknessDelegate(self.table),
        )
        self.table.setItemDelegateForColumn(
            self.model.column_index("sample_name"),
            _SampleNameDelegate(self.table),
        )
        self.table.clicked.connect(self._edit_choice_on_click)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        self.table.setSortingEnabled(True)
        self.sort_model.sort(-1)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSortIndicatorShown(False)
        header.sectionClicked.connect(lambda _section: header.setSortIndicatorShown(True))
        # Acquisition metadata has very different useful widths. Keep the
        # initial width readable while letting scientists resize every column.
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(48)
        header.setDefaultSectionSize(130)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)

        self.review_label = QtWidgets.QLabel()
        self.review_label.setWordWrap(True)
        self.review_label.setStyleSheet("color: #8a5a00;")

        self.readiness_label = QtWidgets.QLabel()
        self.readiness_label.setWordWrap(True)
        self.readiness_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.mask_editor = DetectorMaskEditorWidget()
        self.mask_editor.setMinimumHeight(260)
        self.save_mask_button = QtWidgets.QPushButton("Save and attach mask…")
        self.save_mask_button.clicked.connect(self._save_mask)
        self.save_mask_button.setEnabled(False)
        mask_title = QtWidgets.QLabel("Detector preview and mask")
        mask_title.setObjectName("sectionTitle")
        mask_header = QtWidgets.QHBoxLayout()
        mask_header.addWidget(mask_title)
        mask_header.addStretch(1)
        mask_header.addWidget(self.save_mask_button)
        mask_panel = QtWidgets.QWidget()
        mask_layout = QtWidgets.QVBoxLayout(mask_panel)
        mask_layout.setContentsMargins(0, 0, 0, 0)
        mask_layout.addLayout(mask_header)
        mask_layout.addWidget(self.mask_editor, 1)

        self.content_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.content_splitter.addWidget(self.table)
        self.content_splitter.addWidget(mask_panel)
        self.content_splitter.setStretchFactor(0, 3)
        self.content_splitter.setStretchFactor(1, 2)
        self.content_splitter.setSizes([760, 500])

        title = QtWidgets.QLabel("Acquisitions")
        title.setObjectName("pageTitle")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(toolbar)
        layout.addLayout(filters)
        layout.addWidget(self.readiness_label)
        layout.addWidget(self.content_splitter, 1)
        layout.addWidget(self.review_label)

    def set_workflow(self, workflow: WorkflowContext | None) -> None:
        selected_run_keys = self._selected_run_keys()
        self._has_workflow = workflow is not None
        selection_model = self.table.selectionModel()
        blocker = QtCore.QSignalBlocker(selection_model)
        self.model.set_workflow(workflow)
        self._update_configuration_filter(workflow)
        self._restore_selected_run_keys(selected_run_keys)
        del blocker
        self.import_csv_button.setEnabled(self._has_workflow)
        self.export_csv_button.setEnabled(self._has_workflow)
        if workflow is None:
            self.summary_label.setText("No workflow loaded")
            self._set_readiness(None)
            self._has_excluded = False
            self.reconsider_button.setEnabled(False)
            self.clear_preview()
            return
        processed = sum(1 for key in workflow.runs if workflow.is_run_processed(key))
        warning_count = self.model.warning_count
        warning_text = f" · {warning_count} to review" if warning_count else ""
        self.summary_label.setText(
            f"{len(workflow.runs)} acquisitions · {processed} processed{warning_text}"
        )
        self._has_excluded = bool(workflow.active_excluded_files())
        self.reconsider_button.setEnabled(self._has_excluded)
        self._set_readiness(assess_workflow_readiness(workflow))
        self._update_preview_beam_centers(workflow)
        self._selection_changed()

    def set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        preview_loading = busy and message == "Loading acquisition detector image…"
        self.refresh_button.setDisabled(busy)
        self.reconsider_button.setEnabled(not busy and self._has_excluded)
        self.exclude_button.setDisabled(busy or not self.table.selectionModel().hasSelection())
        self.import_csv_button.setEnabled(not busy and self._has_workflow)
        self.export_csv_button.setEnabled(not busy and self._has_workflow)
        self.table.setEnabled(not busy or preview_loading)
        self.mask_editor.setDisabled(busy)
        self.save_mask_button.setEnabled(not busy and self.mask_editor.has_source)
        for widget in (
            self.search_edit,
            self.warning_only_check,
            self.status_filter,
            self.mode_filter,
            self.entity_filter,
            self.config_filter,
        ):
            widget.setDisabled(busy)

    def refresh_readiness(self, workflow: WorkflowContext | None) -> None:
        """Refresh derived preparation status without rebuilding the detector editor."""
        self._set_readiness(None if workflow is None else assess_workflow_readiness(workflow))

    def _set_readiness(self, report: WorkflowReadiness | None) -> None:
        if report is None:
            self.readiness_label.setText("Processing readiness will appear here.")
            self.readiness_label.setToolTip("")
            self.readiness_label.setStyleSheet("color: #666;")
            return
        details = [*report.errors, *report.warnings]
        if not report.ready:
            text = f"Not ready for processing · {len(report.errors)} blocking issue(s)"
            color = "#b71c1c"
        elif report.warnings:
            text = (
                f"Ready for processing · {report.sample_count} sample acquisition(s) "
                f"· {len(report.warnings)} warning(s)"
            )
            color = "#8a5a00"
        else:
            text = f"Ready for processing · {report.sample_count} sample acquisition(s)"
            color = "#2e7d32"
        if details:
            text = f"{text} — {details[0]}"
        self.readiness_label.setText(text)
        self.readiness_label.setToolTip("\n".join(details))
        self.readiness_label.setStyleSheet(f"color: {color};")

    @staticmethod
    def _filter_combo(label: str, values: list[str]) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItem(label, "")
        for value in values:
            combo.addItem(value, value)
        return combo

    def _update_configuration_filter(self, workflow: WorkflowContext | None) -> None:
        selected = str(self.config_filter.currentData() or "")
        configurations = [] if workflow is None else sorted({key.config_id for key in workflow.runs})
        blocker = QtCore.QSignalBlocker(self.config_filter)
        self.config_filter.clear()
        self.config_filter.addItem("All configurations", "")
        for config_id in configurations:
            self.config_filter.addItem(config_id, config_id)
        selected_index = self.config_filter.findData(selected)
        self.config_filter.setCurrentIndex(max(0, selected_index))
        del blocker
        self.sort_model.set_field_filter(
            "config_id",
            str(self.config_filter.currentData() or ""),
        )

    def _selected_run_keys(self) -> list[object]:
        selected_run_keys: list[object] = []
        for proxy_index in self.table.selectionModel().selectedRows():
            source_index = self.sort_model.mapToSource(proxy_index)
            run_key = self.model.run_key_at(source_index.row())
            if run_key is not None:
                selected_run_keys.append(run_key)
        return selected_run_keys

    def _restore_selected_run_keys(self, run_keys: list[object]) -> None:
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
            run_key = self.model.run_key_at(source_row)
            if run_key not in wanted:
                continue
            proxy_index = self.sort_model.mapFromSource(self.model.index(source_row, 0))
            if proxy_index.isValid():
                selection_model.select(proxy_index, flags)

    @QtCore.Slot(QtCore.QModelIndex)
    def _edit_choice_on_click(self, index: QtCore.QModelIndex) -> None:
        editable_columns = {
            self.model.column_index("mode"),
            self.model.column_index("entity"),
            self.model.column_index("thickness"),
            self.model.column_index("sample_name"),
        }
        if (
            index.isValid()
            and index.column() in editable_columns
            and index.flags() & QtCore.Qt.ItemFlag.ItemIsEditable
            and not self._busy
        ):
            self.table.edit(index)

    @QtCore.Slot()
    def _selection_changed(self, *_args: object) -> None:
        rows = self.table.selectionModel().selectedRows()
        self.exclude_button.setEnabled(bool(rows) and self.table.isEnabled() and not self._busy)
        self.exclude_button.setText(
            f"Exclude selected ({len(rows)})" if len(rows) > 1 else "Exclude selected"
        )
        source_row = self.sort_model.mapToSource(rows[0]).row() if rows else -1
        warning = self.model.warning_at(source_row)
        self.review_label.setText(warning)
        self.review_label.setVisible(bool(warning))
        if not rows:
            # A raw-file preview can refresh the workflow model after
            # conversion, which transiently clears the table selection.
            # Keep the pending preview request in that case.
            if not self._busy:
                self.clear_preview()
            return
        run_key = self.model.run_key_at(source_row)
        if run_key is not None:
            if run_key in {self._preview_run_key, self._requested_preview_run_key}:
                return
            self._preview_run_key = None
            self._requested_preview_run_key = run_key
            self.mask_editor.clear("Loading selected acquisition…")
            self.save_mask_button.setEnabled(False)
            self.preview_requested.emit(run_key)

    @QtCore.Slot(object)
    def set_preview(self, payload: object) -> None:
        source = getattr(payload, "source", None)
        run_key = getattr(payload, "run_key", None)
        masks = getattr(payload, "masks", None)
        beam_centers = getattr(payload, "beam_centers", {})
        display_count_time = getattr(payload, "display_count_time", None)
        if source is None or run_key is None:
            self.clear_preview()
            return
        if (
            self._requested_preview_run_key is not None
            and run_key != self._requested_preview_run_key
        ):
            return
        self._preview_run_key = run_key
        self.mask_editor.set_source(
            source,
            masks,
            display_count_time=display_count_time,
            beam_centers=beam_centers,
        )
        self.save_mask_button.setEnabled(self.mask_editor.has_source and self.table.isEnabled())

    def clear_preview(self) -> None:
        self._preview_run_key = None
        self._requested_preview_run_key = None
        self.mask_editor.clear()
        self.save_mask_button.setEnabled(False)

    def _update_preview_beam_centers(self, workflow: WorkflowContext) -> None:
        run_key = self._preview_run_key
        config_id = getattr(run_key, "config_id", None)
        if not isinstance(config_id, str):
            return
        self.mask_editor.set_beam_centers(workflow.beam_centers.get(config_id, {}))

    @QtCore.Slot()
    def _save_mask(self) -> None:
        source = self.mask_editor.source
        if source is None or not self.mask_editor.has_source:
            return
        run_key = self._preview_run_key
        if run_key is None:
            return
        self.mask_save_requested.emit(run_key.config_id, source, self.mask_editor.masks())

    @QtCore.Slot()
    def _exclude_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        run_keys = []
        for proxy_index in rows:
            source_row = self.sort_model.mapToSource(proxy_index).row()
            run_key = self.model.run_key_at(source_row)
            if run_key is not None:
                run_keys.append(run_key)
        if run_keys:
            self.exclude_requested.emit(run_keys)
