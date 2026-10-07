from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Sequence

from PySide6 import QtCore, QtGui

from scarlet.workflow import RunKey, WorkflowContext


@dataclass(frozen=True)
class RunTableRow:
    run_key: RunKey
    sample_name: str
    config_id: str
    mode: str
    entity: str
    review_warning: str
    status: str
    thickness: str
    file_path: Path


def workflow_run_rows(workflow: WorkflowContext) -> list[RunTableRow]:
    """Build stable, presentation-ready rows from a workflow context."""
    status_by_key = {
        row["run_key"]: row["status"]
        for row in workflow.runs_status_table().rows
    }
    warnings_by_path: dict[Path, str] = {}
    for entry in workflow.logs:
        path_value = entry.meta.get("key")
        if (
            entry.level == "WARN"
            and entry.message.startswith("Could not confidently infer measurement mode")
            and path_value
        ):
            guessed_mode = entry.meta.get("guessed_mode")
            suffix = f" Detected fallback: {guessed_mode}." if guessed_mode else ""
            warnings_by_path[Path(str(path_value)).resolve()] = f"{entry.message}.{suffix}".replace("..", ".")
        elif (
            entry.where == "replace_run"
            and entry.meta.get("previous_mode") != entry.meta.get("new_mode")
            and entry.meta.get("file_path")
        ):
            warnings_by_path.pop(Path(str(entry.meta["file_path"])).resolve(), None)
    rows: list[RunTableRow] = []
    for run_key, registered_path in workflow.runs.items():
        thickness = (
            workflow.get_sample_thickness(run_key.sample_name, run_key.config_id)
            if run_key.sample_name
            else None
        )
        rows.append(
            RunTableRow(
                run_key=run_key,
                sample_name=run_key.sample_name or "",
                config_id=run_key.config_id,
                mode=run_key.mode,
                entity=run_key.entity,
                review_warning=warnings_by_path.get(registered_path.resolve(), ""),
                status=str(status_by_key[run_key.short()]),
                thickness="" if thickness is None else f"{float(thickness):.6g}",
                file_path=Path(registered_path.name),
            )
        )
    return rows


class RunsTableModel(QtCore.QAbstractTableModel):
    """Qt adapter for acquisitions with constrained metadata editing."""

    mode_change_requested = QtCore.Signal(object, str)
    entity_change_requested = QtCore.Signal(object, str)
    thickness_change_requested = QtCore.Signal(object, object)
    sample_name_change_requested = QtCore.Signal(object, str)

    _COLUMNS: tuple[tuple[str, str], ...] = (
        ("sample_name", "Sample"),
        ("config_id", "Configuration"),
        ("mode", "Mode"),
        ("entity", "Entity"),
        ("review_warning", "Review"),
        ("status", "Status"),
        ("thickness", "Thickness"),
        ("file_path", "File"),
    )

    def __init__(
        self,
        rows: Sequence[RunTableRow] = (),
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._rows = list(rows)

    def set_workflow(self, workflow: WorkflowContext | None) -> None:
        self.beginResetModel()
        self._rows = [] if workflow is None else workflow_run_rows(workflow)
        self.endResetModel()

    @classmethod
    def column_index(cls, field: str) -> int:
        for index, (column_field, _label) in enumerate(cls._COLUMNS):
            if column_field == field:
                return index
        raise KeyError(field)

    def run_key_at(self, row: int) -> RunKey | None:
        if 0 <= row < len(self._rows):
            return self._rows[row].run_key
        return None

    def warning_at(self, row: int) -> str:
        if 0 <= row < len(self._rows):
            return self._rows[row].review_warning
        return ""

    @property
    def warning_count(self) -> int:
        return sum(bool(row.review_warning) for row in self._rows)

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._COLUMNS)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if (
            not index.isValid()
            or not (0 <= index.row() < len(self._rows))
            or not (0 <= index.column() < len(self._COLUMNS))
        ):
            return None
        field, _label = self._COLUMNS[index.column()]
        value = getattr(self._rows[index.row()], field)
        if role in {
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        }:
            if field == "review_warning":
                return "⚠" if value else ""
            return str(value)
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            if field == "file_path":
                return str(value)
            if self._rows[index.row()].review_warning:
                return self._rows[index.row()].review_warning
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and field == "status":
            return {
                "processed": QtGui.QColor("#2e7d32"),
                "missing": QtGui.QColor("#8a5a00"),
                "unreadable": QtGui.QColor("#b71c1c"),
            }.get(str(value))
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole and field in {"status", "mode", "entity"}:
            return int(QtCore.Qt.AlignmentFlag.AlignCenter)
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole and field == "review_warning":
            return int(QtCore.Qt.AlignmentFlag.AlignCenter)
        if role == QtCore.Qt.ItemDataRole.BackgroundRole and self._rows[index.row()].review_warning:
            return QtGui.QColor("#fff4ce")
        return None

    def setData(
        self,
        index: QtCore.QModelIndex,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if (
            role != QtCore.Qt.ItemDataRole.EditRole
            or not index.isValid()
            or not (0 <= index.row() < len(self._rows))
            or not (0 <= index.column() < len(self._COLUMNS))
        ):
            return False
        field = self._COLUMNS[index.column()][0]
        row = self._rows[index.row()]
        if field == "sample_name":
            if row.entity != "sample":
                return False
            sample_name = str(value).strip()
            if not sample_name:
                return False
            if sample_name == row.sample_name:
                return True
            self._rows[index.row()] = replace(row, sample_name=sample_name)
            self.dataChanged.emit(
                index,
                index,
                [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole],
            )
            self.sample_name_change_requested.emit(row.run_key, sample_name)
            return True

        if field == "thickness":
            if row.entity not in {"sample", "water"} or not row.sample_name:
                return False
            text = str(value).strip().replace(",", ".")
            if text == "":
                thickness = None
                display_value = ""
            else:
                try:
                    thickness = float(text)
                except ValueError:
                    return False
                if not math.isfinite(thickness) or thickness <= 0.0:
                    return False
                display_value = f"{thickness:.12g}"
            if display_value == row.thickness:
                return True
            self._rows[index.row()] = replace(row, thickness=display_value)
            self.dataChanged.emit(
                index,
                index,
                [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole],
            )
            self.thickness_change_requested.emit(row.run_key, thickness)
            return True

        allowed_values = {
            "mode": {"scattering", "transmission"},
            "entity": {"sample", "empty_beam", "empty_cell", "dark", "water"},
        }
        if field not in allowed_values:
            return False
        normalized_value = str(value).strip().lower()
        if normalized_value not in allowed_values[field]:
            return False
        if normalized_value == getattr(row, field):
            return True
        self._rows[index.row()] = replace(row, **{field: normalized_value})
        self.dataChanged.emit(
            index,
            index,
            [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole],
        )
        if field == "mode":
            self.mode_change_requested.emit(row.run_key, normalized_value)
        else:
            self.entity_change_requested.emit(row.run_key, normalized_value)
        return True

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal and 0 <= section < len(self._COLUMNS):
            return self._COLUMNS[section][1]
        if orientation == QtCore.Qt.Orientation.Vertical:
            return section + 1
        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags
        flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        field = self._COLUMNS[index.column()][0]
        row = self._rows[index.row()]
        if field in {"mode", "entity"} or (field == "sample_name" and row.entity == "sample") or (
            field == "thickness"
            and row.entity in {"sample", "water"}
            and bool(row.sample_name)
        ):
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        return flags
