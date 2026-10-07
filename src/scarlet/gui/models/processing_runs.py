from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PySide6 import QtCore, QtGui

from scarlet.workflow import RunKey, WorkflowContext


@dataclass(frozen=True)
class ProcessingRunRow:
    run_key: RunKey
    sample_name: str
    config_id: str
    status: str
    transmission: str
    thickness: str
    file_path: Path


def processing_run_rows(workflow: WorkflowContext) -> list[ProcessingRunRow]:
    """Return sample-scattering acquisitions that can enter reduction."""
    rows: list[ProcessingRunRow] = []
    for run_key, file_path in workflow.runs.items():
        if run_key.entity != "sample" or run_key.mode != "scattering":
            continue
        if workflow.is_run_processed(run_key):
            status = "processed"
        elif file_path.exists():
            status = "pending"
        else:
            status = "unavailable"
        thickness = workflow.get_sample_thickness(
            run_key.sample_name or "",
            run_key.config_id,
        )
        transmission = workflow.get_transmission(
            run_key.sample_name or "",
            run_key.config_id,
        )
        rows.append(
            ProcessingRunRow(
                run_key=run_key,
                sample_name=run_key.sample_name or "",
                config_id=run_key.config_id,
                status=status,
                transmission="—" if transmission is None else f"{float(transmission):.6g}",
                thickness="" if thickness is None else f"{float(thickness):.6g}",
                file_path=Path(file_path.name),
            )
        )
    return rows


class ProcessingRunsModel(QtCore.QAbstractTableModel):
    """Read-only model for selecting sample-scattering reduction candidates."""

    _COLUMNS: tuple[tuple[str, str], ...] = (
        ("sample_name", "Sample"),
        ("config_id", "Configuration"),
        ("status", "Status"),
        ("transmission", "Transmission"),
        ("thickness", "Thickness (mm)"),
        ("file_path", "File"),
    )

    def __init__(
        self,
        rows: Sequence[ProcessingRunRow] = (),
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._rows = list(rows)

    def set_workflow(self, workflow: WorkflowContext | None) -> None:
        self.beginResetModel()
        self._rows = [] if workflow is None else processing_run_rows(workflow)
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

    def status_count(self, status: str) -> int:
        return sum(row.status == status for row in self._rows)

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
            return str(value)
        if role == QtCore.Qt.ItemDataRole.ToolTipRole and field == "file_path":
            return str(value)
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and field == "status":
            return {
                "processed": QtGui.QColor("#2e7d32"),
                "pending": QtGui.QColor("#8a5a00"),
                "unavailable": QtGui.QColor("#b71c1c"),
            }.get(str(value))
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole and field in {
            "status",
            "transmission",
            "thickness",
        }:
            return int(QtCore.Qt.AlignmentFlag.AlignCenter)
        return None

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if (
            role == QtCore.Qt.ItemDataRole.DisplayRole
            and orientation == QtCore.Qt.Orientation.Horizontal
            and 0 <= section < len(self._COLUMNS)
        ):
            return self._COLUMNS[section][1]
        return super().headerData(section, orientation, role)
