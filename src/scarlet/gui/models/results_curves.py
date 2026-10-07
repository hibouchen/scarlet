from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PySide6 import QtCore, QtGui

from scarlet.workflow import WorkflowContext


@dataclass(frozen=True)
class ResultCurveRow:
    """One text curve available from the workflow output directory."""

    path: Path
    kind: str


def result_curve_rows(workflow: WorkflowContext) -> list[ResultCurveRow]:
    """Discover reduced and stitched I(Q) text outputs in the output directory."""
    if not workflow.output_dir.is_dir():
        return []
    rows = [
        ResultCurveRow(
            path=path.resolve(),
            kind="stitched" if path.name.endswith("_merged.txt") else "reduced",
        )
        for path in workflow.output_dir.glob("*.txt")
        if path.is_file()
    ]
    return sorted(rows, key=lambda row: (row.kind != "stitched", row.path.name.casefold()))


class ResultCurvesModel(QtCore.QAbstractTableModel):
    """Read-only list of plottable I(Q) text files."""

    _COLUMNS: tuple[tuple[str, str], ...] = (("path", "Curve"), ("kind", "Type"))

    def __init__(
        self,
        rows: Sequence[ResultCurveRow] = (),
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._rows = list(rows)

    def set_workflow(self, workflow: WorkflowContext | None) -> None:
        self.beginResetModel()
        self._rows = [] if workflow is None else result_curve_rows(workflow)
        self.endResetModel()

    def path_at(self, row: int) -> Path | None:
        return self._rows[row].path if 0 <= row < len(self._rows) else None

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._COLUMNS)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        field, _label = self._COLUMNS[index.column()]
        row = self._rows[index.row()]
        if role in {QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole}:
            return row.path.name if field == "path" else row.kind
        if role == QtCore.Qt.ItemDataRole.ToolTipRole and field == "path":
            return str(row.path)
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and field == "kind":
            return QtGui.QColor("#2e7d32" if row.kind == "stitched" else "#245a92")
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
