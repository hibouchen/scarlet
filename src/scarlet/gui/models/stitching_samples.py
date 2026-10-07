from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PySide6 import QtCore, QtGui

from scarlet.workflow import WorkflowContext
from scarlet.workflow.pipeline import StichingPipeline


@dataclass(frozen=True)
class StitchingSampleRow:
    """One sample and the state of its merged text output."""

    sample_name: str
    configuration_count: int
    status: str
    output_path: Path


def stitching_sample_rows(workflow: WorkflowContext) -> list[StitchingSampleRow]:
    """Return sample-scattering groups with output status from ``output_dir``."""
    configurations_by_sample: dict[str, set[str]] = {}
    for run_key in workflow.runs:
        if (
            run_key.entity == "sample"
            and run_key.mode == "scattering"
            and run_key.sample_name is not None
        ):
            configurations_by_sample.setdefault(run_key.sample_name, set()).add(run_key.config_id)

    pipeline = StichingPipeline(workflow)
    rows: list[StitchingSampleRow] = []
    for sample_name, configurations in sorted(configurations_by_sample.items()):
        output_path = (workflow.output_dir / f"{sample_name}_merged.txt").resolve()
        # The control table intentionally uses presence rather than timestamps:
        # it is the direct, visible record requested by the operator.
        if output_path.is_file():
            status = "stitched"
        elif pipeline.is_sample_ready(sample_name):
            status = "ready"
        else:
            status = "waiting"
        rows.append(
            StitchingSampleRow(
                sample_name=sample_name,
                configuration_count=len(configurations),
                status=status,
                output_path=output_path,
            )
        )
    return rows


class StitchingSamplesModel(QtCore.QAbstractTableModel):
    """Read-only control table for stitching sample groups."""

    _COLUMNS: tuple[tuple[str, str], ...] = (
        ("sample_name", "Sample"),
        ("configuration_count", "Configurations"),
        ("status", "Status"),
        ("output_path", "Merged output"),
    )

    def __init__(
        self,
        rows: Sequence[StitchingSampleRow] = (),
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._rows = list(rows)

    def set_workflow(self, workflow: WorkflowContext | None) -> None:
        self.beginResetModel()
        self._rows = [] if workflow is None else stitching_sample_rows(workflow)
        self.endResetModel()

    @classmethod
    def column_index(cls, field: str) -> int:
        for index, (column_field, _label) in enumerate(cls._COLUMNS):
            if column_field == field:
                return index
        raise KeyError(field)

    def sample_name_at(self, row: int) -> str | None:
        return self._rows[row].sample_name if 0 <= row < len(self._rows) else None

    def status_count(self, status: str) -> int:
        return sum(row.status == status for row in self._rows)

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._COLUMNS)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        field, _label = self._COLUMNS[index.column()]
        value = getattr(self._rows[index.row()], field)
        if role in {QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole}:
            return value.name if field == "output_path" else str(value)
        if role == QtCore.Qt.ItemDataRole.ToolTipRole and field == "output_path":
            return str(value)
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and field == "status":
            return {
                "stitched": QtGui.QColor("#2e7d32"),
                "ready": QtGui.QColor("#8a5a00"),
                "waiting": QtGui.QColor("#b71c1c"),
            }.get(str(value))
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole and field in {
            "configuration_count",
            "status",
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
