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
    output_label: str
    output_tooltip: str


def stitching_sample_rows(
    workflow: WorkflowContext,
    curve_group: str | None = None,
) -> list[StitchingSampleRow]:
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
        if curve_group == "*":
            sector_names = pipeline.available_sector_names(sample_name)
            output_paths = tuple(
                pipeline.output_path(sample_name, sector_name)
                for sector_name in sector_names
            )
            output_path = output_paths[0] if output_paths else workflow.output_dir.resolve()
            output_label = (
                f"{len(output_paths)} sector output(s)"
                if output_paths
                else "No common sector"
            )
            output_tooltip = "\n".join(str(path) for path in output_paths)
            ready = bool(sector_names) and all(
                pipeline.is_sample_ready(sample_name, sector_name)
                for sector_name in sector_names
            )
            stitched = ready and all(path.is_file() for path in output_paths)
        else:
            sector_name = curve_group or None
            output_path = pipeline.output_path(sample_name, sector_name)
            output_label = output_path.name
            output_tooltip = str(output_path)
            ready = pipeline.is_sample_ready(sample_name, sector_name)
            stitched = output_path.is_file()
        # The control table intentionally uses presence rather than timestamps:
        # it is the direct, visible record requested by the operator.
        if stitched:
            status = "stitched"
        elif ready:
            status = "ready"
        else:
            status = "waiting"
        rows.append(
            StitchingSampleRow(
                sample_name=sample_name,
                configuration_count=len(configurations),
                status=status,
                output_path=output_path,
                output_label=output_label,
                output_tooltip=output_tooltip,
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
        self._workflow: WorkflowContext | None = None
        self._curve_group: str | None = None

    def set_workflow(self, workflow: WorkflowContext | None) -> None:
        self._workflow = workflow
        self._rebuild()

    def set_curve_group(self, curve_group: str | None) -> None:
        normalized = None if curve_group in {None, ""} else str(curve_group)
        if normalized == self._curve_group:
            return
        self._curve_group = normalized
        self._rebuild()

    def _rebuild(self) -> None:
        self.beginResetModel()
        self._rows = (
            []
            if self._workflow is None
            else stitching_sample_rows(self._workflow, self._curve_group)
        )
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
            return self._rows[index.row()].output_label if field == "output_path" else str(value)
        if role == QtCore.Qt.ItemDataRole.ToolTipRole and field == "output_path":
            return self._rows[index.row()].output_tooltip
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
