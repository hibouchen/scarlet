from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from PySide6 import QtCore

from scarlet.workflow import WorkflowContext


@dataclass(frozen=True)
class ReferenceTransmissionRow:
    """Effective water and empty-cell transmission values for one configuration."""

    config_id: str
    water_transmission: str
    empty_cell_transmission: str


def reference_transmission_rows(workflow: WorkflowContext) -> list[ReferenceTransmissionRow]:
    """Return transmission-reference values for configurations entering reduction."""
    config_ids = sorted(
        {
            run_key.config_id
            for run_key in workflow.runs
            if run_key.entity == "sample" and run_key.mode == "scattering"
        }
    )
    rows: list[ReferenceTransmissionRow] = []
    for config_id in config_ids:
        source_config_id = workflow.resolve_transmission_config(config_id)
        water_names = sorted(
            {
                run_key.sample_name
                for run_key in workflow.runs
                if run_key.config_id == source_config_id
                and run_key.entity == "water"
                and run_key.sample_name is not None
            }
        )
        water_value = next(
            (
                workflow.get_transmission(water_name, config_id)
                for water_name in water_names
                if workflow.get_transmission(water_name, config_id) is not None
            ),
            None,
        )
        empty_cell_value = workflow.get_empty_cell_transmission(config_id)
        rows.append(
            ReferenceTransmissionRow(
                config_id=config_id,
                water_transmission=_format_transmission(water_value),
                empty_cell_transmission=_format_transmission(empty_cell_value),
            )
        )
    return rows


def _format_transmission(value: float | None) -> str:
    return "—" if value is None else f"{float(value):.6g}"


class ReferenceTransmissionsModel(QtCore.QAbstractTableModel):
    """Read-only processing-page table of reference transmissions."""

    _COLUMNS: tuple[tuple[str, str], ...] = (
        ("config_id", "Configuration"),
        ("water_transmission", "Water transmission"),
        ("empty_cell_transmission", "Empty-cell transmission"),
    )

    def __init__(
        self,
        rows: Sequence[ReferenceTransmissionRow] = (),
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._rows = list(rows)

    def set_workflow(self, workflow: WorkflowContext | None) -> None:
        self.beginResetModel()
        self._rows = [] if workflow is None else reference_transmission_rows(workflow)
        self.endResetModel()

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
            return str(value)
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole and field != "config_id":
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
