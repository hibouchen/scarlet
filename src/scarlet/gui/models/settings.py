from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

from PySide6 import QtCore, QtGui

from scarlet.workflow import WorkflowContext


@dataclass(frozen=True)
class ConfigurationSettingsRow:
    config_id: str
    centers_status: str
    roi_status: str
    masks_status: str
    flatfield_status: str
    status: str
    centers_detail: str
    roi_detail: str
    masks_detail: str
    flatfield_detail: str


def configuration_settings_rows(
    workflow: WorkflowContext,
) -> list[ConfigurationSettingsRow]:
    """Build one preparation-status row for every known configuration."""
    # ``workflow.runs`` is the canonical active registry: excluding the last
    # run of a configuration must also remove that configuration from Settings.
    # Derived metadata may intentionally remain in the context for audit and
    # possible reinstatement, so it must not create rows by itself.
    config_ids = sorted({key.config_id for key in workflow.runs})
    rows: list[ConfigurationSettingsRow] = []
    for config_id in config_ids:
        centers_status, centers_detail, centers_ready = _centers_state(workflow, config_id)
        roi_status, roi_detail, roi_ready = _roi_state(workflow, config_id)
        masks_status, masks_detail, masks_ready = _masks_state(workflow, config_id)
        flatfield_status, flatfield_detail, flatfield_ready = _flatfield_state(
            workflow,
            config_id,
        )
        rows.append(
            ConfigurationSettingsRow(
                config_id=config_id,
                centers_status=centers_status,
                roi_status=roi_status,
                masks_status=masks_status,
                flatfield_status=flatfield_status,
                status=(
                    "ready"
                    if centers_ready and roi_ready and masks_ready and flatfield_ready
                    else "incomplete"
                ),
                centers_detail=centers_detail,
                roi_detail=roi_detail,
                masks_detail=masks_detail,
                flatfield_detail=flatfield_detail,
            )
        )
    return rows


def _centers_state(
    workflow: WorkflowContext,
    config_id: str,
) -> tuple[str, str, bool]:
    centers = workflow.beam_centers.get(config_id, {})
    valid_centers = {
        detector: center
        for detector, center in centers.items()
        if len(center) == 2 and all(math.isfinite(float(value)) for value in center)
    }
    configuration = workflow.configurations.get(config_id)
    detector_distance = getattr(configuration, "sample_detector_distance", None)
    expected_count = (
        len(detector_distance)
        if isinstance(detector_distance, list)
        else 1 if detector_distance is not None else None
    )
    if not centers:
        expected = f" for {expected_count} detector(s)" if expected_count else ""
        return "missing", f"No beam center is stored{expected}.", False
    if len(valid_centers) != len(centers):
        return "invalid", "At least one beam center contains a non-finite value.", False

    detail = "; ".join(
        f"detector{detector}: ({center[0]:.3f}, {center[1]:.3f})"
        for detector, center in sorted(valid_centers.items())
    )
    if expected_count is not None:
        expected_detectors = set(range(expected_count))
        complete = set(valid_centers) == expected_detectors
        return (
            f"{len(valid_centers)}/{expected_count} defined",
            detail,
            complete,
        )
    return f"{len(valid_centers)} defined", detail, True


def _roi_state(
    workflow: WorkflowContext,
    config_id: str,
) -> tuple[str, str, bool]:
    try:
        source_config = workflow.resolve_transmission_config(config_id)
    except ValueError as exc:
        return "invalid source", str(exc), False
    roi = workflow.get_roi(source_config)
    source_text = "" if source_config == config_id else f" from {source_config}"
    if roi is None:
        return "missing", f"No transmission ROI is stored{source_text}.", False
    x0, x1, y0, y1 = roi
    valid = 0 <= x0 < x1 and 0 <= y0 < y1
    if not valid:
        return "invalid", f"Invalid ROI: ({x0}, {x1}, {y0}, {y1}).", False
    return f"defined{source_text}", f"x={x0}:{x1}, y={y0}:{y1}{source_text}", True


def _masks_state(
    workflow: WorkflowContext,
    config_id: str,
) -> tuple[str, str, bool]:
    mask_file = workflow.get_mask_file(config_id)
    masks = workflow.masks.get(config_id, {})
    if mask_file is not None:
        if mask_file.exists():
            return "assigned", f"Mask bundle: {mask_file}", True
        return "missing file", f"Assigned mask bundle was not found: {mask_file}", False
    if masks:
        detectors = ", ".join(f"detector{number}" for number in sorted(masks))
        return f"{len(masks)} defined", f"In-memory masks: {detectors}", True
    return "not assigned", "No detector mask is assigned.", False


def _flatfield_state(
    workflow: WorkflowContext,
    config_id: str,
) -> tuple[str, str, bool]:
    try:
        source_config = workflow.resolve_flatfield_config(config_id)
        flatfield = workflow.get_flatfield(config_id)
    except ValueError as exc:
        return "invalid source", str(exc), False
    source_text = "" if source_config == config_id else f" from {source_config}"
    if flatfield is None:
        return "not created", f"No flatfield is assigned{source_text}.", False
    if not flatfield.exists():
        return "missing file", f"Flatfield file was not found: {flatfield}", False
    if source_config in workflow.stale_flatfields:
        return "stale", f"Flatfield must be rebuilt: {flatfield}", False
    return f"ready{source_text}", f"Flatfield: {flatfield}", True


class ConfigurationSettingsModel(QtCore.QAbstractTableModel):
    """Read-only overview of configuration preparation state."""

    _COLUMNS: tuple[tuple[str, str], ...] = (
        ("config_id", "Configuration"),
        ("centers_status", "Beam centers"),
        ("roi_status", "Transmission ROI"),
        ("masks_status", "Masks"),
        ("flatfield_status", "Flatfield"),
        ("status", "Status"),
    )

    def __init__(
        self,
        rows: Sequence[ConfigurationSettingsRow] = (),
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._rows = list(rows)

    def set_workflow(self, workflow: WorkflowContext | None) -> None:
        self.beginResetModel()
        self._rows = [] if workflow is None else configuration_settings_rows(workflow)
        self.endResetModel()

    def row_at(self, row: int) -> ConfigurationSettingsRow | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
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
        row = self._rows[index.row()]
        value = getattr(row, field)
        if role in {
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        }:
            return str(value)
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return {
                "centers_status": row.centers_detail,
                "roi_status": row.roi_detail,
                "masks_status": row.masks_detail,
                "flatfield_status": row.flatfield_detail,
            }.get(field)
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and field == "status":
            return QtGui.QColor("#2e7d32" if row.status == "ready" else "#b71c1c")
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
