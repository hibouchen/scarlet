from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
from PySide6 import QtCore, QtWidgets

from scarlet.gui.mask_editor import MaskEditorSource
from scarlet.gui.nxsas_viewer import _detector_tab_label
from scarlet.gui.scarlet_viewer import _DetectorPlotTab, _load_silx_qt
from scarlet.reduction import Sector


class SectorPreviewWidget(QtWidgets.QWidget):
    """Read-only detector display with directional integration sectors."""

    _COLORS = ("#ef5350", "#42a5f5", "#ab47bc", "#26a69a")

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        qt, colormap, plot_2d = _load_silx_qt()
        self._silx_qt = qt
        self._colormap = colormap
        self._plot_2d = plot_2d
        self._source: MaskEditorSource | None = None
        self._detector_tabs: dict[int, _DetectorPlotTab] = {}
        self._beam_centers: dict[int, tuple[float, float]] = {}
        self._sectors: tuple[Sector, ...] = ()
        self._sector_items: dict[int, list[object]] = {}

        self.source_label = QtWidgets.QLabel(
            "Select one acquisition to display its detector image."
        )
        self.source_label.setWordWrap(True)
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(["Linear", "Log"])
        self.scale_combo.currentTextChanged.connect(self._set_scale_mode)
        self.tabs = QtWidgets.QTabWidget()

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.source_label, 1)
        controls.addWidget(QtWidgets.QLabel("Scale"))
        controls.addWidget(self.scale_combo)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.tabs, 1)

    @property
    def has_source(self) -> bool:
        return self._source is not None and bool(self._detector_tabs)

    def clear(
        self,
        message: str = "Select one acquisition to display its detector image.",
    ) -> None:
        self._source = None
        self._detector_tabs.clear()
        self._beam_centers.clear()
        self._sector_items.clear()
        self.tabs.clear()
        self.source_label.setText(message)

    def set_source(
        self,
        source: MaskEditorSource,
        *,
        display_count_time: float | None = None,
        beam_centers: Mapping[int, tuple[float, float]] | None = None,
    ) -> None:
        self._source = source
        self._detector_tabs.clear()
        self._sector_items.clear()
        self.tabs.clear()
        normalization = self._normalization()
        for detector_number, data in sorted(source.detector_data.items()):
            image = np.asarray(data, dtype=np.float64)
            dataset_path = f"{source.entry_path}/instrument/detector{detector_number}/data"
            tab = _DetectorPlotTab(
                self._silx_qt,
                self._plot_2d,
                self._colormap,
                dataset_path,
                image,
            )
            tab.set_colormap(normalization=normalization, reset_zoom=True)
            self._detector_tabs[detector_number] = tab
            self.tabs.addTab(tab.widget, _detector_tab_label(dataset_path))

        self.set_beam_centers(beam_centers or {})
        if self._detector_tabs:
            display_text = "corrected counts"
            if display_count_time is not None:
                display_text += f" (rate × {display_count_time:g} s)"
            self.source_label.setText(f"{source.file_path.name} · {display_text}")
        else:
            self.source_label.setText("This acquisition contains no 2D detector image.")

    def set_sectors(self, sectors: Sequence[Sector]) -> None:
        self._sectors = tuple(sectors)
        self._render_sector_overlays()

    def set_beam_centers(
        self,
        beam_centers: Mapping[int, tuple[float, float]],
    ) -> None:
        normalized: dict[int, tuple[float, float]] = {}
        for detector_number, center in beam_centers.items():
            try:
                x, y = float(center[0]), float(center[1])
            except (IndexError, TypeError, ValueError):
                continue
            if np.isfinite(x) and np.isfinite(y):
                normalized[int(detector_number)] = (x, y)
        self._beam_centers = normalized
        for detector_number, tab in self._detector_tabs.items():
            tab.set_beam_center(self._beam_centers.get(detector_number))
        self._render_sector_overlays()

    def sector_item_count(self, detector_number: int) -> int:
        """Return the number of displayed wedges, mainly for GUI diagnostics."""
        return len(self._sector_items.get(int(detector_number), ()))

    def _render_sector_overlays(self) -> None:
        for detector_number, tab in self._detector_tabs.items():
            for item in self._sector_items.get(detector_number, ()):
                tab.widget.removeItem(item)
            self._sector_items[detector_number] = []
            center = self._beam_centers.get(detector_number)
            if center is None:
                continue
            height, width = tab.data.shape[-2:]
            radius = 1.05 * max(
                math.hypot(x - center[0], y - center[1])
                for x, y in ((0, 0), (width, 0), (0, height), (width, height))
            )
            for sector_index, sector in enumerate(self._sectors):
                directions = [sector.center_deg]
                if sector.include_opposite:
                    directions.append(sector.center_deg + 180.0)
                for direction_index, direction in enumerate(directions):
                    angles = np.deg2rad(
                        np.linspace(
                            direction - sector.width_deg / 2.0,
                            direction + sector.width_deg / 2.0,
                            48,
                        )
                    )
                    x_data = np.concatenate(
                        ([center[0]], center[0] + radius * np.cos(angles), [center[0]])
                    )
                    y_data = np.concatenate(
                        ([center[1]], center[1] + radius * np.sin(angles), [center[1]])
                    )
                    item = tab.widget.addShape(
                        x_data,
                        y_data,
                        legend=f"sector_{sector_index}_{direction_index}",
                        shape="polygon",
                        color=self._COLORS[sector_index % len(self._COLORS)],
                        fill=False,
                        overlay=True,
                        z=10,
                        linewidth=2.5,
                    )
                    self._sector_items[detector_number].append(item)

    @QtCore.Slot(str)
    def _set_scale_mode(self, _text: str) -> None:
        normalization = self._normalization()
        for tab in self._detector_tabs.values():
            tab.set_colormap(normalization=normalization, reset_zoom=False)
        self._render_sector_overlays()

    def _normalization(self) -> str:
        return "log" if self.scale_combo.currentText().lower() == "log" else "linear"
