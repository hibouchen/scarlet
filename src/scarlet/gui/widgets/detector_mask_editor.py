from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PySide6 import QtCore, QtWidgets

from scarlet.gui.mask_editor import MaskEditorSource
from scarlet.gui.scarlet_viewer import _DetectorPlotTab, _load_silx_qt
from scarlet.gui.nxsas_viewer import _detector_tab_label


class DetectorMaskEditorWidget(QtWidgets.QWidget):
    """Reusable silx detector display with the SCARLET mask-selection tools."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        qt, colormap, plot_2d = _load_silx_qt()
        self._silx_qt = qt
        self._colormap = colormap
        self._plot_2d = plot_2d
        self._source: MaskEditorSource | None = None
        self._detector_tabs: dict[int, _DetectorPlotTab] = {}

        self.source_label = QtWidgets.QLabel(
            "Select an acquisition to display its detector images."
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
    def source(self) -> MaskEditorSource | None:
        return self._source

    @property
    def has_source(self) -> bool:
        return self._source is not None and bool(self._detector_tabs)

    def clear(self, message: str = "Select an acquisition to display its detector images.") -> None:
        self._source = None
        self._detector_tabs.clear()
        self.tabs.clear()
        self.source_label.setText(message)

    def set_source(
        self,
        source: MaskEditorSource,
        masks: Mapping[int, np.ndarray] | None = None,
        *,
        display_count_time: float | None = None,
    ) -> None:
        self._source = source
        self._detector_tabs.clear()
        self.tabs.clear()
        existing_masks = masks or {}
        normalization = self._normalization()
        for detector_number, data in sorted(source.detector_data.items()):
            image = np.asarray(data, dtype=np.float64)
            mask = existing_masks.get(detector_number)
            initial_mask = None
            if mask is not None and np.asarray(mask).shape == image.shape:
                initial_mask = np.asarray(mask, dtype=np.uint8)
            dataset_path = f"{source.entry_path}/instrument/detector{detector_number}/data"
            tab = _DetectorPlotTab(
                self._silx_qt,
                self._plot_2d,
                self._colormap,
                dataset_path,
                image,
                initial_mask=initial_mask,
            )
            tab.set_colormap(normalization=normalization, reset_zoom=True)
            self._detector_tabs[detector_number] = tab
            self.tabs.addTab(tab.widget, _detector_tab_label(dataset_path))

        if self._detector_tabs:
            display_text = "corrected counts"
            if display_count_time is not None:
                display_text += f" (rate × {display_count_time:g} s)"
            self.source_label.setText(
                f"{source.file_path.name} · {display_text} · draw on the image to edit the mask "
                "(masked pixels are selected)."
            )
        else:
            self.source_label.setText("This acquisition contains no 2D detector image.")

    def masks(self) -> dict[int, np.ndarray]:
        return {
            detector_number: tab.get_mask_data()
            for detector_number, tab in self._detector_tabs.items()
        }

    @QtCore.Slot(str)
    def _set_scale_mode(self, _text: str) -> None:
        normalization = self._normalization()
        for tab in self._detector_tabs.values():
            tab.set_colormap(normalization=normalization, reset_zoom=False)

    def _normalization(self) -> str:
        return "log" if self.scale_combo.currentText().lower() == "log" else "linear"
