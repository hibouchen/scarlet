from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets


class ExperimentPage(QtWidgets.QWidget):
    """Collect the parameters needed to create or open a workflow."""

    create_requested = QtCore.Signal(object)
    open_requested = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.raw_directory_edit = QtWidgets.QLineEdit()
        self.output_directory_edit = QtWidgets.QLineEdit()
        self.instrument_combo = QtWidgets.QComboBox()
        self.instrument_combo.addItem("SANSLLB", "sansllb")
        self.instrument_combo.addItem("SAM", "sam")
        self.experiment_id_edit = QtWidgets.QLineEdit("experiment")
        self.transmission_combo = QtWidgets.QComboBox()
        self.transmission_combo.addItem("Opaque beamstop", "opaque_beamstop")
        self.transmission_combo.addItem("Semi-transparent beamstop", "semi_transparent_beamstop")

        raw_browse = QtWidgets.QPushButton("Browse…")
        output_browse = QtWidgets.QPushButton("Browse…")
        raw_browse.clicked.connect(self._browse_raw_directory)
        output_browse.clicked.connect(self._browse_output_directory)

        form = QtWidgets.QFormLayout()
        form.addRow("Raw data", self._path_row(self.raw_directory_edit, raw_browse))
        form.addRow("Output", self._path_row(self.output_directory_edit, output_browse))
        form.addRow("Instrument", self.instrument_combo)
        form.addRow("Experiment ID", self.experiment_id_edit)
        form.addRow("Transmission", self.transmission_combo)

        self.open_button = QtWidgets.QPushButton("Open workflow…")
        self.scan_button = QtWidgets.QPushButton("Scan acquisitions")
        self.scan_button.setDefault(True)
        self.open_button.clicked.connect(self._choose_workflow)
        self.scan_button.clicked.connect(self._request_create)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.scan_button)

        title = QtWidgets.QLabel("Start a workflow")
        title.setObjectName("pageTitle")
        intro = QtWidgets.QLabel(
            "Choose a raw acquisition folder. SCARLET detects the runs without converting them."
        )
        intro.setWordWrap(True)

        content = QtWidgets.QWidget()
        content.setMaximumWidth(850)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.addWidget(title)
        content_layout.addWidget(intro)
        content_layout.addSpacing(16)
        content_layout.addLayout(form)
        content_layout.addSpacing(12)
        content_layout.addLayout(buttons)
        content_layout.addStretch(1)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(content)
        layout.addStretch(1)

        self.raw_directory_edit.editingFinished.connect(self._fill_default_output)

    @staticmethod
    def _path_row(line_edit: QtWidgets.QLineEdit, button: QtWidgets.QPushButton) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        layout.addWidget(button)
        return widget

    def set_initial_raw_directory(self, path: str | Path) -> None:
        resolved = Path(path).resolve()
        self.raw_directory_edit.setText(str(resolved))
        self.output_directory_edit.setText(str(resolved / "processed"))

    def set_initial_instrument(self, instrument: str) -> None:
        index = self.instrument_combo.findData(instrument.lower())
        if index >= 0:
            self.instrument_combo.setCurrentIndex(index)

    def set_busy(self, busy: bool) -> None:
        self.scan_button.setDisabled(busy)
        self.open_button.setDisabled(busy)

    @QtCore.Slot()
    def _browse_raw_directory(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select raw acquisition folder",
            self.raw_directory_edit.text(),
        )
        if selected:
            self.raw_directory_edit.setText(selected)
            self._fill_default_output()

    @QtCore.Slot()
    def _browse_output_directory(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self.output_directory_edit.text() or self.raw_directory_edit.text(),
        )
        if selected:
            self.output_directory_edit.setText(selected)

    @QtCore.Slot()
    def _fill_default_output(self) -> None:
        raw_text = self.raw_directory_edit.text().strip()
        if raw_text and not self.output_directory_edit.text().strip():
            self.output_directory_edit.setText(str(Path(raw_text).expanduser() / "processed"))

    @QtCore.Slot()
    def _choose_workflow(self) -> None:
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open SCARLET workflow",
            "",
            "NeXus workflow (*.nxs *.h5 *.hdf5);;All files (*)",
        )
        if selected:
            self.open_requested.emit(Path(selected))

    @QtCore.Slot()
    def _request_create(self) -> None:
        raw_text = self.raw_directory_edit.text().strip()
        output_text = self.output_directory_edit.text().strip()
        experiment_id = self.experiment_id_edit.text().strip()
        if not raw_text or not output_text or not experiment_id:
            QtWidgets.QMessageBox.warning(
                self,
                "Incomplete workflow",
                "Raw data, output, and experiment ID are required.",
            )
            return
        self.create_requested.emit(
            {
                "input_dir": Path(raw_text).expanduser(),
                "output_dir": Path(output_text).expanduser(),
                "experiment_id": experiment_id,
                "instrument_name": str(self.instrument_combo.currentData()),
                "transmission_strategy": str(self.transmission_combo.currentData()),
            }
        )

