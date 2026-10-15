from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from silx.gui.plot import Plot1D

from scarlet.workflow import WorkflowContext
from scarlet.reduction.stiching import SASCurve

from ..models import ResultCurvesModel


class ResultsPage(QtWidgets.QWidget):
    """Plot one or more reduced or stitched I(Q) curves with silx."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._workflow_context: WorkflowContext | None = None
        title = QtWidgets.QLabel("Results")
        title.setObjectName("pageTitle")
        intro = QtWidgets.QLabel(
            "Select one or more reduced or stitched curves to display them together."
        )
        intro.setWordWrap(True)
        self.summary_label = QtWidgets.QLabel("No workflow loaded")
        self.error_label = QtWidgets.QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b71c1c;")

        self.model = ResultCurvesModel(parent=self)
        self.sort_model = QtCore.QSortFilterProxyModel(self)
        self.sort_model.setSourceModel(self.model)
        self.sort_model.setDynamicSortFilter(True)
        self.sort_model.setFilterCaseSensitivity(
            QtCore.Qt.CaseSensitivity.CaseInsensitive
        )
        self.sort_model.setFilterKeyColumn(-1)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Search result files…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self.sort_model.setFilterFixedString)

        self.table = QtWidgets.QTableView()
        self.table.setModel(self.sort_model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(48)
        header.setDefaultSectionSize(180)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.selectionModel().selectionChanged.connect(self._plot_selection)

        self.plot = Plot1D()
        self.plot.getXAxis().setLabel("q")
        self.plot.getYAxis().setLabel("I(q)")
        self.plot.setGraphTitle("I(q) curves")
        self.plot.setGraphGrid(True)
        self.plot.setDataMargins(0.05, 0.05, 0.05, 0.05)
        # Plot1D provides a native, dockable silx legend panel. Keep it open
        # so every selected curve can be identified by its output-file name.
        self.legend_dock = self.plot.getLegendsDockWidget()
        self.legend_dock.setWindowTitle("Curves")
        legend_view = self.legend_dock.findChild(QtWidgets.QAbstractItemView)
        if legend_view is not None:
            palette = legend_view.palette()
            palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#ffffff"))
            palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#ffffff"))
            palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#202124"))
            palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#202124"))
            palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#245a92"))
            palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
            legend_view.setPalette(palette)
            legend_view.setStyleSheet(
                "LegendListView { background: #ffffff; color: #202124; }"
                "LegendListView::item:selected { background: #245a92; color: #ffffff; }"
            )
        self.legend_dock.show()

        self.refresh_button = QtWidgets.QPushButton("Refresh results")
        self.refresh_button.clicked.connect(self._refresh)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.refresh_button)
        controls.addStretch(1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.search_edit)
        layout.addWidget(splitter, 1)
        layout.addLayout(controls)

    def set_workflow(self, workflow: WorkflowContext | None) -> None:
        selected_paths = set(self.selected_paths())
        self._workflow_context = workflow
        self.model.set_workflow(workflow)
        self.table.clearSelection()
        for row in range(self.sort_model.rowCount()):
            source_row = self.sort_model.mapToSource(
                self.sort_model.index(row, 0)
            ).row()
            if self.model.path_at(source_row) in selected_paths:
                self.table.selectRow(row)
        if workflow is None:
            self.summary_label.setText("No workflow loaded")
            self.refresh_button.setEnabled(False)
        else:
            self.summary_label.setText(f"{self.model.rowCount()} curve(s) in output")
            self.refresh_button.setEnabled(True)
        self._plot_selection()

    def set_busy(self, busy: bool) -> None:
        self.refresh_button.setDisabled(busy)
        self.search_edit.setDisabled(busy)
        self.table.setDisabled(busy)

    def selected_paths(self) -> list[Path]:
        return [
            path
            for index in sorted(
                self.table.selectionModel().selectedRows(), key=lambda item: item.row()
            )
            if (
                path := self.model.path_at(
                    self.sort_model.mapToSource(index).row()
                )
            )
            is not None
        ]

    @QtCore.Slot()
    def _refresh(self) -> None:
        workflow = self._workflow_context
        if workflow is not None:
            self.set_workflow(workflow)

    @QtCore.Slot()
    def _plot_selection(self, *_args: object) -> None:
        self.plot.clear()
        errors: list[str] = []
        for path in self.selected_paths():
            try:
                curve = SASCurve.from_txt(path, name=path.stem)
                valid = np.isfinite(curve.q) & np.isfinite(curve.i)
                if not np.any(valid):
                    raise ValueError("contains no finite q/I values")
                y_error = curve.di if np.all(np.isfinite(curve.di)) else None
                self.plot.addCurve(
                    curve.q[valid],
                    curve.i[valid],
                    yerror=None if y_error is None else y_error[valid],
                    legend=curve.name,
                    resetzoom=False,
                )
            except (OSError, ValueError) as exc:
                errors.append(f"{path.name}: {exc}")
        self.error_label.setText("\n".join(errors))
        if self.plot.getAllCurves():
            self.plot.resetZoom()
