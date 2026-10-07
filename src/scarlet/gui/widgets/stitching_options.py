from __future__ import annotations

from PySide6 import QtWidgets

from scarlet.workflow.pipeline import StitchingOptions


class StitchingOptionsDialog(QtWidgets.QDialog):
    """Edit all greedy-stitching parameters except scaling choices."""

    def __init__(
        self,
        options: StitchingOptions,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stitching options")
        self.setMinimumWidth(440)

        self.start_policy = QtWidgets.QComboBox()
        self.start_policy.addItem("Lowest q", "lowest_q")
        self.start_policy.addItem("Best low-q quality", "best_low_q_quality")
        self.grid = QtWidgets.QComboBox()
        self.grid.addItem("Common grid", "common")
        self.grid.addItem("First segment grid", "a")
        self.grid.addItem("Second segment grid", "b")
        self.n_common = QtWidgets.QSpinBox()
        self.n_common.setRange(0, 1_000_000)
        self.n_common.setSpecialValueText("Automatic")
        self.min_points = self._integer_spin(2, 1_000_000)

        self.min_log_width = self._number_spin()
        self.slope_weight = self._number_spin()
        self.width_weight = self._number_spin()
        self.resolution_weight = self._number_spin()
        self.rho_ref = self._number_spin()
        self.max_chi2_red = self._number_spin()
        self.max_slope_z = self._number_spin()
        self.min_new_log_coverage = self._number_spin()
        self.segment_quality_weight = self._number_spin()
        self.new_coverage_weight = self._number_spin()
        self.keep_fraction = self._number_spin()

        form = QtWidgets.QFormLayout()
        form.addRow("Start segment", self.start_policy)
        form.addRow("Overlap grid", self.grid)
        form.addRow("Common grid points", self.n_common)
        form.addRow("Minimum overlap points", self.min_points)
        form.addRow("Minimum log(q) overlap width", self.min_log_width)
        form.addRow("Slope weight", self.slope_weight)
        form.addRow("Width weight", self.width_weight)
        form.addRow("Resolution weight", self.resolution_weight)
        form.addRow("Reference correlation (rho)", self.rho_ref)
        form.addRow("Maximum reduced chi²", self.max_chi2_red)
        form.addRow("Maximum slope z-score", self.max_slope_z)
        form.addRow("Minimum new log(q) coverage", self.min_new_log_coverage)
        form.addRow("Segment quality weight", self.segment_quality_weight)
        form.addRow("New coverage weight", self.new_coverage_weight)
        form.addRow("Keep fraction", self.keep_fraction)

        self.restore_defaults_button = QtWidgets.QPushButton("Restore defaults")
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.addButton(self.restore_defaults_button, QtWidgets.QDialogButtonBox.ButtonRole.ResetRole)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.restore_defaults_button.clicked.connect(lambda: self.set_options(StitchingOptions()))
        self.set_options(options)

    @staticmethod
    def _integer_spin(minimum: int, maximum: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        return spin

    @staticmethod
    def _number_spin() -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, 1_000_000.0)
        spin.setDecimals(6)
        spin.setSingleStep(0.05)
        return spin

    def set_options(self, options: StitchingOptions) -> None:
        self.start_policy.setCurrentIndex(self.start_policy.findData(options.start_policy))
        self.grid.setCurrentIndex(self.grid.findData(options.grid))
        self.n_common.setValue(0 if options.n_common is None else options.n_common)
        self.min_points.setValue(options.min_points)
        for widget, value in (
            (self.min_log_width, options.min_log_width),
            (self.slope_weight, options.slope_weight),
            (self.width_weight, options.width_weight),
            (self.resolution_weight, options.resolution_weight),
            (self.rho_ref, options.rho_ref),
            (self.max_chi2_red, options.max_chi2_red),
            (self.max_slope_z, options.max_slope_z),
            (self.min_new_log_coverage, options.min_new_log_coverage),
            (self.segment_quality_weight, options.segment_quality_weight),
            (self.new_coverage_weight, options.new_coverage_weight),
            (self.keep_fraction, options.keep_fraction),
        ):
            widget.setValue(value)

    def options(self) -> StitchingOptions:
        """Return validated algorithm options from the dialog widgets."""
        n_common = self.n_common.value()
        return StitchingOptions(
            start_policy=str(self.start_policy.currentData()),
            grid=str(self.grid.currentData()),
            n_common=None if n_common == 0 else n_common,
            min_points=self.min_points.value(),
            min_log_width=self.min_log_width.value(),
            slope_weight=self.slope_weight.value(),
            width_weight=self.width_weight.value(),
            resolution_weight=self.resolution_weight.value(),
            rho_ref=self.rho_ref.value(),
            max_chi2_red=self.max_chi2_red.value(),
            max_slope_z=self.max_slope_z.value(),
            min_new_log_coverage=self.min_new_log_coverage.value(),
            segment_quality_weight=self.segment_quality_weight.value(),
            new_coverage_weight=self.new_coverage_weight.value(),
            keep_fraction=self.keep_fraction.value(),
        )
