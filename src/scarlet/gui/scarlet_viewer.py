from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .mask_editor import MaskEditorSource, write_mask_bundle
from .nxsas_viewer import (
    NexusFileSummary,
    _DIRECT_VIEW_DEFINITIONS,
    _detector_tab_label,
    _read_nexus_definition,
    format_nexus_summary,
    list_nexus_files,
    prepare_view_file,
    read_nexus_dataset,
    scan_nexus_file,
)


def _load_silx_qt():
    from silx.gui import qt
    from silx.gui.colors import Colormap
    from silx.gui.plot import Plot2D

    return qt, Colormap, Plot2D


def _make_auto_fit_plot_class(qt, Plot2D):
    class _AutoFitPlot2D(Plot2D):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._auto_fit_on_resize = False

        def set_auto_fit_on_resize(self, enabled: bool) -> None:
            self._auto_fit_on_resize = bool(enabled)

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            if self._auto_fit_on_resize:
                qt.QTimer.singleShot(0, self.resetZoom)

    return _AutoFitPlot2D


class _DetectorPlotTab:
    def __init__(
        self,
        qt,
        Plot2D,
        Colormap,
        dataset_path: str,
        data: np.ndarray,
        *,
        initial_mask: np.ndarray | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.data = np.asarray(data, dtype=np.float64)
        AutoFitPlot2D = _make_auto_fit_plot_class(qt, Plot2D)
        self.widget = AutoFitPlot2D()
        self.widget.setKeepDataAspectRatio(True)
        self.widget.set_auto_fit_on_resize(True)
        self.widget.getXAxis().setLabel("X (pixel)")
        self.widget.getYAxis().setLabel("Y (pixel)")
        self._Colormap = Colormap
        self._colormap_name = "viridis"
        self._normalization = "linear"
        self._initial_mask = None if initial_mask is None else np.asarray(initial_mask, dtype=np.uint8)
        self._mask_initialized = False

    def set_scale_mode(self, mode: str, *, reset_zoom: bool = False) -> None:
        self._normalization = mode
        self.render(reset_zoom=reset_zoom)

    def render(self, *, reset_zoom: bool = True) -> None:
        current_mask = _get_selection_mask(self.widget)
        colormap = self._Colormap(name=self._colormap_name, normalization=self._normalization)
        self.widget.addImage(
            self.data,
            legend="image",
            replace=True,
            resetzoom=reset_zoom,
            colormap=colormap,
        )
        self.widget.setGraphTitle(_detector_tab_label(self.dataset_path))
        if current_mask is not None:
            _set_selection_mask(self.widget, current_mask)
            self._mask_initialized = True
        elif not self._mask_initialized and self._initial_mask is not None:
            _set_selection_mask(self.widget, self._initial_mask)
            self._mask_initialized = True

    def set_colormap(self, *, name: str | None = None, normalization: str | None = None, reset_zoom: bool = False) -> None:
        if name is not None:
            self._colormap_name = str(name)
        if normalization is not None:
            self._normalization = str(normalization)
        self.render(reset_zoom=reset_zoom)

    def get_colormap_state(self) -> tuple[str, str]:
        image = self.widget.getImage("image")
        if image is not None and hasattr(image, "getColormap"):
            colormap = image.getColormap()
            name_getter = getattr(colormap, "getName", None)
            normalization_getter = getattr(colormap, "getNormalization", None)
            if callable(name_getter):
                self._colormap_name = str(name_getter())
            if callable(normalization_getter):
                self._normalization = str(normalization_getter())
        return self._colormap_name, self._normalization

    def get_mask_data(self) -> np.ndarray:
        mask = _get_selection_mask(self.widget)
        if mask is None:
            return np.zeros(self.data.shape, dtype=np.uint8)
        return np.asarray(mask > 0, dtype=np.uint8)


def _get_selection_mask(plot) -> np.ndarray | None:
    try:
        mask = plot.getSelectionMask(copy=True)
    except TypeError:
        mask = plot.getSelectionMask()
    if mask is None:
        return None
    return np.array(mask, copy=True)


def _set_selection_mask(plot, mask: np.ndarray) -> None:
    try:
        plot.setSelectionMask(mask, copy=True)
    except TypeError:
        plot.setSelectionMask(mask)


def build_mask_source(source_file: Path | str, view_file: Path | str) -> MaskEditorSource:
    from scarlet.workflow.configuration import configuration_from_nexus

    source_file = Path(source_file).resolve()
    view_file = Path(view_file).resolve()
    entry_path, detector_data = _read_detector_data_for_mask_bundle(view_file)
    configuration, issues = configuration_from_nexus(view_file, entry_path=entry_path, detector_index=0)
    return MaskEditorSource(
        file_path=source_file,
        entry_path=entry_path,
        detector_data=detector_data,
        configuration=configuration,
        configuration_issues=issues,
    )


def default_mask_output_path(source_file: Path | str) -> Path:
    source_file = Path(source_file).resolve()
    return source_file.with_name(f"{source_file.stem}_masks.nxs")


def _read_detector_data_for_mask_bundle(file_path: Path | str) -> tuple[str, dict[int, np.ndarray]]:
    file_path = Path(file_path).resolve()
    with h5py.File(file_path, "r") as handle:
        definition = _read_nexus_definition(handle)
        entry_path = _resolve_entry_path(handle)
        detector_data: dict[int, np.ndarray] = {}
        if definition == "SCARLET_masks":
            mask_group_path = f"{entry_path}/mask"
            if mask_group_path not in handle:
                raise ValueError(f"Missing mask group in {file_path}: {mask_group_path}")
            for name, dataset in handle[mask_group_path].items():
                detector_index = _detector_index_from_path(f"{mask_group_path}/{name}")
                detector_data[detector_index] = np.asarray(dataset[()], dtype=np.float64)
        else:
            instrument_path = f"{entry_path}/instrument"
            if instrument_path not in handle:
                raise ValueError(f"Missing instrument group in {file_path}: {instrument_path}")
            for name, group in handle[instrument_path].items():
                if not isinstance(group, h5py.Group):
                    continue
                dataset_path = f"{instrument_path}/{name}/data"
                if dataset_path not in handle:
                    continue
                detector_index = _detector_index_from_path(dataset_path)
                detector_data[detector_index] = np.asarray(handle[dataset_path][()], dtype=np.float64)
        if not detector_data:
            raise ValueError(f"No detector data available to save masks from {file_path}")
        return entry_path, detector_data


def _resolve_entry_path(handle: h5py.File) -> str:
    for entry_path in ("/raw_data", "/entry", "/entry0", "/entry1"):
        if entry_path in handle and isinstance(handle[entry_path], h5py.Group):
            return entry_path
    raise ValueError("No entry group found in file.")


def _detector_index_from_path(dataset_path: str) -> int:
    import re

    match = re.search(r"/detector(\d+)/data$", dataset_path)
    if match is not None:
        return int(match.group(1))
    match = re.search(r"/mask/mask_detector(\d+)$", dataset_path)
    if match is not None:
        return int(match.group(1))
    raise ValueError(f"Unsupported detector dataset path: {dataset_path}")


class ViewerWindow:
    def __init__(
        self,
        *,
        initial_directory: Optional[Path] = None,
        initial_instrument: str = "sansllb",
    ) -> None:
        qt, Colormap, Plot2D = _load_silx_qt()
        self.qt = qt
        self._Colormap = Colormap
        self._Plot2D = Plot2D
        self._temp_dir = tempfile.TemporaryDirectory(prefix="scarlet_silx_viewer_")

        self.app = qt.QApplication.instance() or qt.QApplication([])
        self.window = qt.QMainWindow()
        self.window.setWindowTitle("viewer")
        self.window.resize(1500, 950)
        self.window.destroyed.connect(lambda *_args: self._temp_dir.cleanup())

        self.directory: Optional[Path] = None
        self.current_source_file: Optional[Path] = None
        self.current_view_file: Optional[Path] = None
        self.current_summary: Optional[NexusFileSummary] = None
        self.current_dataset_path: Optional[str] = None
        self.detector_tabs: dict[str, _DetectorPlotTab] = {}
        self._syncing_tree = False
        self._syncing_tabs = False
        self._files: list[Path] = []
        self._colormap_name = "viridis"
        self._normalization = "linear"

        self._build_ui(initial_instrument=initial_instrument)
        if initial_directory is not None:
            self.set_directory(initial_directory)

    def _build_ui(self, *, initial_instrument: str) -> None:
        qt = self.qt

        central = qt.QWidget()
        self.window.setCentralWidget(central)
        root_layout = qt.QHBoxLayout(central)

        left_panel = qt.QWidget()
        left_layout = qt.QVBoxLayout(left_panel)
        root_layout.addWidget(left_panel, 1)

        right_splitter = qt.QSplitter(qt.Qt.Vertical)
        root_layout.addWidget(right_splitter, 3)

        self.folder_label = qt.QLabel("Folder: -")
        self.folder_label.setWordWrap(True)
        left_layout.addWidget(self.folder_label)

        open_button = qt.QPushButton("Open Folder")
        open_button.clicked.connect(self.open_directory_dialog)
        left_layout.addWidget(open_button)

        instrument_layout = qt.QHBoxLayout()
        instrument_layout.addWidget(qt.QLabel("Instrument"))
        self.instrument_combo = qt.QComboBox()
        self.instrument_combo.addItems(["SANSLLB", "SAM"])
        self.instrument_combo.setCurrentText(initial_instrument.upper())
        self.instrument_combo.currentTextChanged.connect(self._on_instrument_changed)
        instrument_layout.addWidget(self.instrument_combo, 1)
        left_layout.addLayout(instrument_layout)

        self.file_list = qt.QListWidget()
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        left_layout.addWidget(self.file_list, 1)

        top_panel = qt.QWidget()
        top_layout = qt.QVBoxLayout(top_panel)
        right_splitter.addWidget(top_panel)

        self.summary_text = qt.QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumBlockCount(128)
        top_layout.addWidget(self.summary_text)

        self.tree = qt.QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Path", "Kind", "Shape", "Dtype", "NX_class"])
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        top_layout.addWidget(self.tree, 1)

        bottom_panel = qt.QWidget()
        bottom_layout = qt.QVBoxLayout(bottom_panel)
        right_splitter.addWidget(bottom_panel)

        preview_controls = qt.QHBoxLayout()
        self.preview_label = qt.QLabel("Select a file to display its detectors or masks.")
        self.preview_label.setWordWrap(True)
        preview_controls.addWidget(self.preview_label, 1)

        self.scale_combo = qt.QComboBox()
        self.scale_combo.addItems(["Linear", "Log"])
        self.scale_combo.currentTextChanged.connect(self._on_scale_mode_changed)
        preview_controls.addWidget(self.scale_combo)
        bottom_layout.addLayout(preview_controls)

        self.tabs = qt.QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        bottom_layout.addWidget(self.tabs, 1)

        self.status_bar = self.window.statusBar()
        self.status_bar.showMessage("Select a data folder to browse files.")

        menu = self.window.menuBar().addMenu("File")
        open_action = menu.addAction("Open Data Folder")
        open_action.triggered.connect(self.open_directory_dialog)
        save_mask_action = menu.addAction("Save Masks")
        save_mask_action.triggered.connect(self.save_masks_dialog)
        close_action = menu.addAction("Close")
        close_action.triggered.connect(self.window.close)

        right_splitter.setStretchFactor(0, 2)
        right_splitter.setStretchFactor(1, 3)

    def run(self) -> int:
        self.window.show()
        return int(self.app.exec())

    def open_directory_dialog(self) -> None:
        directory = self.qt.QFileDialog.getExistingDirectory(self.window, "Select data folder")
        if not directory:
            return
        self.set_directory(directory)

    def set_directory(self, directory: Path | str) -> None:
        directory = Path(directory).resolve()
        files = list_nexus_files(directory)
        self.directory = directory
        self._files = files
        self.folder_label.setText(f"Folder: {directory}")
        self.file_list.clear()
        self.file_list.addItems([path.name for path in files])
        self._clear_view()
        if files:
            self.file_list.setCurrentRow(0)
            self.status_bar.showMessage(f"Loaded folder {directory} ({len(files)} file(s)).")
        else:
            self.preview_label.setText("No supported NeXus file found in this folder.")
            self.status_bar.showMessage(f"No supported NeXus file found in {directory}.")

    def load_file(self, file_path: Path | str) -> None:
        self._sync_display_preferences_from_tabs()
        source_file = Path(file_path).resolve()
        prepared = prepare_view_file(
            source_file,
            apparatus=self.instrument_combo.currentText(),
            temp_dir=Path(self._temp_dir.name),
        )
        summary = scan_nexus_file(prepared.view_file)
        summary = NexusFileSummary(
            file_path=prepared.source_file,
            definition=summary.definition,
            sample_name=summary.sample_name,
            detector0_distance_m=summary.detector0_distance_m,
            collimation_distance_m=summary.collimation_distance_m,
            wavelength_a=summary.wavelength_a,
            entry_paths=summary.entry_paths,
            detector_paths=summary.detector_paths,
            image_dataset_paths=summary.image_dataset_paths,
            nodes=summary.nodes,
        )
        self.current_source_file = prepared.source_file
        self.current_view_file = prepared.view_file
        self.current_summary = summary
        self.current_dataset_path = None
        self.summary_text.setPlainText(format_nexus_summary(summary))
        self._populate_tree(summary)
        self._build_detector_tabs(summary.detector_paths)
        if summary.detector_paths:
            self._select_dataset(summary.detector_paths[0])
        else:
            self.preview_label.setText("Selected file has no detector or mask dataset to preview.")
        if prepared.converted:
            self.status_bar.showMessage(
                f"Loaded {prepared.source_file.name} via temporary {prepared.apparatus} conversion."
            )
        else:
            self.status_bar.showMessage(f"Loaded {prepared.source_file.name}")

    def _populate_tree(self, summary: NexusFileSummary) -> None:
        self.tree.clear()
        items: dict[str, object] = {}
        for node in summary.nodes:
            parent_path = "" if node.path.count("/") == 1 else node.path.rsplit("/", 1)[0]
            if parent_path:
                parent_item = items[parent_path]
                item = self.qt.QTreeWidgetItem(parent_item)
            else:
                item = self.qt.QTreeWidgetItem(self.tree)
            item.setText(0, node.name)
            item.setText(1, node.kind)
            item.setText(2, "" if node.shape is None else str(node.shape))
            item.setText(3, node.dtype or "")
            item.setText(4, node.nx_class or "")
            item.setData(0, self.qt.Qt.UserRole, node.path)
            items[node.path] = item
        self.tree.expandToDepth(1)

    def _build_detector_tabs(self, detector_paths: list[str]) -> None:
        self._syncing_tabs = True
        try:
            self.tabs.clear()
            self.detector_tabs.clear()
            if self.current_view_file is None:
                return
            definition = self.current_summary.definition if self.current_summary is not None else None
            for dataset_path in detector_paths:
                data = np.asarray(read_nexus_dataset(self.current_view_file, dataset_path))
                if data.ndim != 2 or not (
                    np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_)
                ):
                    continue
                initial_mask = None
                if definition == "SCARLET_masks":
                    initial_mask = np.asarray(data > 0, dtype=np.uint8)
                tab = _DetectorPlotTab(
                    self.qt,
                    self._Plot2D,
                    self._Colormap,
                    dataset_path,
                    data,
                    initial_mask=initial_mask,
                )
                tab.set_colormap(
                    name=self._colormap_name,
                    normalization=self._normalization,
                    reset_zoom=True,
                )
                self.detector_tabs[dataset_path] = tab
                self.tabs.addTab(tab.widget, _detector_tab_label(dataset_path))
        finally:
            self._syncing_tabs = False

    def _on_file_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._files):
            return
        try:
            self.load_file(self._files[row])
        except Exception as exc:
            self.status_bar.showMessage(f"Cannot load {self._files[row].name}: {exc}")
            self.qt.QMessageBox.critical(self.window, "Load error", str(exc))

    def _on_instrument_changed(self, _text: str) -> None:
        if self.current_source_file is None:
            return
        try:
            self.load_file(self.current_source_file)
        except Exception as exc:
            self.status_bar.showMessage(f"Cannot load {self.current_source_file.name}: {exc}")
            self.qt.QMessageBox.critical(self.window, "Conversion error", str(exc))

    def _on_scale_mode_changed(self, _text: str) -> None:
        self._normalization = self._scale_mode()
        for tab in self.detector_tabs.values():
            tab.set_colormap(normalization=self._normalization, reset_zoom=False)

    def _on_tree_selection_changed(self) -> None:
        if self._syncing_tree:
            return
        items = self.tree.selectedItems()
        if not items or self.current_view_file is None:
            return
        dataset_path = items[0].data(0, self.qt.Qt.UserRole)
        if not isinstance(dataset_path, str):
            return
        if dataset_path in self.detector_tabs:
            self._select_dataset(dataset_path)
            return
        try:
            value = read_nexus_dataset(self.current_view_file, dataset_path)
        except Exception as exc:
            self.status_bar.showMessage(str(exc))
            return
        self.preview_label.setText(f"{dataset_path}: {self._format_dataset_value(value)}")

    def _select_dataset(self, dataset_path: str) -> None:
        if dataset_path not in self.detector_tabs:
            return
        tab_index = list(self.detector_tabs).index(dataset_path)
        self._syncing_tabs = True
        try:
            self.tabs.setCurrentIndex(tab_index)
        finally:
            self._syncing_tabs = False
        self.current_dataset_path = dataset_path
        self.preview_label.setText(self._preview_text(dataset_path))
        self._select_tree_item(dataset_path)

    def _on_tab_changed(self, index: int) -> None:
        if self._syncing_tabs or index < 0 or index >= len(self.detector_tabs):
            return
        dataset_path = list(self.detector_tabs)[index]
        self.current_dataset_path = dataset_path
        self.qt.QTimer.singleShot(0, self.detector_tabs[dataset_path].widget.resetZoom)
        self.preview_label.setText(self._preview_text(dataset_path))
        self._select_tree_item(dataset_path)

    def _select_tree_item(self, dataset_path: str) -> None:
        self._syncing_tree = True
        try:
            matches = self.tree.findItems("", self.qt.Qt.MatchContains | self.qt.Qt.MatchRecursive, 0)
            for item in matches:
                if item.data(0, self.qt.Qt.UserRole) == dataset_path:
                    self.tree.setCurrentItem(item)
                    break
        finally:
            self._syncing_tree = False

    def _preview_text(self, dataset_path: str) -> str:
        tab = self.detector_tabs[dataset_path]
        return f"Preview: {_detector_tab_label(dataset_path)} ({tab.data.shape[1]}x{tab.data.shape[0]})"

    def _scale_mode(self) -> str:
        return "log" if self.scale_combo.currentText().lower() == "log" else "linear"

    def _sync_display_preferences_from_tabs(self) -> None:
        if not self.detector_tabs:
            return
        dataset_path = self.current_dataset_path
        if dataset_path not in self.detector_tabs:
            dataset_path = next(iter(self.detector_tabs))
        if dataset_path is None:
            return

        colormap_name, normalization = self.detector_tabs[dataset_path].get_colormap_state()
        self._colormap_name = colormap_name
        self._normalization = "log" if str(normalization).lower() == "log" else "linear"

        desired_text = "Log" if self._normalization == "log" else "Linear"
        if self.scale_combo.currentText() != desired_text:
            self.scale_combo.setCurrentText(desired_text)

    def _clear_view(self) -> None:
        self._sync_display_preferences_from_tabs()
        self.current_source_file = None
        self.current_view_file = None
        self.current_summary = None
        self.current_dataset_path = None
        self.summary_text.clear()
        self.tree.clear()
        self.tabs.clear()
        self.detector_tabs.clear()

    @staticmethod
    def _format_dataset_value(value) -> str:
        array = np.asarray(value)
        if isinstance(value, (bytes, bytearray)):
            return value.decode(errors="replace")
        if array.ndim == 0:
            scalar = array.reshape(()).item()
            if isinstance(scalar, bytes):
                return scalar.decode(errors="replace")
            return str(scalar)
        if array.ndim == 1 and array.size <= 16:
            return np.array2string(array, threshold=16)
        return f"Array shape={array.shape}, dtype={array.dtype}"

    def save_masks_dialog(self) -> None:
        if self.current_source_file is None or self.current_view_file is None:
            self.status_bar.showMessage("No file loaded.")
            return
        if not self.detector_tabs:
            self.status_bar.showMessage("No detector image available for mask export.")
            return
        suggested = default_mask_output_path(self.current_source_file)
        filename, _selected_filter = self.qt.QFileDialog.getSaveFileName(
            self.window,
            "Save mask bundle",
            str(suggested),
            "NeXus/HDF5 (*.nxs *.h5 *.hdf5);;All files (*)",
        )
        if not filename:
            return
        try:
            source = build_mask_source(self.current_source_file, self.current_view_file)
            masks = {
                _detector_index_from_path(dataset_path): tab.get_mask_data()
                for dataset_path, tab in self.detector_tabs.items()
            }
            write_mask_bundle(Path(filename), source, masks, overwrite=True)
        except Exception as exc:
            self.status_bar.showMessage(f"Cannot save masks: {exc}")
            self.qt.QMessageBox.critical(self.window, "Save masks error", str(exc))
            return
        self.status_bar.showMessage(f"Saved masks to {filename}")


def run_viewer(
    initial_directory: Optional[Path | str] = None,
    *,
    instrument: str = "sansllb",
) -> int:
    window = ViewerWindow(
        initial_directory=None if initial_directory is None else Path(initial_directory),
        initial_instrument=instrument,
    )
    return window.run()


def run_scarlet_viewer(
    initial_directory: Optional[Path | str] = None,
    *,
    instrument: str = "sansllb",
) -> int:
    return run_viewer(initial_directory, instrument=instrument)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="viewer", description="SCARLET silx-based file viewer")
    parser.add_argument("directory", nargs="?", help="Optional data folder loaded at startup")
    parser.add_argument(
        "--instrument",
        choices=("sam", "sansllb"),
        default="sansllb",
        help="Instrument used to convert raw files before display (default: sansllb)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_viewer(
            None if args.directory is None else Path(args.directory),
            instrument=args.instrument,
        )
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        print(
            f"Missing dependency: {missing}. Install viewer dependencies: `pip install -e .[viewer]`.",
            file=sys.stderr,
        )
        return 2
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 2


__all__ = [
    "ViewerWindow",
    "build_parser",
    "main",
    "run_viewer",
    "run_scarlet_viewer",
]
