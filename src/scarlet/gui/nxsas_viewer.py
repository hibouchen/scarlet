from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
import tempfile
from typing import Optional

import h5py
import numpy as np

from .mask_editor import _ppm_bytes, _to_rgb_preview

_NEXUS_SUFFIXES = {".nxs", ".h5", ".hdf", ".hdf5"}
_IGNORED_DEFINITIONS = {"SCARLET_refs_sub", "SCARLET_refs_norm"}
_DIRECT_VIEW_DEFINITIONS = {"NXsas_raw", "SCARLET_masks"}


@dataclass(frozen=True)
class NexusNodeInfo:
    path: str
    kind: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    nx_class: str | None = None

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class NexusFileSummary:
    file_path: Path
    definition: str | None
    sample_name: str | None
    detector0_distance_m: float | None
    collimation_distance_m: float | None
    wavelength_a: float | None
    entry_paths: list[str]
    detector_paths: list[str]
    image_dataset_paths: list[str]
    nodes: list[NexusNodeInfo]


@dataclass
class _DetectorTabState:
    dataset_path: str
    data: np.ndarray
    frame: object
    canvas: object
    photo: object | None = None


@dataclass(frozen=True)
class PreparedViewFile:
    source_file: Path
    view_file: Path
    converted: bool
    apparatus: str | None = None


def list_nexus_files(directory: Path | str) -> list[Path]:
    directory = Path(directory).resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    files: list[Path] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _NEXUS_SUFFIXES:
            continue
        if _should_ignore_nexus_file(path):
            continue
        files.append(path.resolve())
    return files


def scan_nexus_file(file_path: Path | str) -> NexusFileSummary:
    file_path = Path(file_path).resolve()
    nodes: list[NexusNodeInfo] = []
    entry_paths: list[str] = []
    detector_paths: list[str] = []
    image_dataset_paths: list[str] = []

    with h5py.File(file_path, "r") as handle:
        definition = _read_nexus_definition(handle)
        sample_name = _read_sample_name(handle)

        def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            path = f"/{name}"
            if isinstance(obj, h5py.Group):
                nx_class = _decode_attr(obj.attrs.get("NX_class"))
                nodes.append(NexusNodeInfo(path=path, kind="group", nx_class=nx_class))
                if nx_class == "NXentry" or path in {"/raw_data", "/entry", "/entry0", "/entry1"}:
                    entry_paths.append(path)
                return

            shape = tuple(int(dim) for dim in obj.shape)
            dtype = str(obj.dtype)
            nodes.append(NexusNodeInfo(path=path, kind="dataset", shape=shape, dtype=dtype))
            if _is_displayable_detector_dataset_path(path) and len(shape) == 2:
                detector_paths.append(path)
            if _is_numeric_image_dataset(obj):
                image_dataset_paths.append(path)

        handle.visititems(visit)

    detector0_distance_m, collimation_distance_m, wavelength_a = _read_configuration_summary(
        file_path,
        entry_paths=entry_paths,
    )

    return NexusFileSummary(
        file_path=file_path,
        definition=definition,
        sample_name=sample_name,
        detector0_distance_m=detector0_distance_m,
        collimation_distance_m=collimation_distance_m,
        wavelength_a=wavelength_a,
        entry_paths=sorted(set(entry_paths)),
        detector_paths=sorted(set(detector_paths)),
        image_dataset_paths=sorted(set(image_dataset_paths)),
        nodes=nodes,
    )


def format_nexus_summary(summary: NexusFileSummary) -> str:
    size_kib = summary.file_path.stat().st_size / 1024.0
    lines = [
        summary.file_path.name,
        "" if summary.sample_name is None else summary.sample_name,
        (
            f"distance= {_format_optional_float(summary.detector0_distance_m)} m ; "
            f"collimation= {_format_optional_float(summary.collimation_distance_m)} m ; "
            f"wavelength= {_format_optional_float(summary.wavelength_a)} A"
        ),
        f"Size: {size_kib:.1f} KiB",
        f"Definition: {summary.definition or '-'}",
        f"NXentry groups: {len(summary.entry_paths)}",
        f"Detector images: {len(summary.detector_paths)}",
        f"2D numeric datasets: {len(summary.image_dataset_paths)}",
    ]
    if summary.entry_paths:
        lines.append("Entries: " + ", ".join(summary.entry_paths))
    if summary.detector_paths:
        lines.append("Detectors: " + ", ".join(summary.detector_paths))
    return "\n".join(lines)


def read_nexus_dataset(file_path: Path | str, dataset_path: str):
    file_path = Path(file_path).resolve()
    with h5py.File(file_path, "r") as handle:
        if dataset_path not in handle:
            raise ValueError(f"Dataset not found: {dataset_path}")
        return handle[dataset_path][()]


def prepare_view_file(
    file_path: Path | str,
    *,
    apparatus: str,
    temp_dir: Path | str,
) -> PreparedViewFile:
    from scarlet.io.converters import convert_to_scarlet_nxsas_raw

    source_file = Path(file_path).resolve()
    definition = _read_nexus_definition_from_file(source_file)
    if definition in _DIRECT_VIEW_DEFINITIONS:
        return PreparedViewFile(source_file=source_file, view_file=source_file, converted=False, apparatus=None)

    normalized_apparatus = apparatus.strip().lower()
    if normalized_apparatus not in {"sam", "sansllb"}:
        raise ValueError(f"Unsupported apparatus {apparatus!r}. Expected 'sam' or 'sansllb'.")

    temp_dir = Path(temp_dir).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / f"{source_file.stem}_{normalized_apparatus}_viewer.nxs"
    convert_to_scarlet_nxsas_raw(
        normalized_apparatus,
        source_file,
        output_path,
        overwrite=True,
    )
    return PreparedViewFile(
        source_file=source_file,
        view_file=output_path.resolve(),
        converted=True,
        apparatus=normalized_apparatus,
    )


def _decode_attr(value) -> str | None:
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_attr(value.reshape(()).item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.size == 1:
            return _decode_attr(value.reshape(-1)[0].item())
        decoded_items = [_decode_attr(item.item() if hasattr(item, "item") else item) for item in value.reshape(-1)]
        return ", ".join(item for item in decoded_items if item is not None)
    return str(value)


def _read_nexus_definition(handle: h5py.File) -> str | None:
    for dataset_path in (
        "/entry/definition",
        "/raw_data/definition",
        "/entry0/definition",
        "/entry1/definition",
    ):
        if dataset_path not in handle:
            continue
        try:
            return _decode_attr(handle[dataset_path][()])
        except Exception:
            return None
    return None


def _read_nexus_definition_from_file(file_path: Path | str) -> str | None:
    try:
        with h5py.File(Path(file_path).resolve(), "r") as handle:
            return _read_nexus_definition(handle)
    except OSError:
        return None


def _read_sample_name(handle: h5py.File) -> str | None:
    for dataset_path in (
        "/raw_data/sample/name",
        "/entry/sample/name",
        "/entry0/sample/name",
        "/entry1/sample/name",
    ):
        if dataset_path not in handle:
            continue
        try:
            return _decode_attr(handle[dataset_path][()])
        except Exception:
            return None
    return None


def _read_configuration_summary(
    file_path: Path,
    *,
    entry_paths: list[str],
) -> tuple[float | None, float | None, float | None]:
    from scarlet.workflow.configuration import configuration_from_nexus

    entry_path = entry_paths[0] if entry_paths else "/raw_data"
    try:
        configuration, _issues = configuration_from_nexus(file_path, entry_path=entry_path, detector_index=0)
    except Exception:
        return None, None, None

    detector0_distance_m = _first_distance_value(configuration.sample_detector_distance)
    collimation_distance_m = None
    if configuration.collimation is not None:
        collimation_distance_m = _normalize_optional_float(configuration.collimation.collimation_distance)
    wavelength_a = _normalize_optional_float(configuration.wavelength)
    return detector0_distance_m, collimation_distance_m, wavelength_a


def _first_distance_value(value) -> float | None:
    if isinstance(value, list):
        if not value:
            return None
        return _normalize_optional_float(value[0])
    return _normalize_optional_float(value)


def _normalize_optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{value:.6g}"


def _should_ignore_nexus_file(file_path: Path | str) -> bool:
    try:
        with h5py.File(file_path, "r") as handle:
            definition = _read_nexus_definition(handle)
    except OSError:
        return False
    return definition in _IGNORED_DEFINITIONS


def _is_displayable_detector_dataset_path(path: str) -> bool:
    return (
        re.fullmatch(r"/(?:raw_data|entry|entry0|entry1)/instrument/detector\d+/data", path) is not None
        or re.fullmatch(r"/entry/mask/mask_detector\d+", path) is not None
    )


def _is_numeric_image_dataset(dataset: h5py.Dataset) -> bool:
    return len(dataset.shape) == 2 and (
        np.issubdtype(dataset.dtype, np.number) or np.issubdtype(dataset.dtype, np.bool_)
    )


def _format_dataset_value(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="replace")
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.reshape(()).item()
        if isinstance(scalar, bytes):
            return scalar.decode(errors="replace")
        return str(scalar)
    if array.ndim == 1 and array.size <= 16:
        return np.array2string(array, threshold=16)
    return f"Array shape={array.shape}, dtype={array.dtype}"


def _detector_tab_label(dataset_path: str) -> str:
    match = re.search(r"/(detector\d+)/data$", dataset_path)
    if match is not None:
        return match.group(1)
    match = re.search(r"/mask/(mask_detector\d+)$", dataset_path)
    if match is not None:
        return match.group(1)
    return dataset_path.rsplit("/", 2)[-2]


class _NxsasViewerApp:
    def __init__(self, *, initial_directory: Optional[Path] = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self._ttk = ttk
        self._temp_dir = tempfile.TemporaryDirectory(prefix="scarlet_nxsas_viewer_")
        self.root = tk.Tk()
        self.root.title("SCARLET NXsas Viewer")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.directory: Optional[Path] = None
        self.current_file: Optional[Path] = None
        self.current_source_file: Optional[Path] = None
        self.current_summary: Optional[NexusFileSummary] = None
        self.current_dataset_path: Optional[str] = None
        self.detector_tabs: dict[str, _DetectorTabState] = {}
        self._tab_frames_by_dataset: dict[str, object] = {}
        self._dataset_by_tab_frame: dict[str, str] = {}
        self._syncing_tree_selection = False
        self._syncing_tab_selection = False

        self.status_var = tk.StringVar(value="Select a data folder to browse NXsas files.")
        self.folder_var = tk.StringVar(value="Folder: -")
        self.preview_var = tk.StringVar(value="Select a file to display its detectors or masks.")
        self.instrument_var = tk.StringVar(value="SANSLLB")
        self.log_scale_var = tk.BooleanVar(value=False)
        self.scale_mode_var = tk.StringVar(value="Linear")

        self._build_menu()

        container = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(container, padding=8)
        container.add(left, weight=1)

        right = ttk.Panedwindow(container, orient=tk.VERTICAL)
        container.add(right, weight=3)

        ttk.Label(left, textvariable=self.folder_var, wraplength=320).pack(anchor="w", fill=tk.X)
        ttk.Button(left, text="Open Folder", command=self.select_directory_dialog).pack(anchor="w", pady=(8, 8))
        instrument_frame = ttk.Frame(left)
        instrument_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(instrument_frame, text="Instrument").pack(side=tk.LEFT)
        instrument_selector = ttk.Combobox(
            instrument_frame,
            textvariable=self.instrument_var,
            values=("SAM", "SANSLLB"),
            state="readonly",
            width=10,
        )
        instrument_selector.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        instrument_selector.bind("<<ComboboxSelected>>", self._on_instrument_changed)

        file_frame = ttk.Frame(left)
        file_frame.pack(fill=tk.BOTH, expand=True)
        self.file_list = tk.Listbox(file_frame, exportselection=False)
        file_scroll = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=file_scroll.set)
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_list.bind("<<ListboxSelect>>", self._on_file_selected)

        top = ttk.Frame(right, padding=8)
        right.add(top, weight=2)
        bottom = ttk.Frame(right, padding=8)
        right.add(bottom, weight=3)

        self.summary_text = tk.Text(top, height=7, wrap="word")
        self.summary_text.pack(fill=tk.X)
        self.summary_text.configure(state="disabled")

        tree_frame = ttk.Frame(top)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.tree = ttk.Treeview(
            tree_frame,
            columns=("kind", "shape", "dtype", "nx_class"),
            show="tree headings",
        )
        self.tree.heading("#0", text="Path")
        self.tree.heading("kind", text="Kind")
        self.tree.heading("shape", text="Shape")
        self.tree.heading("dtype", text="Dtype")
        self.tree.heading("nx_class", text="NX_class")
        self.tree.column("#0", width=320, anchor="w")
        self.tree.column("kind", width=90, anchor="w")
        self.tree.column("shape", width=120, anchor="w")
        self.tree.column("dtype", width=120, anchor="w")
        self.tree.column("nx_class", width=120, anchor="w")
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_selected)

        preview_controls = ttk.Frame(bottom)
        preview_controls.pack(fill=tk.X)
        ttk.Label(preview_controls, textvariable=self.preview_var, wraplength=820).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            anchor="w",
        )
        ttk.Button(
            preview_controls,
            textvariable=self.scale_mode_var,
            command=self.toggle_scale_mode,
        ).pack(side=tk.RIGHT, padx=(8, 0))
        self.detector_notebook = ttk.Notebook(bottom)
        self.detector_notebook.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.detector_notebook.bind("<<NotebookTabChanged>>", self._on_detector_tab_changed)

        ttk.Label(self.root, textvariable=self.status_var, padding=8).pack(side=tk.BOTTOM, anchor="w")

        if initial_directory is not None:
            self.set_directory(initial_directory)

    def _build_menu(self) -> None:
        menu = self._tk.Menu(self.root)
        file_menu = self._tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open Data Folder", command=self.select_directory_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self._close)
        menu.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menu)

    def _close(self) -> None:
        self._temp_dir.cleanup()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()

    def select_directory_dialog(self) -> None:
        from tkinter import filedialog, messagebox

        path = filedialog.askdirectory(title="Select NXsas data folder")
        if not path:
            return
        try:
            self.set_directory(Path(path))
        except Exception as exc:
            messagebox.showerror("Folder error", str(exc))

    def set_directory(self, directory: Path | str) -> None:
        directory = Path(directory).resolve()
        files = list_nexus_files(directory)
        self.directory = directory
        self.folder_var.set(f"Folder: {directory}")
        self.file_list.delete(0, self._tk.END)
        for path in files:
            self.file_list.insert(self._tk.END, path.name)
        self._files = files
        self.current_file = None
        self.current_summary = None
        self.current_dataset_path = None
        self._set_summary_text("No file selected.")
        self.tree.delete(*self.tree.get_children())
        self._clear_detector_tabs()
        if files:
            self.file_list.selection_set(0)
            self._on_file_selected()
            self.status_var.set(f"Loaded folder {directory} ({len(files)} file(s)).")
        else:
            self.preview_var.set("No supported NeXus file found in this folder.")
            self.status_var.set(f"No NeXus file found in {directory}.")

    def _on_file_selected(self, _event=None) -> None:
        from tkinter import messagebox

        selection = self.file_list.curselection()
        if not selection:
            return
        file_path = self._files[int(selection[0])]
        try:
            self.load_file(file_path)
        except Exception as exc:
            self.status_var.set(f"Cannot load {Path(file_path).name}: {exc}")
            messagebox.showerror("Load error", str(exc))

    def load_file(self, file_path: Path | str) -> None:
        source_file = Path(file_path).resolve()
        prepared = prepare_view_file(
            source_file,
            apparatus=self.instrument_var.get(),
            temp_dir=Path(self._temp_dir.name),
        )
        summary = scan_nexus_file(prepared.view_file)
        self.current_summary = replace(summary, file_path=prepared.source_file)
        self.current_file = prepared.view_file
        self.current_source_file = prepared.source_file
        self.current_dataset_path = None
        self._set_summary_text(format_nexus_summary(self.current_summary))
        self._populate_tree(self.current_summary)
        self._build_detector_tabs(self.current_summary.detector_paths)
        if self.current_summary.detector_paths:
            self._select_dataset(self.current_summary.detector_paths[0])
        else:
            self.preview_var.set("Selected file has no detector or mask dataset to preview.")
        if prepared.converted:
            self.status_var.set(
                f"Loaded {prepared.source_file.name} via temporary {prepared.apparatus} conversion."
            )
        else:
            self.status_var.set(f"Loaded {prepared.source_file.name}")

    def _on_instrument_changed(self, _event=None) -> None:
        from tkinter import messagebox

        if self.current_source_file is None:
            return
        try:
            self.load_file(self.current_source_file)
        except Exception as exc:
            self.status_var.set(f"Cannot load {self.current_source_file.name}: {exc}")
            messagebox.showerror("Conversion error", str(exc))

    def _populate_tree(self, summary: NexusFileSummary) -> None:
        self.tree.delete(*self.tree.get_children())
        for node in summary.nodes:
            parent = "" if node.path.count("/") == 1 else node.path.rsplit("/", 1)[0]
            self.tree.insert(
                parent,
                "end",
                iid=node.path,
                text=node.name,
                values=(
                    node.kind,
                    "" if node.shape is None else str(node.shape),
                    node.dtype or "",
                    node.nx_class or "",
                ),
            )

    def _on_tree_selected(self, _event=None) -> None:
        if self._syncing_tree_selection:
            return
        selection = self.tree.selection()
        if not selection or self.current_file is None:
            return
        node_path = selection[0]
        if not node_path.startswith("/"):
            return
        if self.tree.set(node_path, "kind") != "dataset":
            return
        if node_path in self.detector_tabs and self._active_detector_dataset_path() == node_path:
            self.current_dataset_path = node_path
            self._update_preview_label(node_path)
            return
        self.current_dataset_path = node_path
        if node_path in self.detector_tabs:
            self._select_dataset(node_path)
            return
        self.preview_var.set(f"{node_path}: {_format_dataset_value(read_nexus_dataset(self.current_file, node_path))}")

    def _select_dataset(self, dataset_path: str) -> None:
        if not self.tree.exists(dataset_path):
            return
        parent = dataset_path
        while parent:
            self.tree.item(parent, open=True)
            if parent.count("/") == 1:
                break
            parent = parent.rsplit("/", 1)[0]
        self._syncing_tree_selection = True
        try:
            self.tree.selection_set(dataset_path)
            self.tree.see(dataset_path)
        finally:
            self._syncing_tree_selection = False
        if dataset_path in self._tab_frames_by_dataset:
            self._syncing_tab_selection = True
            try:
                self.detector_notebook.select(self._tab_frames_by_dataset[dataset_path])
            finally:
                self._syncing_tab_selection = False
            self._update_preview_label(dataset_path)

    def _build_detector_tabs(self, detector_paths: list[str]) -> None:
        self._clear_detector_tabs()
        if self.current_file is None:
            return
        for dataset_path in detector_paths:
            data = np.asarray(read_nexus_dataset(self.current_file, dataset_path))
            if data.ndim != 2 or not (
                np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_)
            ):
                continue
            frame = self._ttk.Frame(self.detector_notebook, padding=4)
            canvas = self._tk.Canvas(frame, bg="black", highlightthickness=0)
            canvas.pack(fill=self._tk.BOTH, expand=True)
            canvas.bind("<Configure>", lambda _event, path=dataset_path: self._render_detector_tab(path))
            self.detector_notebook.add(frame, text=_detector_tab_label(dataset_path))
            state = _DetectorTabState(
                dataset_path=dataset_path,
                data=data.astype(np.float64, copy=False),
                frame=frame,
                canvas=canvas,
            )
            self.detector_tabs[dataset_path] = state
            self._tab_frames_by_dataset[dataset_path] = frame
            self._dataset_by_tab_frame[str(frame)] = dataset_path
        if self.detector_tabs:
            self.root.after_idle(self._render_all_detector_tabs)

    def _render_all_detector_tabs(self) -> None:
        for dataset_path in self.detector_tabs:
            self._render_detector_tab(dataset_path)
        self._update_preview_label(self._active_detector_dataset_path())

    def _render_detector_tab(self, dataset_path: str) -> None:
        state = self.detector_tabs.get(dataset_path)
        if state is None:
            return
        display_min, display_max = _default_display_limits(state.data)
        max_width = max(state.canvas.winfo_width(), 400)
        max_height = max(state.canvas.winfo_height(), 300)
        scale_x = max(1, max_width // state.data.shape[1])
        scale_y = max(1, max_height // state.data.shape[0])
        scale = max(1, min(scale_x, scale_y))
        rgb = _to_rgb_preview(
            state.data,
            np.zeros_like(state.data, dtype=np.uint8),
            scale=scale,
            display_min=display_min,
            display_max=display_max,
            log_scale=bool(self.log_scale_var.get()),
        )
        state.photo = self._tk.PhotoImage(data=_ppm_bytes(rgb), format="PPM")
        state.canvas.delete("all")
        state.canvas.create_image(0, 0, anchor="nw", image=state.photo)
        state.canvas.config(
            scrollregion=(0, 0, rgb.shape[1], rgb.shape[0]),
            width=rgb.shape[1],
            height=rgb.shape[0],
        )

    def _on_detector_tab_changed(self, _event=None) -> None:
        if self._syncing_tab_selection:
            return
        dataset_path = self._active_detector_dataset_path()
        if dataset_path is None:
            return
        if dataset_path == self.current_dataset_path:
            self._update_preview_label(dataset_path)
            return
        self.current_dataset_path = dataset_path
        if self.tree.exists(dataset_path):
            self._syncing_tree_selection = True
            try:
                self.tree.selection_set(dataset_path)
                self.tree.see(dataset_path)
            finally:
                self._syncing_tree_selection = False
        self._update_preview_label(dataset_path)

    def _active_detector_dataset_path(self) -> str | None:
        current_tab = self.detector_notebook.select()
        if not current_tab:
            return None
        return self._dataset_by_tab_frame.get(current_tab)

    def _update_preview_label(self, dataset_path: str | None) -> None:
        if dataset_path is None:
            if self.detector_tabs:
                self.preview_var.set("Select a detector tab to preview.")
            else:
                self.preview_var.set("Selected file has no detector or mask dataset to preview.")
            return
        state = self.detector_tabs.get(dataset_path)
        if state is None:
            self.preview_var.set(dataset_path)
            return
        self.preview_var.set(
            f"Preview: {_detector_tab_label(dataset_path)} ({state.data.shape[1]}x{state.data.shape[0]})"
        )

    def toggle_scale_mode(self) -> None:
        self.log_scale_var.set(not bool(self.log_scale_var.get()))
        self.scale_mode_var.set("Log" if self.log_scale_var.get() else "Linear")
        self._render_all_detector_tabs()

    def _clear_detector_tabs(self) -> None:
        for tab_id in self.detector_notebook.tabs():
            self.detector_notebook.forget(tab_id)
        self.detector_tabs.clear()
        self._tab_frames_by_dataset.clear()
        self._dataset_by_tab_frame.clear()
        self._syncing_tree_selection = False
        self._syncing_tab_selection = False

    def _set_summary_text(self, text: str) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", self._tk.END)
        self.summary_text.insert("1.0", text)
        self.summary_text.configure(state="disabled")


def _default_display_limits(data: np.ndarray) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def run_nxsas_viewer(initial_directory: Optional[Path | str] = None) -> None:
    app = _NxsasViewerApp(
        initial_directory=None if initial_directory is None else Path(initial_directory),
    )
    app.run()


__all__ = [
    "NexusFileSummary",
    "NexusNodeInfo",
    "PreparedViewFile",
    "format_nexus_summary",
    "list_nexus_files",
    "prepare_view_file",
    "read_nexus_dataset",
    "run_nxsas_viewer",
    "scan_nexus_file",
]
