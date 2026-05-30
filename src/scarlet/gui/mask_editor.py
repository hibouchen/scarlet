from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Mapping, Optional

import h5py
import numpy as np

from scarlet.workflow.configuration import Aperture, Collimation, Configuration, configuration_from_nexus


@dataclass(frozen=True)
class MaskEditorSource:
    file_path: Path
    entry_path: str
    detector_data: dict[int, np.ndarray]
    configuration: Configuration
    configuration_issues: list[str]

    @property
    def detector_indices(self) -> list[int]:
        return sorted(self.detector_data)


def _resolve_entry_path(f: h5py.File) -> str:
    for entry_path in ("/raw_data", "/entry", "/entry0", "/entry1"):
        if entry_path in f and isinstance(f[entry_path], h5py.Group):
            return entry_path
    raise ValueError("No raw-data entry group found")


def _list_detector_indices_in_file(file_path: Path | str) -> tuple[str, list[int]]:
    file_path = Path(file_path).resolve()
    with h5py.File(file_path, "r") as f:
        entry_path = _resolve_entry_path(f)
        instrument_path = f"{entry_path}/instrument"
        if instrument_path not in f or not isinstance(f[instrument_path], h5py.Group):
            return entry_path, []

        indices: list[int] = []
        for name in f[instrument_path].keys():
            match = re.fullmatch(r"detector(\d+)", name)
            if match is None:
                continue
            data_path = f"{instrument_path}/{name}/data"
            if data_path in f:
                indices.append(int(match.group(1)))
        return entry_path, sorted(indices)


def load_mask_source(file_path: Path | str) -> MaskEditorSource:
    file_path = Path(file_path).resolve()
    entry_path, detector_indices = _list_detector_indices_in_file(file_path)
    if not detector_indices:
        raise ValueError(f"No detector data found in {file_path}")

    detector_data: dict[int, np.ndarray] = {}
    with h5py.File(file_path, "r") as f:
        for detector_index in detector_indices:
            dataset_path = f"{entry_path}/instrument/detector{detector_index}/data"
            if dataset_path not in f:
                continue
            data = np.asarray(f[dataset_path][()], dtype=np.float64)
            if data.ndim != 2:
                raise ValueError(
                    f"Detector dataset must be 2D for mask editing: {dataset_path} has shape {data.shape}"
                )
            detector_data[detector_index] = data

    configuration, issues = configuration_from_nexus(file_path, entry_path=entry_path)
    return MaskEditorSource(
        file_path=file_path,
        entry_path=entry_path,
        detector_data=detector_data,
        configuration=configuration,
        configuration_issues=issues,
    )


def _write_dataset(parent: h5py.Group, name: str, value) -> h5py.Dataset:
    if isinstance(value, (str, Path)):
        return parent.create_dataset(name, data=np.bytes_(str(value)))
    return parent.create_dataset(name, data=value)


def _write_aperture(parent: h5py.Group, name: str, aperture: Aperture) -> None:
    group = parent.create_group(name)
    if aperture.type == "slit":
        group.attrs["NX_class"] = np.bytes_("NXslit")
        if aperture.x_gap is not None:
            _write_dataset(group, "x_gap", float(aperture.x_gap))
        if aperture.y_gap is not None:
            _write_dataset(group, "y_gap", float(aperture.y_gap))
        return
    if aperture.type == "pinhole":
        group.attrs["NX_class"] = np.bytes_("NXpinhole")
        if aperture.diameter is not None:
            _write_dataset(group, "diameter", float(aperture.diameter))
        return
    raise ValueError(f"Unsupported aperture type: {aperture.type!r}")


def _write_configuration_snapshot(entry: h5py.Group, configuration: Configuration) -> None:
    if configuration.config_id is not None:
        _write_dataset(entry, "config_id", configuration.config_id)

    cfg = entry.create_group("configuration")
    cfg.attrs["NX_class"] = np.bytes_("NXcollection")
    if np.isfinite(configuration.wavelength):
        _write_dataset(cfg, "wavelength", float(configuration.wavelength))
    if isinstance(configuration.sample_detector_distance, list):
        if len(configuration.sample_detector_distance) == 1 and np.isfinite(configuration.sample_detector_distance[0]):
            _write_dataset(cfg, "sample_detector_distance", float(configuration.sample_detector_distance[0]))
    elif np.isfinite(configuration.sample_detector_distance):
        _write_dataset(cfg, "sample_detector_distance", float(configuration.sample_detector_distance))
    if configuration.notes:
        _write_dataset(cfg, "notes", configuration.notes)

    collimation = configuration.collimation
    if collimation is None:
        return
    _write_collimation(cfg, collimation)


def _write_collimation(parent: h5py.Group, collimation: Collimation) -> None:
    col = parent.create_group("collimation")
    col.attrs["NX_class"] = np.bytes_("NXcollection")
    _write_dataset(col, "collimation_distance", float(collimation.collimation_distance))
    _write_dataset(
        col,
        "last_aperture_to_sample_distance",
        float(collimation.last_aperture_to_sample_distance),
    )
    _write_aperture(col, "aperture1", collimation.aperture1)
    _write_aperture(col, "aperture2", collimation.aperture2)


def write_mask_bundle(
    output_path: Path | str,
    source: MaskEditorSource,
    masks: Mapping[int, np.ndarray],
    *,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output file exists: {output_path}")
        output_path.unlink()

    normalized_masks: dict[int, np.ndarray] = {}
    for detector_index, data in source.detector_data.items():
        if detector_index not in masks:
            continue
        mask = np.asarray(masks[detector_index], dtype=np.uint8)
        if mask.shape != data.shape:
            raise ValueError(
                f"Mask shape mismatch for detector{detector_index}: expected {data.shape}, got {mask.shape}"
            )
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"Mask for detector{detector_index} must contain only 0/1 values")
        normalized_masks[detector_index] = mask

    created_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with h5py.File(output_path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        _write_dataset(entry, "definition", "SCARLET_masks")
        _write_dataset(entry, "schema_version", "1.0")
        _write_configuration_snapshot(entry, source.configuration)

        mask_group = entry.create_group("mask")
        mask_group.attrs["NX_class"] = np.bytes_("NXcollection")
        for detector_index, mask in sorted(normalized_masks.items()):
            _write_dataset(mask_group, f"mask_detector{detector_index}", mask)

        meta = entry.create_group("meta")
        meta.attrs["NX_class"] = np.bytes_("NXcollection")
        _write_dataset(meta, "created_utc", created_utc)
        _write_dataset(meta, "mask_convention", "1=masked, 0=valid")
        _write_dataset(meta, "source_file", source.file_path.resolve())
        _write_dataset(meta, "source_entry_path", source.entry_path)
        if source.configuration_issues:
            _write_dataset(meta, "configuration_issues", "\n".join(source.configuration_issues))

    return output_path


def _to_rgb_preview(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    scale: int,
    display_min: float,
    display_max: float,
    log_scale: bool,
) -> np.ndarray:
    if not np.isfinite(display_min):
        display_min = 0.0
    if not np.isfinite(display_max) or display_max <= display_min:
        display_max = display_min + 1.0

    finite = np.isfinite(data)
    if not np.any(finite):
        finite = np.ones_like(data, dtype=bool)

    clipped = np.clip(np.nan_to_num(data, nan=display_min), display_min, display_max)
    if log_scale:
        positive = clipped[finite & (clipped > 0.0)]
        if positive.size:
            log_min = max(display_min, float(np.min(positive)))
            log_max = max(display_max, log_min * (1.0 + 1e-6))
            normalized = (np.log10(np.clip(clipped, log_min, log_max)) - np.log10(log_min)) / (
                np.log10(log_max) - np.log10(log_min)
            )
        else:
            normalized = (clipped - display_min) / (display_max - display_min)
    else:
        normalized = (clipped - display_min) / (display_max - display_min)

    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    rgb = (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)
    rgb[mask.astype(bool)] = np.array([255, 64, 64], dtype=np.uint8)
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return rgb


def _ppm_bytes(rgb: np.ndarray) -> bytes:
    height, width, _channels = rgb.shape
    return f"P6 {width} {height} 255\n".encode("ascii") + rgb.tobytes()


class _MaskEditorApp:
    def __init__(self, *, initial_file: Optional[Path] = None, initial_output: Optional[Path] = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title("SCARLET Mask Editor")
        self.root.geometry("1200x900")

        self.source: Optional[MaskEditorSource] = None
        self.output_path: Optional[Path] = initial_output
        self.masks: dict[int, np.ndarray] = {}
        self.display_limits: dict[int, tuple[float, float]] = {}
        self.current_detector_position = 0
        self.current_scale = 1
        self._updating_display_controls = False
        self._photo = None

        self.status_var = tk.StringVar(value="Load a NeXus file to start.")
        self.detector_var = tk.StringVar(value="")
        self.brush_size_var = tk.IntVar(value=4)
        self.display_min_var = tk.DoubleVar(value=0.0)
        self.display_max_var = tk.DoubleVar(value=1.0)
        self.log_scale_var = tk.BooleanVar(value=False)
        self.scale_mode_var = tk.StringVar(value="Linear")

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(controls, text="Open NeXus", command=self.open_source_dialog).pack(side=tk.LEFT)
        ttk.Button(controls, text="Previous", command=self.previous_detector).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Next", command=self.next_detector).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(controls, text="Clear Detector Mask", command=self.clear_current_mask).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Save Masks", command=self.save_masks_dialog).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(controls, text="Brush").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Scale(
            controls,
            from_=1,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.brush_size_var,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(controls, textvariable=self.scale_mode_var, command=self.toggle_scale_mode).pack(side=tk.LEFT, padx=(8, 0))

        contrast = ttk.Frame(self.root, padding=(8, 0))
        contrast.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(contrast, text="Min").pack(side=tk.LEFT)
        self.min_scale = ttk.Scale(
            contrast,
            orient=tk.HORIZONTAL,
            variable=self.display_min_var,
            command=self._on_min_limit_changed,
        )
        self.min_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 8))
        self.min_value_label = ttk.Label(contrast, text="")
        self.min_value_label.pack(side=tk.LEFT)
        ttk.Label(contrast, text="Max").pack(side=tk.LEFT, padx=(12, 0))
        self.max_scale = ttk.Scale(
            contrast,
            orient=tk.HORIZONTAL,
            variable=self.display_max_var,
            command=self._on_max_limit_changed,
        )
        self.max_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 8))
        self.max_value_label = ttk.Label(contrast, text="")
        self.max_value_label.pack(side=tk.LEFT)

        ttk.Label(self.root, textvariable=self.detector_var, padding=(8, 0)).pack(side=tk.TOP, anchor="w")
        ttk.Label(
            self.root,
            text="Left click/drag: mask, Right click/drag: erase",
            padding=(8, 0),
        ).pack(side=tk.TOP, anchor="w")

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._paint_mask)
        self.canvas.bind("<B1-Motion>", self._paint_mask)
        self.canvas.bind("<Button-3>", self._erase_mask)
        self.canvas.bind("<B3-Motion>", self._erase_mask)

        ttk.Label(self.root, textvariable=self.status_var, padding=8).pack(side=tk.BOTTOM, anchor="w")

        self.root.bind("<Left>", lambda _event: self.previous_detector())
        self.root.bind("<Right>", lambda _event: self.next_detector())
        self.root.bind("<Control-s>", lambda _event: self.save_masks_dialog())

        if initial_file is not None:
            self.load_source(initial_file)

    def run(self) -> None:
        self.root.mainloop()

    def open_source_dialog(self) -> None:
        from tkinter import filedialog, messagebox

        path = filedialog.askopenfilename(
            title="Open NeXus source file",
            filetypes=[("NeXus/HDF5", "*.nxs *.h5 *.hdf *.hdf5"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.load_source(Path(path))
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def load_source(self, file_path: Path) -> None:
        self.source = load_mask_source(file_path)
        self.output_path = self.output_path or file_path.with_name(f"{file_path.stem}_masks.nxs")
        self.masks = {
            detector_index: np.zeros_like(data, dtype=np.uint8)
            for detector_index, data in self.source.detector_data.items()
        }
        self.display_limits = {
            detector_index: self._default_display_limits(data)
            for detector_index, data in self.source.detector_data.items()
        }
        self.current_detector_position = 0
        self.status_var.set(f"Loaded {file_path}")
        self._refresh()

    def _current_detector_index(self) -> int:
        assert self.source is not None
        return self.source.detector_indices[self.current_detector_position]

    def _refresh(self) -> None:
        if self.source is None:
            self.canvas.delete("all")
            self.detector_var.set("")
            return

        detector_index = self._current_detector_index()
        data = self.source.detector_data[detector_index]
        mask = self.masks[detector_index]
        self._sync_display_controls(detector_index, data)
        max_width = max(self.canvas.winfo_width(), 800)
        max_height = max(self.canvas.winfo_height(), 600)
        scale_x = max(1, max_width // data.shape[1])
        scale_y = max(1, max_height // data.shape[0])
        self.current_scale = max(1, min(scale_x, scale_y))

        display_min, display_max = self.display_limits[detector_index]
        rgb = _to_rgb_preview(
            data,
            mask,
            scale=self.current_scale,
            display_min=display_min,
            display_max=display_max,
            log_scale=bool(self.log_scale_var.get()),
        )
        self._photo = self._tk.PhotoImage(data=_ppm_bytes(rgb), format="PPM")
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.config(
            scrollregion=(0, 0, rgb.shape[1], rgb.shape[0]),
            width=rgb.shape[1],
            height=rgb.shape[0],
        )
        self.detector_var.set(
            f"Detector {detector_index} "
            f"({self.current_detector_position + 1}/{len(self.source.detector_indices)}) "
            f"- image {data.shape[1]}x{data.shape[0]}"
        )

    def _default_display_limits(self, data: np.ndarray) -> tuple[float, float]:
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

    def _sync_display_controls(self, detector_index: int, data: np.ndarray) -> None:
        lo, hi = self._default_display_limits(data)
        current_min, current_max = self.display_limits.get(detector_index, (lo, hi))
        current_min = min(max(current_min, lo), hi)
        current_max = min(max(current_max, lo), hi)
        if current_max <= current_min:
            current_max = hi
            if current_max <= current_min:
                current_max = current_min + 1.0
        self.display_limits[detector_index] = (current_min, current_max)

        self._updating_display_controls = True
        try:
            self.min_scale.configure(from_=lo, to=hi)
            self.max_scale.configure(from_=lo, to=hi)
            self.display_min_var.set(current_min)
            self.display_max_var.set(current_max)
            self.min_value_label.config(text=f"{current_min:.6g}")
            self.max_value_label.config(text=f"{current_max:.6g}")
            self.scale_mode_var.set("Log" if self.log_scale_var.get() else "Linear")
        finally:
            self._updating_display_controls = False

    def _apply_display_limit_change(self, *, moved: str) -> None:
        if self.source is None or self._updating_display_controls:
            return
        detector_index = self._current_detector_index()
        data = self.source.detector_data[detector_index]
        lo, hi = self._default_display_limits(data)
        display_min = min(max(float(self.display_min_var.get()), lo), hi)
        display_max = min(max(float(self.display_max_var.get()), lo), hi)
        if display_min >= display_max:
            if moved == "min":
                display_max = min(hi, display_min + max((hi - lo) / 1000.0, 1e-9))
            else:
                display_min = max(lo, display_max - max((hi - lo) / 1000.0, 1e-9))
        if display_min >= display_max:
            display_max = display_min + 1e-9

        self.display_limits[detector_index] = (display_min, display_max)
        self._updating_display_controls = True
        try:
            self.display_min_var.set(display_min)
            self.display_max_var.set(display_max)
            self.min_value_label.config(text=f"{display_min:.6g}")
            self.max_value_label.config(text=f"{display_max:.6g}")
        finally:
            self._updating_display_controls = False
        self._refresh()

    def _on_min_limit_changed(self, _value: str) -> None:
        self._apply_display_limit_change(moved="min")

    def _on_max_limit_changed(self, _value: str) -> None:
        self._apply_display_limit_change(moved="max")

    def toggle_scale_mode(self) -> None:
        self.log_scale_var.set(not bool(self.log_scale_var.get()))
        self.scale_mode_var.set("Log" if self.log_scale_var.get() else "Linear")
        self._refresh()

    def _apply_brush(self, event, value: int) -> None:
        if self.source is None:
            return
        detector_index = self._current_detector_index()
        mask = self.masks[detector_index]
        x = int(event.x // self.current_scale)
        y = int(event.y // self.current_scale)
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
            return
        radius = max(1, int(self.brush_size_var.get()))
        y0 = max(0, y - radius)
        y1 = min(mask.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(mask.shape[1], x + radius + 1)
        ys, xs = np.ogrid[y0:y1, x0:x1]
        disk = (xs - x) ** 2 + (ys - y) ** 2 <= radius ** 2
        region = mask[y0:y1, x0:x1]
        region[disk] = value
        self._refresh()

    def _paint_mask(self, event) -> None:
        self._apply_brush(event, 1)

    def _erase_mask(self, event) -> None:
        self._apply_brush(event, 0)

    def previous_detector(self) -> None:
        if self.source is None or self.current_detector_position == 0:
            return
        self.current_detector_position -= 1
        self._refresh()

    def next_detector(self) -> None:
        from tkinter import messagebox

        if self.source is None:
            return
        if self.current_detector_position + 1 < len(self.source.detector_indices):
            self.current_detector_position += 1
            self._refresh()
            return
        messagebox.showinfo("Mask editor", "Last detector reached. You can save the mask bundle.")

    def clear_current_mask(self) -> None:
        if self.source is None:
            return
        detector_index = self._current_detector_index()
        self.masks[detector_index].fill(0)
        self._refresh()

    def save_masks_dialog(self) -> None:
        from tkinter import filedialog, messagebox

        if self.source is None:
            return
        initial = self.output_path or self.source.file_path.with_name(f"{self.source.file_path.stem}_masks.nxs")
        path = filedialog.asksaveasfilename(
            title="Save mask bundle",
            defaultextension=".nxs",
            initialfile=initial.name,
            initialdir=str(initial.parent),
            filetypes=[("NeXus/HDF5", "*.nxs *.h5 *.hdf5"), ("All files", "*.*")],
        )
        if not path:
            return
        output_path = Path(path)
        try:
            write_mask_bundle(output_path, self.source, self.masks, overwrite=True)
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))
            return
        self.output_path = output_path
        self.status_var.set(f"Saved masks to {output_path}")
        messagebox.showinfo("Mask editor", f"Saved masks to {output_path}")


def run_mask_editor(
    initial_file: Optional[Path | str] = None,
    *,
    output_file: Optional[Path | str] = None,
) -> None:
    app = _MaskEditorApp(
        initial_file=None if initial_file is None else Path(initial_file),
        initial_output=None if output_file is None else Path(output_file),
    )
    app.run()


__all__ = [
    "MaskEditorSource",
    "load_mask_source",
    "run_mask_editor",
    "write_mask_bundle",
]
