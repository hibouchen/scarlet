from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Literal, Optional, Sequence, Union

import h5py
import numpy as np

from .corrections import ROI, normalize_image

NormalizeBy = Literal["monitor", "count_time", "none"]

RAW_ENTRY_PATH = "/raw_data"
PROCESSED_ENTRY_PATH = "/processed_data"
REFERENCE_BUNDLE_ENTRY_PATH = "/entry"
DEFAULT_ENTRY_CANDIDATES = ("/raw_data", "/entry", "/entry0", "/entry1")


@dataclass(frozen=True)
class Frame:
    """Detector image plus the normalization value used to read it."""

    data: np.ndarray
    normalization: float
    entry_path: str
    detector_path: str


def as_hdf5_path(path: str) -> str:
    """Normalize a user-provided NeXus path to an absolute HDF5 path."""
    path = str(path).strip()
    if not path:
        raise ValueError("HDF5 path must not be empty")
    return path if path.startswith("/") else f"/{path}"


def write_dataset(parent: h5py.Group, name: str, value) -> h5py.Dataset:
    """Create a scalar/string/array dataset with SCARLET's current string convention."""
    if isinstance(value, (str, Path)):
        return parent.create_dataset(name, data=np.bytes_(str(value)))
    return parent.create_dataset(name, data=value)


def scalar(value) -> float:
    """Read a scalar-like HDF5/numpy value as float."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar value, got shape {arr.shape}")
    return float(arr.reshape(()))


def _read_text_scalar(value) -> str:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar text value, got shape {arr.shape}")
    item = arr.reshape(()).item()
    if isinstance(item, (bytes, bytearray)):
        return item.decode(errors="replace")
    return str(item)


def read_optional_scalar(f: h5py.File, path: str) -> Optional[float]:
    """Read an optional scalar dataset from an open HDF5 file."""
    if path not in f:
        return None
    return scalar(f[path][()])


def resolve_entry_path(
    f: h5py.File,
    preferred: Optional[str] = RAW_ENTRY_PATH,
    *,
    candidates: Sequence[str] = DEFAULT_ENTRY_CANDIDATES,
) -> str:
    """
    Resolve the raw-data NXentry path in a NeXus file.

    ``/raw_data`` is the SCARLET default. ``/entry`` and ``/entry0``/``/entry1``
    are kept as compatibility fallbacks for older generated files and tests.
    """
    if preferred is not None:
        preferred = as_hdf5_path(preferred)
        if preferred in f and isinstance(f[preferred], h5py.Group):
            return preferred

    for candidate in candidates:
        candidate = as_hdf5_path(candidate)
        if candidate in f and isinstance(f[candidate], h5py.Group):
            return candidate

    raise ValueError(
        "No raw-data entry group found. Expected one of: "
        + ", ".join(dict.fromkeys([*([] if preferred is None else [preferred]), *candidates]))
    )


def read_normalization(f: h5py.File, entry_path: str, normalize_by: NormalizeBy) -> float:
    """Read the monitor/count-time normalization value for an entry."""
    entry_path = as_hdf5_path(entry_path)
    if normalize_by == "none":
        return 1.0

    if normalize_by == "count_time":
        candidates = (
            f"{entry_path}/control/count_time",
            f"{entry_path}/instrument/monitor0/count_time",
        )
    elif normalize_by == "monitor":
        candidates = (
            f"{entry_path}/control/integral",
            f"{entry_path}/instrument/monitor0/integral",
            f"{entry_path}/control/count_time",
            f"{entry_path}/instrument/monitor0/count_time",
        )
    else:  # pragma: no cover - guarded by typing and argparse choices
        raise ValueError(f"Unsupported normalization mode: {normalize_by!r}")

    for path in candidates:
        value = read_optional_scalar(f, path)
        if value is None:
            continue
        if np.isfinite(value) and value > 0:
            return float(value)

    raise ValueError(f"Cannot find a positive normalization value in {entry_path!r}")


def read_frame_from_entry(
    f: h5py.File,
    entry_path: str,
    *,
    detector_index: int,
    normalize_by: NormalizeBy,
) -> Frame:
    """Read and normalize a detector frame from an already-open file."""
    entry_path = as_hdf5_path(entry_path)
    detector_path = f"{entry_path}/instrument/detector{detector_index}"
    data_path = f"{detector_path}/data"
    if data_path not in f:
        raise ValueError(f"Missing detector data: {data_path}")

    normalization = read_normalization(f, entry_path, normalize_by)
    data = normalize_image(f[data_path][()], normalization)
    return Frame(data=data, normalization=normalization, entry_path=entry_path, detector_path=detector_path)


def read_frame_from_file(
    file_path: Union[str, Path],
    *,
    entry_path: Optional[str] = RAW_ENTRY_PATH,
    detector_index: int,
    normalize_by: NormalizeBy,
) -> Frame:
    """Open a NeXus file and read one detector frame from its raw-data entry."""
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as f:
        resolved_entry = resolve_entry_path(f, entry_path)
        return read_frame_from_entry(
            f,
            resolved_entry,
            detector_index=detector_index,
            normalize_by=normalize_by,
        )


def list_detector_indices_in_entry(
    f: h5py.File,
    entry_path: str,
) -> list[int]:
    """Return the sorted detector indices available under an NXentry."""
    entry_path = as_hdf5_path(entry_path)
    instrument_path = f"{entry_path}/instrument"
    if instrument_path not in f or not isinstance(f[instrument_path], h5py.Group):
        return []

    indices: list[int] = []
    for name in f[instrument_path].keys():
        match = re.fullmatch(r"detector(\d+)", name)
        if match is None:
            continue
        data_path = f"{instrument_path}/{name}/data"
        if data_path in f:
            indices.append(int(match.group(1)))
    return sorted(indices)


def list_detector_indices_in_file(
    file_path: Union[str, Path],
    *,
    entry_path: Optional[str] = RAW_ENTRY_PATH,
) -> tuple[str, list[int]]:
    """Open a NeXus file and return its resolved entry path plus detector indices."""
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as f:
        resolved_entry = resolve_entry_path(f, entry_path)
        return resolved_entry, list_detector_indices_in_entry(f, resolved_entry)


def compute_q_axes(
    f: h5py.File,
    entry_path: str,
    *,
    detector_index: int,
    shape: tuple[int, int],
    beam_center: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D Qx/Qy axes in 1/angstrom for one detector image.

    The axes are derived from the detector geometry stored in the raw entry:
    wavelength, sample-detector distance, pixel sizes, and beam center.
    A separable exact-on-axis formula is used so the image can be displayed
    against 1D Qx and Qy axes.
    """
    entry_path = as_hdf5_path(entry_path)
    detector_path = f"{entry_path}/instrument/detector{detector_index}"
    wavelength_path = f"{entry_path}/instrument/monochromator/wavelength"
    if wavelength_path not in f:
        raise ValueError(f"Missing wavelength dataset: {wavelength_path}")

    wavelength_a = scalar(f[wavelength_path][()])
    if not np.isfinite(wavelength_a) or wavelength_a <= 0.0:
        raise ValueError(f"Invalid wavelength in {wavelength_path}: {wavelength_a!r}")

    required = {}
    for name in ("distance", "x_pixel_size", "y_pixel_size", "beam_center_x", "beam_center_y"):
        path = f"{detector_path}/{name}"
        if path not in f:
            raise ValueError(f"Missing detector geometry dataset: {path}")
        required[name] = scalar(f[path][()])

    distance_m = float(required["distance"])
    x_pixel_size_m = float(required["x_pixel_size"])
    y_pixel_size_m = float(required["y_pixel_size"])
    if beam_center is None:
        beam_center_x = float(required["beam_center_x"])
        beam_center_y = float(required["beam_center_y"])
    else:
        beam_center_x = float(beam_center[0])
        beam_center_y = float(beam_center[1])
    if distance_m <= 0.0:
        raise ValueError(f"Invalid detector distance in {detector_path}/distance: {distance_m!r}")

    ny, nx = shape
    wavevector = (2.0 * np.pi) / wavelength_a

    x_offsets_m = (np.arange(nx, dtype=np.float64) - beam_center_x) * x_pixel_size_m
    y_offsets_m = (np.arange(ny, dtype=np.float64) - beam_center_y) * y_pixel_size_m

    qx = wavevector * x_offsets_m / np.sqrt(distance_m * distance_m + x_offsets_m * x_offsets_m)
    qy = wavevector * y_offsets_m / np.sqrt(distance_m * distance_m + y_offsets_m * y_offsets_m)
    return qx, qy


def reference_entry_path(
    refs_file: h5py.File,
    reference_name: str,
    *,
    refs_entry_path: str = REFERENCE_BUNDLE_ENTRY_PATH,
) -> Optional[str]:
    """Return the copied raw entry for a named reference inside a refs bundle."""
    refs_entry_path = as_hdf5_path(refs_entry_path)
    path = f"{refs_entry_path}/references/{reference_name}/entry"
    if path in refs_file:
        return path
    return None


def read_reference_frame(
    refs_file: h5py.File,
    reference_name: str,
    *,
    detector_index: int,
    normalize_by: NormalizeBy,
    refs_entry_path: str = REFERENCE_BUNDLE_ENTRY_PATH,
) -> Optional[Frame]:
    """Read a named detector reference embedded in a refs bundle."""
    entry_path = reference_entry_path(refs_file, reference_name, refs_entry_path=refs_entry_path)
    if entry_path is None:
        return None
    return read_frame_from_entry(
        refs_file,
        entry_path,
        detector_index=detector_index,
        normalize_by=normalize_by,
    )


def read_transmission_roi(
    refs_file: h5py.File,
    *,
    refs_entry_path: str = REFERENCE_BUNDLE_ENTRY_PATH,
) -> ROI:
    """Read the SCARLET half-open transmission ROI from a refs bundle."""
    refs_entry_path = as_hdf5_path(refs_entry_path)
    base = f"{refs_entry_path}/transmission_roi"
    if base not in refs_file:
        raise ValueError(f"Missing {base} in reference bundle")
    return tuple(
        int(scalar(refs_file[f"{base}/{name}"][()]))
        for name in ("x0", "x1", "y0", "y1")
    )  # type: ignore[return-value]


def read_transmission_roi_detector(
    refs_file: h5py.File,
    *,
    refs_entry_path: str = REFERENCE_BUNDLE_ENTRY_PATH,
) -> int:
    """Read the detector index used for transmission ROI estimation."""
    refs_entry_path = as_hdf5_path(refs_entry_path)
    path = f"{refs_entry_path}/transmission_roi/detector"
    if path not in refs_file:
        raise ValueError(f"Missing {path} in reference bundle")

    try:
        return int(scalar(refs_file[path][()]))
    except Exception:
        text = _read_text_scalar(refs_file[path][()]).strip()
        match = re.fullmatch(r"detector(\d+)", text, flags=re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
        return int(text)


def read_combined_mask(
    refs_file: h5py.File,
    detector_index: int,
    shape: tuple[int, ...],
    *,
    refs_entry_path: str = REFERENCE_BUNDLE_ENTRY_PATH,
) -> Optional[np.ndarray]:
    """Read and OR-combine user and beamstop masks from a refs bundle."""
    refs_entry_path = as_hdf5_path(refs_entry_path)
    mask_base = f"{refs_entry_path}/mask"
    if mask_base not in refs_file:
        return None

    combined = np.zeros(shape, dtype=bool)
    found = False
    for prefix in ("mask", "beamstop_mask"):
        path = f"{mask_base}/{prefix}_detector{detector_index}"
        if path not in refs_file:
            continue
        mask = np.asarray(refs_file[path][()])
        if mask.shape != shape:
            raise ValueError(f"Mask shape mismatch for {path}: expected {shape}, got {mask.shape}")
        combined |= mask.astype(bool)
        found = True
    return combined if found else None


def copy_raw_file_for_processing(
    sample_file: Union[str, Path],
    output_path: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Path:
    """
    Prepare an output NeXus file by copying the raw sample file if needed.

    If ``output_path`` is the same file as ``sample_file``, no copy is made;
    the file will later be opened in-place.
    """
    sample_file = Path(sample_file)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    same_file = False
    try:
        same_file = output_path.resolve() == sample_file.resolve()
    except FileNotFoundError:
        same_file = False

    if same_file:
        if not output_path.exists():
            raise FileNotFoundError(f"Sample file does not exist: {sample_file}")
        return output_path

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output file exists: {output_path}")
        output_path.unlink()

    shutil.copy2(sample_file, output_path)
    return output_path


def create_processed_entry(
    f: h5py.File,
    *,
    processed_entry_path: str = PROCESSED_ENTRY_PATH,
    overwrite: bool = False,
) -> h5py.Group:
    """Create the processed-data NXentry, optionally replacing an existing one."""
    processed_entry_path = as_hdf5_path(processed_entry_path)
    if processed_entry_path in f:
        if not overwrite:
            raise FileExistsError(f"Processed entry already exists: {processed_entry_path}")
        del f[processed_entry_path]

    entry_name = processed_entry_path.strip("/")
    if "/" in entry_name:
        parent_path, name = processed_entry_path.rsplit("/", 1)
        parent = f.require_group(parent_path)
        entry = parent.create_group(name)
    else:
        entry = f.create_group(entry_name)
    entry.attrs["NX_class"] = np.bytes_("NXentry")
    return entry
