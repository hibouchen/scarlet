from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Optional

import h5py
import numpy as np

if TYPE_CHECKING:
    from scarlet.workflow.configuration import Configuration


@dataclass(frozen=True)
class DetectorData:
    detector_number: int
    data: np.ndarray
    error: np.ndarray
    pixel_size: tuple[float, float] | None = None


@dataclass(frozen=True)
class NexusRawData:
    file_path: Path
    entry_path: str
    monitor: float
    count_time: float | None
    detectors: dict[int, DetectorData]
    configuration: Configuration
    configuration_issues: list[str]

    @property
    def detector_numbers(self) -> list[int]:
        return sorted(self.detectors)


def resolve_entry_path(file_path: Path | str) -> str:
    """Return the first supported NXentry path found in a SCARLET-compatible file."""
    file_path = Path(file_path).resolve()
    with h5py.File(file_path, "r") as handle:
        return _resolve_entry_path(handle, file_path=file_path)


def list_detector_numbers(file_path: Path | str) -> list[int]:
    """List detector indices for datasets stored as ``detectorN/data`` in the input file."""
    file_path = Path(file_path).resolve()
    with h5py.File(file_path, "r") as handle:
        entry_path = _resolve_entry_path(handle, file_path=file_path)
        instrument_path = f"{entry_path}/instrument"
        if instrument_path not in handle or not isinstance(handle[instrument_path], h5py.Group):
            raise ValueError(f"Missing instrument group in {file_path}: {instrument_path}")

        detector_numbers: list[int] = []
        for name, group in handle[instrument_path].items():
            if not isinstance(group, h5py.Group):
                continue
            match = re.fullmatch(r"detector(\d+)", name)
            if match is None:
                continue
            if f"{instrument_path}/{name}/data" in handle:
                detector_numbers.append(int(match.group(1)))
        if not detector_numbers:
            raise ValueError(f"No detectorN/data datasets found in {file_path}")
        return sorted(detector_numbers)


def read_monitor_value(file_path: Path | str) -> float:
    """Read the monitor integral stored under ``<entry>/control/integral``."""
    file_path = Path(file_path).resolve()
    with h5py.File(file_path, "r") as handle:
        entry_path = _resolve_entry_path(handle, file_path=file_path)
        return _read_monitor_value(handle, entry_path, file_path=file_path)


def read_count_time_value(file_path: Path | str) -> float | None:
    """Read the optional count time stored under ``<entry>/control/count_time``."""
    file_path = Path(file_path).resolve()
    with h5py.File(file_path, "r") as handle:
        entry_path = _resolve_entry_path(handle, file_path=file_path)
        return _read_optional_scalar(handle, f"{entry_path}/control/count_time")


def read_detector_data(
    file_path: Path | str,
    detector_number: int,
    *,
    normalize_by_monitor: bool = False,
) -> np.ndarray:
    """Read a single detector image and optionally normalize it by the monitor integral."""
    detector = read_detector(file_path, detector_number, normalize_by_monitor=normalize_by_monitor)
    return detector.data


def read_detector_error(
    file_path: Path | str,
    detector_number: int,
    *,
    normalize_by_monitor: bool = False,
) -> np.ndarray:
    """Read the Poisson error estimate associated with one detector image."""
    detector = read_detector(file_path, detector_number, normalize_by_monitor=normalize_by_monitor)
    return detector.error


def read_detector_pixel_size(file_path: Path | str, detector_number: int) -> tuple[float, float] | None:
    """Read detector pixel sizes if both X and Y pixel sizes are present in the file."""
    detector = read_detector(file_path, detector_number)
    return detector.pixel_size


def read_detector_deadtime(file_path: Path | str, detector_number: int) -> float | None:
    """Read the optional detector dead time stored under ``dead_time`` or ``deadtime``."""
    file_path = Path(file_path).resolve()
    with h5py.File(file_path, "r") as handle:
        entry_path = _resolve_entry_path(handle, file_path=file_path)
        return _read_detector_deadtime(handle, entry_path, detector_number, file_path=file_path)


def read_empty_beam_transmission_source_file(file_path: Path | str) -> Path:
    """Read the source-file path of ``empty_beam_transmission`` from a refs bundle."""
    file_path = Path(file_path).resolve()
    with h5py.File(file_path, "r") as handle:
        entry_path = _resolve_entry_path(handle, file_path=file_path)
        return _read_empty_beam_transmission_source_file(handle, entry_path, file_path=file_path)


def get_roi(file_path: Path | str) -> tuple[list[int], int]:
    """Read the transmission ROI and detector index from a refs bundle."""
    file_path = Path(file_path).resolve()
    with h5py.File(file_path, "r") as handle:
        entry_path = _resolve_entry_path(handle, file_path=file_path)
        return _read_transmission_roi(handle, entry_path, file_path=file_path)


def read_detector(
    file_path: Path | str,
    detector_number: int,
    *,
    normalize_by_monitor: bool = False,
) -> DetectorData:
    """Read one detector image together with its error estimate and optional pixel sizes."""
    file_path = Path(file_path).resolve()
    with h5py.File(file_path, "r") as handle:
        entry_path = _resolve_entry_path(handle, file_path=file_path)
        monitor = _read_monitor_value(handle, entry_path, file_path=file_path)
        return _read_detector(handle, entry_path, detector_number, monitor=monitor, normalize_by_monitor=normalize_by_monitor)


def read_all_detectors(
    file_path: Path | str,
    *,
    normalize_by_monitor: bool = False,
) -> dict[int, DetectorData]:
    """Read every detector image found in the file and return them keyed by detector number."""
    file_path = Path(file_path).resolve()
    with h5py.File(file_path, "r") as handle:
        entry_path = _resolve_entry_path(handle, file_path=file_path)
        monitor = _read_monitor_value(handle, entry_path, file_path=file_path)
        detector_numbers = list_detector_numbers(file_path)
        return {
            detector_number: _read_detector(
                handle,
                entry_path,
                detector_number,
                monitor=monitor,
                normalize_by_monitor=normalize_by_monitor,
            )
            for detector_number in detector_numbers
        }


def read_configuration(file_path: Path | str) -> tuple[Configuration, list[str]]:
    """Extract the experimental configuration inferred from a SCARLET-compatible NeXus file."""
    from scarlet.workflow.configuration import configuration_from_nexus

    file_path = Path(file_path).resolve()
    entry_path = resolve_entry_path(file_path)
    return configuration_from_nexus(file_path, entry_path=entry_path)


def load_nxsas_raw(
    file_path: Path | str,
    *,
    normalize_by_monitor: bool = False,
) -> NexusRawData:
    """Load a full SCARLET NXsas_raw file into a structured in-memory representation."""
    from scarlet.workflow.configuration import configuration_from_nexus

    file_path = Path(file_path).resolve()
    with h5py.File(file_path, "r") as handle:
        entry_path = _resolve_entry_path(handle, file_path=file_path)
        monitor = _read_monitor_value(handle, entry_path, file_path=file_path)
        count_time = _read_optional_scalar(handle, f"{entry_path}/control/count_time")
        detector_numbers = _list_detector_numbers_from_handle(handle, entry_path, file_path=file_path)
        detectors = {
            detector_number: _read_detector(
                handle,
                entry_path,
                detector_number,
                monitor=monitor,
                normalize_by_monitor=normalize_by_monitor,
            )
            for detector_number in detector_numbers
        }

    configuration, issues = configuration_from_nexus(file_path, entry_path=entry_path)
    return NexusRawData(
        file_path=file_path,
        entry_path=entry_path,
        monitor=monitor,
        count_time=count_time,
        detectors=detectors,
        configuration=configuration,
        configuration_issues=issues,
    )


def _resolve_entry_path(handle: h5py.File, *, file_path: Path) -> str:
    """Resolve the preferred entry path among the supported NXentry locations."""
    for entry_path in ("/raw_data", "/entry", "/entry0", "/entry1"):
        if entry_path in handle and isinstance(handle[entry_path], h5py.Group):
            return entry_path
    raise ValueError(f"No raw-data entry group found in {file_path}")


def _read_monitor_value(handle: h5py.File, entry_path: str, *, file_path: Path) -> float:
    """Read and validate the monitor integral from an already opened HDF5 handle."""
    dataset_path = f"{entry_path}/control/integral"
    if dataset_path not in handle:
        raise ValueError(f"Missing monitor integral in {file_path}: {dataset_path}")
    monitor = float(np.asarray(handle[dataset_path][()]).reshape(()))
    if not np.isfinite(monitor) or monitor <= 0.0:
        raise ValueError(f"Monitor integral must be > 0 in {file_path}: {dataset_path}")
    return monitor


def _read_optional_scalar(handle: h5py.File, dataset_path: str) -> float | None:
    """Read an optional scalar dataset and return ``None`` when missing or non-finite."""
    if dataset_path not in handle:
        return None
    value = float(np.asarray(handle[dataset_path][()]).reshape(()))
    return value if np.isfinite(value) else None


def _read_detector_deadtime(
    handle: h5py.File,
    entry_path: str,
    detector_number: int,
    *,
    file_path: Path,
) -> float | None:
    """Read and validate the optional dead time for a detector from an open file handle."""
    detector_path = f"{entry_path}/instrument/detector{detector_number}"
    if detector_path not in handle or not isinstance(handle[detector_path], h5py.Group):
        raise ValueError(f"Missing detector group in {file_path}: {detector_path}")
    for dataset_name in ("dead_time", "deadtime"):
        dataset_path = f"{detector_path}/{dataset_name}"
        if dataset_path not in handle:
            continue
        value = float(np.asarray(handle[dataset_path][()]).reshape(()))
        if not np.isfinite(value):
            return None
        if value < 0.0:
            raise ValueError(f"dead_time must be >= 0 in {file_path}: {dataset_path}")
        return value
    return None


def _read_empty_beam_transmission_source_file(
    handle: h5py.File,
    entry_path: str,
    *,
    file_path: Path,
) -> Path:
    """Read the stored empty-beam transmission source path from a refs bundle."""
    _require_refs_bundle_definition(handle, entry_path, file_path=file_path)

    dataset_path = f"{entry_path}/meta/empty_beam_transmission_source_file"
    source_file = _read_required_text(handle, dataset_path, file_path=file_path)
    return Path(source_file).resolve()


def _read_transmission_roi(
    handle: h5py.File,
    entry_path: str,
    *,
    file_path: Path,
) -> tuple[list[int], int]:
    """Read and validate the transmission ROI and detector stored in a refs bundle."""
    _require_refs_bundle_definition(handle, entry_path, file_path=file_path)

    roi_root = f"{entry_path}/transmission_roi"
    roi = [
        _read_required_int(handle, f"{roi_root}/{field}", file_path=file_path)
        for field in ("x0", "x1", "y0", "y1")
    ]
    x0, x1, y0, y1 = roi
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid transmission ROI in {file_path}: {tuple(roi)}")

    detector = _read_required_detector_number(handle, f"{roi_root}/detector", file_path=file_path)
    return roi, detector


def _list_detector_numbers_from_handle(
    handle: h5py.File,
    entry_path: str,
    *,
    file_path: Path,
) -> list[int]:
    """List detector indices from an open handle without reopening the file."""
    instrument_path = f"{entry_path}/instrument"
    if instrument_path not in handle or not isinstance(handle[instrument_path], h5py.Group):
        raise ValueError(f"Missing instrument group in {file_path}: {instrument_path}")

    detector_numbers: list[int] = []
    for name, group in handle[instrument_path].items():
        if not isinstance(group, h5py.Group):
            continue
        match = re.fullmatch(r"detector(\d+)", name)
        if match is None:
            continue
        if f"{instrument_path}/{name}/data" in handle:
            detector_numbers.append(int(match.group(1)))
    if not detector_numbers:
        raise ValueError(f"No detectorN/data datasets found in {file_path}")
    return sorted(detector_numbers)


def _read_detector(
    handle: h5py.File,
    entry_path: str,
    detector_number: int,
    *,
    monitor: float,
    normalize_by_monitor: bool,
) -> DetectorData:
    """Read one detector dataset from an open file handle and derive its default error model."""
    detector_path = f"{entry_path}/instrument/detector{detector_number}"
    data_path = f"{detector_path}/data"
    if data_path not in handle:
        raise ValueError(f"Missing detector data: {data_path}")

    raw_data = np.asarray(handle[data_path][()], dtype=np.float64)
    if raw_data.ndim != 2:
        raise ValueError(f"Detector data must be 2D at {data_path}, got shape {raw_data.shape}")

    data = raw_data / monitor if normalize_by_monitor else raw_data
    error = np.sqrt(np.clip(raw_data, 0.0, None))
    if normalize_by_monitor:
        error = error / monitor

    x_pixel_size = _read_optional_scalar(handle, f"{detector_path}/x_pixel_size")
    y_pixel_size = _read_optional_scalar(handle, f"{detector_path}/y_pixel_size")
    pixel_size = None if x_pixel_size is None or y_pixel_size is None else (x_pixel_size, y_pixel_size)

    return DetectorData(
        detector_number=int(detector_number),
        data=data,
        error=error,
        pixel_size=pixel_size,
    )


def _read_required_text(handle: h5py.File, dataset_path: str, *, file_path: Path) -> str:
    """Read a required scalar text dataset from an open HDF5 handle."""
    if dataset_path not in handle:
        raise ValueError(f"Missing dataset in {file_path}: {dataset_path}")
    raw_value = np.asarray(handle[dataset_path][()]).reshape(()).item()
    if isinstance(raw_value, (bytes, bytearray)):
        raw_value = raw_value.decode(errors="replace")
    text = str(raw_value).strip()
    if not text:
        raise ValueError(f"Empty text dataset in {file_path}: {dataset_path}")
    return text


def _read_required_int(handle: h5py.File, dataset_path: str, *, file_path: Path) -> int:
    """Read a required scalar integer dataset from an open HDF5 handle."""
    if dataset_path not in handle:
        raise ValueError(f"Missing dataset in {file_path}: {dataset_path}")
    return int(np.asarray(handle[dataset_path][()]).reshape(()))


def _read_required_detector_number(handle: h5py.File, dataset_path: str, *, file_path: Path) -> int:
    """Read a required detector index stored either as an integer or ``detectorN`` text."""
    if dataset_path not in handle:
        raise ValueError(f"Missing dataset in {file_path}: {dataset_path}")

    raw_value = np.asarray(handle[dataset_path][()]).reshape(()).item()
    if isinstance(raw_value, (bytes, bytearray)):
        raw_value = raw_value.decode(errors="replace")
    if isinstance(raw_value, str):
        match = re.fullmatch(r"detector(\d+)", raw_value.strip(), flags=re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
    return int(raw_value)


def _require_refs_bundle_definition(handle: h5py.File, entry_path: str, *, file_path: Path) -> None:
    """Validate that the opened file is a supported SCARLET refs bundle."""
    definition = _read_required_text(handle, f"{entry_path}/definition", file_path=file_path)
    if definition not in {"SCARLET_refs_sub", "SCARLET_refs_norm"}:
        raise ValueError(f"Unsupported refs bundle definition: {definition!r}")


__all__ = [
    "DetectorData",
    "NexusRawData",
    "get_roi",
    "list_detector_numbers",
    "load_nxsas_raw",
    "read_all_detectors",
    "read_configuration",
    "read_count_time_value",
    "read_empty_beam_transmission_source_file",
    "read_detector",
    "read_detector_data",
    "read_detector_deadtime",
    "read_detector_error",
    "read_detector_pixel_size",
    "read_monitor_value",
    "resolve_entry_path",
]
