from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import h5py
import numpy as np

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


__all__ = [
    "NexusFileSummary",
    "NexusNodeInfo",
    "PreparedViewFile",
    "format_nexus_summary",
    "list_nexus_files",
    "prepare_view_file",
    "read_nexus_dataset",
    "scan_nexus_file",
]
