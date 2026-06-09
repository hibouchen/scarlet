from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping, Optional, Union

import h5py
import numpy as np
from scarlet.io.nexus_reader import resolve_entry_path
from scarlet.reduction.correction import normalize_by_solid_angle, subtract_scattering_references


#TODO: check if the omputation of the water_scattering corrected in the ref_norm_file is correct, and if it can be simplified by reusing the existing functions in reduction.correction and reduction.transmission
ROI = tuple[int, int, int, int]


def _read_monitor_value(f: h5py.File, entry_path: str) -> float:
    dataset_path = f"{entry_path}/control/integral"
    if dataset_path not in f:
        raise ValueError(f"Missing monitor integral: {dataset_path}")
    try:
        monitor_value = float(np.asarray(f[dataset_path][()]).reshape(()))
    except Exception as exc:
        raise ValueError(f"Cannot read monitor integral: {dataset_path}") from exc
    if not np.isfinite(monitor_value) or monitor_value <= 0:
        raise ValueError(f"Monitor integral must be > 0: {dataset_path}")
    return monitor_value


def _require_same_shape(label: str, reference: np.ndarray, other: np.ndarray) -> None:
    if reference.shape != other.shape:
        raise ValueError(f"Shape mismatch for {label}: expected {reference.shape}, got {other.shape}")


def _rewrite_internal_soft_links(
    copied_group: h5py.Group,
    *,
    copied_entry_path: str,
    source_entry_path: str,
) -> None:
    for key in list(copied_group.keys()):
        child_path = f"{copied_group.name}/{key}"
        link = copied_group.file.get(child_path, getlink=True)
        if isinstance(link, h5py.SoftLink) and isinstance(link.path, str):
            if link.path == source_entry_path or link.path.startswith(f"{source_entry_path}/"):
                suffix = link.path[len(source_entry_path) :]
                del copied_group[key]
                copied_group[key] = h5py.SoftLink(f"{copied_entry_path}{suffix}")
                continue

        child = copied_group[key]
        if isinstance(child, h5py.Group):
            _rewrite_internal_soft_links(
                child,
                copied_entry_path=copied_entry_path,
                source_entry_path=source_entry_path,
            )


def _read_reference_detector_image(
    refs_file: h5py.File,
    reference_name: str,
    *,
    detector_number: int,
    refs_entry_path: str = "/entry",
) -> Optional[np.ndarray]:
    reference_root = f"{refs_entry_path}/references/{reference_name}/entry"
    if reference_root not in refs_file or not isinstance(refs_file[reference_root], h5py.Group):
        return None

    data_path = f"{reference_root}/instrument/detector{detector_number}/data"
    if data_path not in refs_file:
        return None

    monitor_value = _read_monitor_value(refs_file, reference_root)
    data = np.asarray(refs_file[data_path][()], dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Reference detector data must be 2D, got shape {data.shape}")
    return data / monitor_value


def _read_reference_detector_geometry(
    refs_file: h5py.File,
    reference_name: str,
    *,
    detector_number: int,
    refs_entry_path: str = "/entry",
) -> tuple[tuple[float, float], tuple[float, float]]:
    detector_root = f"{refs_entry_path}/references/{reference_name}/entry/instrument/detector{detector_number}"
    if detector_root not in refs_file or not isinstance(refs_file[detector_root], h5py.Group):
        raise ValueError(f"Missing detector geometry group: {detector_root}")

    def _read_scalar(dataset_path: str) -> float:
        if dataset_path not in refs_file:
            raise ValueError(f"Missing detector geometry field: {dataset_path}")
        try:
            value = float(np.asarray(refs_file[dataset_path][()]).reshape(()))
        except Exception as exc:
            raise ValueError(f"Cannot read detector geometry field: {dataset_path}") from exc
        if not np.isfinite(value):
            raise ValueError(f"Detector geometry field must be finite: {dataset_path}")
        return value

    x_pixel_size = _read_scalar(f"{detector_root}/x_pixel_size")
    y_pixel_size = _read_scalar(f"{detector_root}/y_pixel_size")

    beam_center_x_path = f"{detector_root}/beam_center_x"
    beam_center_y_path = f"{detector_root}/beam_center_y"
    if beam_center_x_path not in refs_file or beam_center_y_path not in refs_file:
        fallback_root = f"{refs_entry_path}/beam_center/detector{detector_number}"
        beam_center_x_path = f"{fallback_root}/beam_center_x"
        beam_center_y_path = f"{fallback_root}/beam_center_y"

    beam_center_x = _read_scalar(beam_center_x_path)
    beam_center_y = _read_scalar(beam_center_y_path)

    return (beam_center_x, beam_center_y), (x_pixel_size, y_pixel_size)


def _read_reference_entry_path(
    refs_file: h5py.File,
    reference_name: str,
    *,
    refs_entry_path: str = "/entry",
) -> Optional[str]:
    entry_path = f"{refs_entry_path}/references/{reference_name}/entry"
    if entry_path in refs_file and isinstance(refs_file[entry_path], h5py.Group):
        return entry_path
    return None


def _read_transmission_roi(refs_file: h5py.File, *, refs_entry_path: str = "/entry") -> ROI:
    roi_root = f"{refs_entry_path}/transmission_roi"
    required_fields = ("x0", "x1", "y0", "y1")
    values: list[int] = []
    for field in required_fields:
        dataset_path = f"{roi_root}/{field}"
        if dataset_path not in refs_file:
            raise ValueError(f"Missing transmission ROI field: {dataset_path}")
        values.append(int(np.asarray(refs_file[dataset_path][()]).reshape(())))
    roi = tuple(values)
    x0, x1, y0, y1 = roi
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid transmission ROI in refs_norm: {roi}")
    return roi


def _read_transmission_roi_detector(refs_file: h5py.File, *, refs_entry_path: str = "/entry") -> int:
    dataset_path = f"{refs_entry_path}/transmission_roi/detector"
    if dataset_path not in refs_file:
        raise ValueError(f"Missing transmission ROI detector: {dataset_path}")

    raw = np.asarray(refs_file[dataset_path][()]).reshape(()).item()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode(errors="replace")
    if isinstance(raw, str):
        text = raw.strip()
        match = re.fullmatch(r"detector(\d+)", text, flags=re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
        return int(text)
    return int(raw)


def _roi_sum(data: np.ndarray, roi: ROI) -> float:
    x0, x1, y0, y1 = roi
    if data.ndim != 2:
        raise ValueError(f"Transmission ROI requires a 2D detector image, got ndim={data.ndim}")
    ny, nx = data.shape
    if not (0 <= x0 < x1 <= nx and 0 <= y0 < y1 <= ny):
        raise ValueError(f"Invalid ROI {roi} for detector image shape {data.shape}")
    value = float(np.nansum(data[y0:y1, x0:x1]))
    if not np.isfinite(value):
        raise ValueError(f"Non-finite ROI sum for ROI={roi}")
    return value


def _write_sample_transmission(entry_group: h5py.Group, value: float) -> None:
    sample_group = entry_group.require_group("sample")
    sample_group.attrs["NX_class"] = np.bytes_("NXsample")
    if "transmission" in sample_group:
        del sample_group["transmission"]
    sample_group.create_dataset("transmission", data=float(value))


def _read_reference_sample_transmission(
    refs_file: h5py.File,
    reference_name: str,
    *,
    refs_entry_path: str = "/entry",
) -> float | None:
    entry_path = _read_reference_entry_path(
        refs_file,
        reference_name,
        refs_entry_path=refs_entry_path,
    )
    if entry_path is None:
        return None

    dataset_path = f"{entry_path}/sample/transmission"
    if dataset_path not in refs_file:
        return None

    try:
        value = float(np.asarray(refs_file[dataset_path][()]).reshape(()))
    except Exception as exc:
        raise ValueError(f"Cannot read sample transmission: {dataset_path}") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"sample transmission must be > 0: {dataset_path}")
    return value


def _compute_reference_transmission_value(
    refs_file: h5py.File,
    transmission_reference_name: str,
    *,
    detector_number: int,
    roi: ROI,
    refs_entry_path: str,
) -> float:
    transmission_image = _read_reference_detector_image(
        refs_file,
        transmission_reference_name,
        detector_number=detector_number,
        refs_entry_path=refs_entry_path,
    )
    if transmission_image is None:
        raise ValueError(f"Missing {transmission_reference_name} reference")

    empty_beam_transmission = _read_reference_detector_image(
        refs_file,
        "empty_beam_transmission",
        detector_number=detector_number,
        refs_entry_path=refs_entry_path,
    )
    if empty_beam_transmission is None:
        raise ValueError("Missing empty_beam_transmission reference")
    _require_same_shape("empty_beam_transmission", transmission_image, empty_beam_transmission)

    empty_beam_roi_sum = _roi_sum(empty_beam_transmission, roi)
    if empty_beam_roi_sum == 0.0:
        raise ValueError("Cannot compute transmission: empty-beam ROI sum is zero")
    return _roi_sum(transmission_image, roi) / empty_beam_roi_sum


def _resolve_reference_transmission_value(
    refs_file: h5py.File,
    *,
    transmission_reference_name: str,
    scattering_reference_name: str | None = None,
    refs_entry_path: str = "/entry",
) -> float:
    transmission_value = _read_reference_sample_transmission(
        refs_file,
        transmission_reference_name,
        refs_entry_path=refs_entry_path,
    )
    scattering_value = None
    if scattering_reference_name is not None:
        scattering_value = _read_reference_sample_transmission(
            refs_file,
            scattering_reference_name,
            refs_entry_path=refs_entry_path,
        )

    if transmission_value is not None and scattering_value is not None:
        if not np.isclose(transmission_value, scattering_value, rtol=1e-7, atol=0.0):
            raise ValueError(
                "Inconsistent stored sample/transmission values between "
                f"{transmission_reference_name} ({transmission_value}) and "
                f"{scattering_reference_name} ({scattering_value})"
            )
        return transmission_value
    if transmission_value is not None:
        return transmission_value
    if scattering_value is not None:
        return scattering_value

    roi = _read_transmission_roi(refs_file, refs_entry_path=refs_entry_path)
    detector_number = _read_transmission_roi_detector(refs_file, refs_entry_path=refs_entry_path)
    return _compute_reference_transmission_value(
        refs_file,
        transmission_reference_name,
        detector_number=detector_number,
        roi=roi,
        refs_entry_path=refs_entry_path,
    )


def update_detector0_beam_center_from_empty_beam_transmission(
    file_path: Union[str, Path],
    *,
    entry_path: str = "/entry",
) -> Path:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Reference bundle not found: {file_path}")

    with h5py.File(file_path, "r") as refs_file:
        if entry_path not in refs_file or not isinstance(refs_file[entry_path], h5py.Group):
            raise ValueError(f"Missing entry group: {entry_path}")
        entry = refs_file[entry_path]
        if "definition" not in entry:
            raise ValueError(f"Missing {entry_path}/definition")
        definition_raw = entry["definition"][()]
        definition = (
            definition_raw.decode(errors="replace")
            if isinstance(definition_raw, (bytes, bytearray))
            else str(definition_raw)
        )
        if definition not in {"SCARLET_refs_sub", "SCARLET_refs_norm"}:
            raise ValueError(f"Unsupported refs bundle definition: {definition!r}")

        source_dataset_path = f"{entry_path}/meta/empty_beam_transmission_source_file"
        if source_dataset_path not in refs_file:
            raise ValueError(f"Missing source file dataset: {source_dataset_path}")
        source_raw = np.asarray(refs_file[source_dataset_path][()]).reshape(()).item()
        if isinstance(source_raw, (bytes, bytearray)):
            source_raw = source_raw.decode(errors="replace")
        empty_beam_transmission_file = Path(str(source_raw)).resolve()
        roi = _read_transmission_roi(refs_file, refs_entry_path=entry_path)

    if not empty_beam_transmission_file.exists():
        raise FileNotFoundError(f"Empty-beam transmission source not found: {empty_beam_transmission_file}")

    x0, x1, y0, y1 = roi
    source_entry_path = resolve_entry_path(empty_beam_transmission_file)
    with h5py.File(empty_beam_transmission_file, "r") as source_file:
        data_path = f"{source_entry_path}/instrument/detector0/data"
        if data_path not in source_file:
            raise ValueError(f"Missing detector0 data in {empty_beam_transmission_file}: {data_path}")
        data = np.asarray(source_file[data_path][()], dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"detector0 data must be 2D, got shape {data.shape}")
        if not (0 <= x0 < x1 <= data.shape[1] and 0 <= y0 < y1 <= data.shape[0]):
            raise ValueError(f"Invalid ROI {roi} for detector0 data shape {data.shape}")

        roi_data = data[y0:y1, x0:x1]
        roi_sum = float(np.sum(roi_data, dtype=np.float64))
        if not np.isfinite(roi_sum) or roi_sum <= 0.0:
            raise ValueError(f"ROI sum must be > 0 to compute beam center, got {roi_sum!r}")

        x_coords = np.arange(x0, x1, dtype=np.float64)
        y_coords = np.arange(y0, y1, dtype=np.float64)
        beam_center_x = float(np.sum(roi_data * x_coords[np.newaxis, :], dtype=np.float64) / roi_sum)
        beam_center_y = float(np.sum(roi_data * y_coords[:, np.newaxis], dtype=np.float64) / roi_sum)

    return insert_beam_centers_in_refs_file(
        file_path,
        0,
        beam_center_x,
        beam_center_y,
        entry_path=entry_path,
    )


def insert_beam_centers_in_refs_file(
    file_path: Union[str, Path],
    detector_number: int,
    beam_center_x: float,
    beam_center_y: float,
    *,
    entry_path: str = "/entry",
) -> Path:
    from . import configuration as _cfg

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Reference bundle not found: {file_path}")
    detector_number = int(detector_number)
    if detector_number < 0:
        raise ValueError(f"Detector index must be >= 0, got {detector_number}")
    beam_center_x = _cfg._require_number("beam_center_x", beam_center_x)
    beam_center_y = _cfg._require_number("beam_center_y", beam_center_y)

    with h5py.File(file_path, "r+") as f:
        if entry_path not in f or not isinstance(f[entry_path], h5py.Group):
            raise ValueError(f"Missing entry group: {entry_path}")
        entry = f[entry_path]

        if "definition" not in entry:
            raise ValueError(f"Missing {entry_path}/definition")
        definition_raw = entry["definition"][()]
        definition = (
            definition_raw.decode(errors="replace")
            if isinstance(definition_raw, (bytes, bytearray))
            else str(definition_raw)
        )
        if definition not in {"SCARLET_refs_sub", "SCARLET_refs_norm"}:
            raise ValueError(f"Unsupported refs bundle definition: {definition!r}")

        beam_center_group = entry.require_group("beam_center")
        beam_center_group.attrs["NX_class"] = np.bytes_("NXcollection")
        detector_group = beam_center_group.require_group(f"detector{detector_number}")
        detector_group.attrs["NX_class"] = np.bytes_("NXcollection")
        _cfg._replace_dataset(detector_group, "beam_center_x", beam_center_x)
        _cfg._replace_dataset(detector_group, "beam_center_y", beam_center_y)

    return file_path


def insert_masks_in_refs_file(
    file_path: Union[str, Path],
    *,
    detector_number: Optional[int] = None,
    mask: Optional[Union[np.ndarray, str, Path]] = None,
    mask_convention: Optional[str] = None,
    entry_path: str = "/entry",
) -> Path:
    from . import configuration as _cfg

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Reference bundle not found: {file_path}")
    if mask is None and mask_convention is None:
        raise ValueError("Provide mask or mask_convention")
    if detector_number is not None:
        detector_number = int(detector_number)
        if detector_number < 0:
            raise ValueError(f"Detector index must be >= 0, got {detector_number}")

    def _read_text_dataset(group: h5py.Group, dataset_path: str) -> Optional[str]:
        if dataset_path not in group:
            return None
        raw = group[dataset_path][()]
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode(errors="replace")
        return str(raw)

    def _load_masks_from_bundle(mask_file: Union[str, Path]) -> tuple[dict[int, np.ndarray], Optional[str]]:
        mask_file = Path(mask_file)
        if not mask_file.exists():
            raise FileNotFoundError(f"Mask bundle not found: {mask_file}")

        with h5py.File(mask_file, "r") as source:
            if "/entry" not in source or not isinstance(source["/entry"], h5py.Group):
                raise ValueError(f"Missing /entry group in mask bundle: {mask_file}")
            entry = source["/entry"]
            definition = _read_text_dataset(entry, "definition")
            if definition != "SCARLET_masks":
                raise ValueError(f"Unsupported mask bundle definition: {definition!r}")
            if "mask" not in entry or not isinstance(entry["mask"], h5py.Group):
                raise ValueError(f"Missing /entry/mask group in mask bundle: {mask_file}")

            loaded_masks: dict[int, np.ndarray] = {}
            for dataset_name, dataset in entry["mask"].items():
                if not isinstance(dataset, h5py.Dataset):
                    continue
                match = re.fullmatch(r"mask_detector(\d+)", dataset_name)
                if match is None:
                    continue
                detector_index = int(match.group(1))
                loaded_masks[detector_index] = _cfg._normalize_mask_array(
                    np.asarray(dataset[()]),
                    label=dataset_name,
                )
            if not loaded_masks:
                raise ValueError(f"No mask_detectorN datasets found in mask bundle: {mask_file}")

            return loaded_masks, _read_text_dataset(entry, "meta/mask_convention")

    with h5py.File(file_path, "r+") as f:
        if entry_path not in f or not isinstance(f[entry_path], h5py.Group):
            raise ValueError(f"Missing entry group: {entry_path}")
        entry = f[entry_path]

        if "definition" not in entry:
            raise ValueError(f"Missing {entry_path}/definition")
        definition_raw = entry["definition"][()]
        definition = (
            definition_raw.decode(errors="replace")
            if isinstance(definition_raw, (bytes, bytearray))
            else str(definition_raw)
        )
        if definition not in {"SCARLET_refs_sub", "SCARLET_refs_norm"}:
            raise ValueError(f"Unsupported refs bundle definition: {definition!r}")

        mask_group = entry.require_group("mask")
        mask_group.attrs["NX_class"] = np.bytes_("NXcollection")

        def write_single_mask(detector_index: int, value: Union[np.ndarray, str, Path]) -> None:
            normalized = _cfg._normalize_mask_array(value, label=f"mask_detector{detector_index}")
            expected_shape = _cfg._reference_detector_shape(entry, detector_index)
            if expected_shape is not None and normalized.shape != expected_shape:
                raise ValueError(
                    f"mask_detector{detector_index} shape mismatch: "
                    f"expected {expected_shape}, got {normalized.shape}"
                )
            _cfg._replace_dataset(mask_group, f"mask_detector{detector_index}", normalized)
            legacy_beamstop_name = f"beamstop_mask_detector{detector_index}"
            if legacy_beamstop_name in mask_group:
                del mask_group[legacy_beamstop_name]

        if mask is not None:
            if isinstance(mask, (str, Path)):
                loaded_masks, loaded_mask_convention = _load_masks_from_bundle(mask)
                if detector_number is None:
                    for loaded_detector_index, loaded_mask in sorted(loaded_masks.items()):
                        write_single_mask(loaded_detector_index, loaded_mask)
                else:
                    if detector_number not in loaded_masks:
                        raise ValueError(f"mask bundle does not contain mask_detector{detector_number}")
                    write_single_mask(detector_number, loaded_masks[detector_number])
                if mask_convention is None:
                    mask_convention = loaded_mask_convention
            else:
                if detector_number is None:
                    raise ValueError("detector_number is required when inserting an array mask")
                write_single_mask(detector_number, mask)

        meta = entry.require_group("meta")
        meta.attrs["NX_class"] = np.bytes_("NXcollection")
        if mask_convention is None:
            mask_convention = (
                entry["meta/mask_convention"][()].decode(errors="replace")
                if "meta/mask_convention" in entry
                else "1=masked, 0=valid"
            )
        _cfg._replace_dataset(meta, "mask_convention", mask_convention)

    return file_path


def compute_reference_transmissions(
    refs_file_path: Union[str, Path],
    *,
    refs_entry_path: str = "/entry",
) -> dict[str, float]:
    """
    Compute and store the transmission of reference measurements embedded in a refs bundle.

    The computed scalar is written in-place to ``/sample/transmission`` inside the
    copied reference entries. For each transmission reference, the same scalar is also
    written to the associated scattering reference when present.
    """
    refs_file_path = Path(refs_file_path).resolve()
    updated: dict[str, float] = {}

    with h5py.File(refs_file_path, "r+") as refs_file:
        definition_path = f"{refs_entry_path}/definition"
        if definition_path not in refs_file:
            raise ValueError(f"Missing refs definition: {definition_path}")
        definition = np.asarray(refs_file[definition_path][()]).reshape(()).item()
        if isinstance(definition, (bytes, bytearray)):
            definition = definition.decode(errors="replace")
        definition = str(definition)
        if definition not in {"SCARLET_refs_sub", "SCARLET_refs_norm"}:
            raise ValueError(f"Unsupported reference bundle definition: {definition!r}")

        roi = _read_transmission_roi(refs_file, refs_entry_path=refs_entry_path)
        detector_number = _read_transmission_roi_detector(refs_file, refs_entry_path=refs_entry_path)

        if _read_reference_entry_path(refs_file, "empty_beam_transmission", refs_entry_path=refs_entry_path) is None:
            raise ValueError("refs bundle must contain empty_beam_transmission to compute reference transmissions")

        reference_specs: list[tuple[str, Optional[str], float]] = [
            ("empty_beam_transmission", "empty_beam_scattering", 1.0),
        ]
        for transmission_name, scattering_name in (
            ("empty_cell_transmission", "empty_cell_scattering"),
            ("water_transmission", "water_scattering"),
        ):
            if _read_reference_entry_path(refs_file, transmission_name, refs_entry_path=refs_entry_path) is None:
                continue
            value = _compute_reference_transmission_value(
                refs_file,
                transmission_name,
                detector_number=detector_number,
                roi=roi,
                refs_entry_path=refs_entry_path,
            )
            reference_specs.append((transmission_name, scattering_name, value))

        for transmission_name, scattering_name, value in reference_specs:
            transmission_entry_path = _read_reference_entry_path(
                refs_file,
                transmission_name,
                refs_entry_path=refs_entry_path,
            )
            if transmission_entry_path is not None:
                _write_sample_transmission(refs_file[transmission_entry_path], value)
                updated[transmission_name] = value

            if scattering_name is None:
                continue
            scattering_entry_path = _read_reference_entry_path(
                refs_file,
                scattering_name,
                refs_entry_path=refs_entry_path,
            )
            if scattering_entry_path is not None:
                _write_sample_transmission(refs_file[scattering_entry_path], value)
                updated[scattering_name] = value

    return updated


def _reference_detector_numbers(
    refs_file: h5py.File,
    *,
    entry_path: str = "/entry",
) -> list[int]:
    references = refs_file.get(f"{entry_path}/references")
    if not isinstance(references, h5py.Group):
        raise ValueError(f"Missing references group: {entry_path}/references")
    water_scattering = references.get("water_scattering")
    if not isinstance(water_scattering, h5py.Group):
        raise ValueError("refs_norm must contain a water_scattering reference")
    water_entry = water_scattering.get("entry")
    if not isinstance(water_entry, h5py.Group):
        raise ValueError("refs_norm water_scattering reference is missing its entry group")
    instrument = water_entry.get("instrument")
    if not isinstance(instrument, h5py.Group):
        raise ValueError("refs_norm water_scattering reference is missing its instrument group")

    detector_numbers: list[int] = []
    for name, group in instrument.items():
        if not isinstance(group, h5py.Group):
            continue
        match = re.fullmatch(r"detector(\d+)", name)
        if match is None:
            continue
        if "data" in group:
            detector_numbers.append(int(match.group(1)))
    detector_numbers.sort()
    if not detector_numbers:
        raise ValueError("refs_norm must contain at least one water_scattering detector image")
    return detector_numbers


def _read_sample_detector_distance(
    refs_file: h5py.File,
    *,
    detector_number: int,
    entry_path: str = "/entry",
) -> float:
    dataset_path = f"{entry_path}/configuration/sample_detector_distance"
    if dataset_path not in refs_file:
        raise ValueError(f"Missing configuration distance: {dataset_path}")

    raw = np.asarray(refs_file[dataset_path][()], dtype=np.float64)
    if raw.ndim == 0:
        value = float(raw.reshape(()))
    elif raw.ndim == 1:
        if detector_number >= raw.size:
            raise ValueError(f"Missing sample_detector_distance for detector{detector_number}: {dataset_path}")
        value = float(raw[detector_number])
    else:
        raise ValueError(f"Invalid sample_detector_distance shape at {dataset_path}: {raw.shape}")

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"sample_detector_distance must be > 0 at {dataset_path}")
    return value


def _compute_corrected_water_scattering_for_detector(
    refs_file: h5py.File,
    *,
    detector_number: int,
    entry_path: str = "/entry",
) -> np.ndarray:
    water_scattering = _read_reference_detector_image(
        refs_file,
        "water_scattering",
        detector_number=detector_number,
        refs_entry_path=entry_path,
    )
    if water_scattering is None:
        raise ValueError("refs_norm must contain a water_scattering reference")

    dark = _read_reference_detector_image(
        refs_file,
        "dark",
        detector_number=detector_number,
        refs_entry_path=entry_path,
    )
    if dark is None:
        dark = np.zeros_like(water_scattering, dtype=np.float64)
    _require_same_shape("dark", water_scattering, dark)

    empty_beam_scattering = _read_reference_detector_image(
        refs_file,
        "empty_beam_scattering",
        detector_number=detector_number,
        refs_entry_path=entry_path,
    )
    if empty_beam_scattering is not None:
        _require_same_shape("empty_beam_scattering", water_scattering, empty_beam_scattering)

    empty_cell_scattering = _read_reference_detector_image(
        refs_file,
        "empty_cell_scattering",
        detector_number=detector_number,
        refs_entry_path=entry_path,
    )
    if empty_cell_scattering is not None:
        _require_same_shape("empty_cell_scattering", water_scattering, empty_cell_scattering)

    water_transmission_value = _resolve_reference_transmission_value(
        refs_file,
        transmission_reference_name="water_transmission",
        scattering_reference_name="water_scattering",
        refs_entry_path=entry_path,
    )
    empty_cell_transmission_value = None
    if empty_cell_scattering is not None:
        empty_cell_transmission_value = _resolve_reference_transmission_value(
            refs_file,
            transmission_reference_name="empty_cell_transmission",
            scattering_reference_name="empty_cell_scattering",
            refs_entry_path=entry_path,
        )

    empty_beam_transmission_value = None
    if empty_beam_scattering is not None:
        empty_beam_transmission_value = _resolve_reference_transmission_value(
            refs_file,
            transmission_reference_name="empty_beam_transmission",
            scattering_reference_name="empty_beam_scattering",
            refs_entry_path=entry_path,
        )

    corrected = subtract_scattering_references(
        water_scattering,
        water_transmission_value,
        dark=dark,
        empty_cell=empty_cell_scattering,
        empty_cell_transmission=empty_cell_transmission_value,
        empty_beam=empty_beam_scattering,
        empty_beam_transmission=empty_beam_transmission_value,
    )
    beam_center, pixel_size = _read_reference_detector_geometry(
        refs_file,
        "water_scattering",
        detector_number=detector_number,
        refs_entry_path=entry_path,
    )
    detector_distance = _read_sample_detector_distance(
        refs_file,
        detector_number=detector_number,
        entry_path=entry_path,
    )
    return normalize_by_solid_angle(
        corrected,
        detector_distance=detector_distance,
        beam_center=beam_center,
        pixel_size=pixel_size,
    )


def compute_corrected_water_scattering(
    ref_norm_file: Union[str, Path],
    *,
    entry_path: str = "/entry",
) -> dict[int, np.ndarray]:
    ref_norm_file = Path(ref_norm_file).resolve()

    with h5py.File(ref_norm_file, "r") as f:
        if entry_path not in f or not isinstance(f[entry_path], h5py.Group):
            raise ValueError(f"Missing entry group: {entry_path}")
        entry = f[entry_path]
        if "definition" not in entry:
            raise ValueError(f"Missing {entry_path}/definition")
        definition_raw = entry["definition"][()]
        definition = (
            definition_raw.decode(errors="replace")
            if isinstance(definition_raw, (bytes, bytearray))
            else str(definition_raw)
        )
        if definition != "SCARLET_refs_norm":
            raise ValueError(f"Unsupported reference bundle definition: {definition!r}")

        detector_numbers = _reference_detector_numbers(f, entry_path=entry_path)
        return {
            detector_number: _compute_corrected_water_scattering_for_detector(
                f,
                detector_number=detector_number,
                entry_path=entry_path,
            )
            for detector_number in detector_numbers
        }


def write_corrected_water_scattering(
    ref_norm_file: Union[str, Path],
    *,
    entry_path: str = "/entry",
) -> dict[int, np.ndarray]:

    ref_norm_file = Path(ref_norm_file).resolve()
    corrected_by_detector: dict[int, np.ndarray] = {}

    with h5py.File(ref_norm_file, "r+") as refs_file:
        if entry_path not in refs_file or not isinstance(refs_file[entry_path], h5py.Group):
            raise ValueError(f"Missing entry group: {entry_path}")
        entry = refs_file[entry_path]

        if "definition" not in entry:
            raise ValueError(f"Missing {entry_path}/definition")
        definition_raw = entry["definition"][()]
        definition = (
            definition_raw.decode(errors="replace")
            if isinstance(definition_raw, (bytes, bytearray))
            else str(definition_raw)
        )
        if definition != "SCARLET_refs_norm":
            raise ValueError(f"Unsupported refs bundle definition: {definition!r}")

        references = entry.get("references")
        if not isinstance(references, h5py.Group):
            raise ValueError(f"Missing references group: {entry_path}/references")

        corrected_by_detector = compute_corrected_water_scattering(
            ref_norm_file,
            entry_path=entry_path,
        )

        if "water_corrected" in references:
            del references["water_corrected"]
        source_entry_path = f"{entry_path}/references/water_scattering/entry"
        refs_file.copy(f"{entry_path}/references/water_scattering", references, name="water_corrected")

        water_corrected = references["water_corrected"]
        corrected_entry = water_corrected["entry"]
        _rewrite_internal_soft_links(
            corrected_entry,
            copied_entry_path=corrected_entry.name,
            source_entry_path=source_entry_path,
        )
        control = corrected_entry.require_group("control")
        if "integral" in control:
            del control["integral"]
        control.create_dataset("integral", data=1.0)

        for detector_number, corrected in corrected_by_detector.items():
            detector_group = corrected_entry[f"instrument/detector{detector_number}"]
            if "data" in detector_group:
                del detector_group["data"]
            detector_group.create_dataset("data", data=np.asarray(corrected, dtype=np.float64))

        water_transmission = references.get("water_transmission")
        if isinstance(water_transmission, h5py.Group):
            transmission_entry = water_transmission.get("entry")
            if isinstance(transmission_entry, h5py.Group) and "sample/transmission" in transmission_entry:
                transmission_value = float(np.asarray(transmission_entry["sample/transmission"][()]).reshape(()))
                sample = corrected_entry.require_group("sample")
                sample.attrs["NX_class"] = np.bytes_("NXsample")
                if "transmission" in sample:
                    del sample["transmission"]
                sample.create_dataset("transmission", data=transmission_value)

    return corrected_by_detector


def write_refs_sub_file(
    file_path: Union[str, Path],
    configuration: "Configuration",
    *,
    empty_beam_transmission: Union[str, Path],
    dark: Optional[Union[str, Path]] = None,
    empty_beam_scattering: Optional[Union[str, Path]] = None,
    empty_cell_transmission: Optional[Union[str, Path]] = None,
    empty_cell_scattering: Optional[Union[str, Path]] = None,
    transmission_roi_detector: Union[int, str] = 0,
    transmission_roi: tuple[int, int, int, int],
    transmission_roi_notes: Optional[str] = None,
    beam_centers: Optional[Mapping[int, tuple[float, float]]] = None,
    masks: Optional[Mapping[int, np.ndarray]] = None,
    attenuation_factor: Optional[float] = None,
    created_utc: Optional[str] = None,
    mask_convention: str = "1=masked, 0=valid",
    scarlet_version: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    from . import configuration as _cfg

    file_path = Path(file_path)
    if file_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output file exists: {file_path}")
        file_path.unlink()

    references = {
        "dark": dark,
        "empty_beam_transmission": empty_beam_transmission,
        "empty_beam_scattering": empty_beam_scattering,
        "empty_cell_transmission": empty_cell_transmission,
        "empty_cell_scattering": empty_cell_scattering,
    }
    resolved_beam_centers = (
        None
        if beam_centers is None
        else {int(k): (float(v[0]), float(v[1])) for k, v in beam_centers.items()}
    )
    if resolved_beam_centers is None:
        resolved_beam_centers = _cfg._read_beam_centers_from_file(Path(empty_beam_transmission))

    with h5py.File(file_path, "w") as f:
        if created_utc is None:
            created_utc = _cfg._file_created_utc(file_path)

        entry = f.create_group("entry")
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        _cfg._write_dataset(entry, "definition", "SCARLET_refs_sub")
        _cfg._write_dataset(entry, "schema_version", "1.0")
        _cfg._write_configuration_group(entry, configuration, output_kind="refs_sub")

        refs = entry.create_group("references")
        refs.attrs["NX_class"] = np.bytes_("NXcollection")
        source_paths: dict[str, Path] = {}
        for name, source_path in references.items():
            if source_path is None:
                continue
            source_paths[name] = _cfg._copy_reference(refs, name, source_path)

        _cfg._write_beam_center_group(entry, beam_centers=resolved_beam_centers)
        _cfg._write_mask_group(entry, masks=masks)
        _cfg._write_transmission_roi_group(
            entry,
            transmission_roi_detector=transmission_roi_detector,
            transmission_roi=transmission_roi,
            transmission_roi_notes=transmission_roi_notes,
        )
        _cfg._write_transmission_setup_group(entry, attenuation_factor=attenuation_factor)
        _cfg._write_meta_group(
            entry,
            source_paths=source_paths,
            created_utc=created_utc,
            mask_convention=mask_convention,
            scarlet_version=scarlet_version,
        )

    return file_path


def write_refs_norm_file(
    file_path: Union[str, Path],
    configuration: "Configuration",
    *,
    water_scattering: Union[str, Path],
    water_transmission: Union[str, Path],
    dark: Optional[Union[str, Path]] = None,
    empty_beam_transmission: Optional[Union[str, Path]] = None,
    empty_beam_scattering: Optional[Union[str, Path]] = None,
    empty_cell_transmission: Optional[Union[str, Path]] = None,
    empty_cell_scattering: Optional[Union[str, Path]] = None,
    water_scattering_source_config_id: Optional[str] = None,
    water_transmission_source_config_id: Optional[str] = None,
    transmission_roi_detector: Union[int, str] = 0,
    transmission_roi: tuple[int, int, int, int],
    transmission_roi_method: Optional[str] = None,
    transmission_roi_notes: Optional[str] = None,
    masks: Optional[Mapping[int, np.ndarray]] = None,
    beamstop_masks: Optional[Mapping[int, np.ndarray]] = None,
    attenuation_factor: Optional[float] = None,
    created_utc: Optional[str] = None,
    mask_convention: str = "1=masked, 0=valid",
    scarlet_version: Optional[str] = None,
    overwrite: bool = True,
) -> Path:
    from . import configuration as _cfg

    file_path = Path(file_path)
    if file_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output file exists: {file_path}")
        file_path.unlink()

    references = {
        "dark": dark,
        "empty_beam_transmission": empty_beam_transmission,
        "empty_beam_scattering": empty_beam_scattering,
        "empty_cell_transmission": empty_cell_transmission,
        "empty_cell_scattering": empty_cell_scattering,
        "water_scattering": water_scattering,
        "water_transmission": water_transmission,
    }

    with h5py.File(file_path, "w") as f:
        if created_utc is None:
            created_utc = _cfg._file_created_utc(file_path)

        entry = f.create_group("entry")
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        _cfg._write_dataset(entry, "definition", "SCARLET_refs_norm")
        _cfg._write_dataset(entry, "schema_version", "1.0")
        _cfg._write_configuration_group(entry, configuration, output_kind="refs_norm")

        refs = entry.create_group("references")
        refs.attrs["NX_class"] = np.bytes_("NXcollection")
        source_paths: dict[str, Path] = {}
        for name, source_path in references.items():
            if source_path is None:
                continue
            source_paths[name] = _cfg._copy_reference(refs, name, source_path)

        if water_scattering_source_config_id is not None:
            _cfg._write_dataset(refs["water_scattering"], "source_config_id", water_scattering_source_config_id)
        if water_transmission_source_config_id is not None:
            _cfg._write_dataset(refs["water_transmission"], "source_config_id", water_transmission_source_config_id)

        _cfg._write_mask_group(entry, masks=masks, beamstop_masks=beamstop_masks)
        _cfg._write_transmission_roi_group(
            entry,
            transmission_roi_detector=transmission_roi_detector,
            transmission_roi=transmission_roi,
            transmission_roi_method=transmission_roi_method,
            transmission_roi_notes=transmission_roi_notes,
        )
        _cfg._write_transmission_setup_group(entry, attenuation_factor=attenuation_factor)
        _cfg._write_meta_group(
            entry,
            source_paths=source_paths,
            created_utc=created_utc,
            mask_convention=mask_convention,
            scarlet_version=scarlet_version,
        )

    return file_path
