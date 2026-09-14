from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Mapping

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
    sample_detector_distance = configuration.sample_detector_distance
    if isinstance(sample_detector_distance, list):
        values = np.asarray(sample_detector_distance, dtype=np.float64)
        if values.size == 1 and np.isfinite(values[0]):
            _write_dataset(cfg, "sample_detector_distance", float(values[0]))
        elif values.size > 1 and np.all(np.isfinite(values)):
            _write_dataset(cfg, "sample_detector_distance", values)
    elif np.isfinite(sample_detector_distance):
        _write_dataset(cfg, "sample_detector_distance", float(sample_detector_distance))
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


__all__ = [
    "MaskEditorSource",
    "load_mask_source",
    "write_mask_bundle",
]
