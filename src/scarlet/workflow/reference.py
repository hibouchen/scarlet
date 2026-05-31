from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping, Optional, Union

import h5py
import numpy as np


def update_detector0_beam_center_from_empty_beam_transmission(
    file_path: Union[str, Path],
    *,
    entry_path: str = "/entry",
) -> Path:
    from scarlet.reduction.transmission import _resolve_entry_path, _read_transmission_roi

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
        if definition != "SCARLET_refs_sub":
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
    with h5py.File(empty_beam_transmission_file, "r") as source_file:
        source_entry_path = _resolve_entry_path(source_file)
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
        if definition != "SCARLET_refs_sub":
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


def compute_corrected_water_scattering(
    ref_norm_file: Union[str, Path],
    *,
    detector_number: int = 0,
    entry_path: str = "/entry",
) -> np.ndarray:
    from scarlet.reduction.transmission import (
        _read_reference_detector_image,
        _read_transmission_roi,
        _read_transmission_roi_detector,
        _roi_sum,
        _require_same_shape,
    )

    ref_norm_file = Path(ref_norm_file).resolve()
    detector_number = int(detector_number)
    if detector_number < 0:
        raise ValueError(f"Detector index must be >= 0, got {detector_number}")

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

        water_scattering = _read_reference_detector_image(
            f,
            "water_scattering",
            detector_number=detector_number,
            refs_entry_path=entry_path,
        )
        if water_scattering is None:
            raise ValueError("refs_norm must contain a water_scattering reference")

        dark = _read_reference_detector_image(
            f,
            "dark",
            detector_number=detector_number,
            refs_entry_path=entry_path,
        )
        if dark is None:
            dark = np.zeros_like(water_scattering, dtype=np.float64)
        _require_same_shape("dark", water_scattering, dark)

        empty_beam_scattering = _read_reference_detector_image(
            f,
            "empty_beam_scattering",
            detector_number=detector_number,
            refs_entry_path=entry_path,
        )
        if empty_beam_scattering is not None:
            _require_same_shape("empty_beam_scattering", water_scattering, empty_beam_scattering)

        empty_cell_scattering = _read_reference_detector_image(
            f,
            "empty_cell_scattering",
            detector_number=detector_number,
            refs_entry_path=entry_path,
        )
        if empty_cell_scattering is not None:
            _require_same_shape("empty_cell_scattering", water_scattering, empty_cell_scattering)

        roi_detector_number = _read_transmission_roi_detector(f, refs_entry_path=entry_path)

        water_transmission = _read_reference_detector_image(
            f,
            "water_transmission",
            detector_number=roi_detector_number,
            refs_entry_path=entry_path,
        )
        if water_transmission is None:
            raise ValueError("refs_norm must contain a water_transmission reference")

        empty_beam_transmission = _read_reference_detector_image(
            f,
            "empty_beam_transmission",
            detector_number=roi_detector_number,
            refs_entry_path=entry_path,
        )
        if empty_beam_transmission is None:
            raise ValueError("refs_norm must contain an empty_beam_transmission reference")

        dark_for_transmission = _read_reference_detector_image(
            f,
            "dark",
            detector_number=roi_detector_number,
            refs_entry_path=entry_path,
        )
        if dark_for_transmission is None:
            dark_for_transmission = np.zeros_like(water_transmission, dtype=np.float64)

        _require_same_shape("dark", water_transmission, dark_for_transmission)
        _require_same_shape("empty_beam_transmission", water_transmission, empty_beam_transmission)

        roi = _read_transmission_roi(f, refs_entry_path=entry_path)
        water_transmission_value = _roi_sum(water_transmission - dark_for_transmission, roi) / _roi_sum(
            empty_beam_transmission - dark_for_transmission,
            roi,
        )

        water_background_corrected = water_scattering - dark
        if empty_beam_scattering is not None:
            water_background_corrected = water_background_corrected - (empty_beam_scattering - dark)

        if empty_cell_scattering is None:
            return water_background_corrected

        empty_cell_corrected = empty_cell_scattering - dark
        if empty_beam_scattering is not None:
            empty_cell_corrected = empty_cell_corrected - (empty_beam_scattering - dark)
        return water_background_corrected - water_transmission_value * empty_cell_corrected


def write_corrected_water_scattering(
    ref_norm_file: Union[str, Path],
    *,
    entry_path: str = "/entry",
) -> dict[int, np.ndarray]:
    from scarlet.reduction import normalize_by_solid_angle

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

        configuration_group = entry.get("configuration")
        if not isinstance(configuration_group, h5py.Group) or "sample_detector_distance" not in configuration_group:
            raise ValueError(f"Missing {entry_path}/configuration/sample_detector_distance")
        detector_distance = float(np.asarray(configuration_group["sample_detector_distance"][()]).reshape(()))

        for detector_number in detector_numbers:
            corrected = compute_corrected_water_scattering(
                ref_norm_file,
                detector_number=detector_number,
                entry_path=entry_path,
            )
            detector_group = water_entry.get(f"instrument/detector{detector_number}")
            if not isinstance(detector_group, h5py.Group):
                raise ValueError(f"Missing water_scattering detector group: detector{detector_number}")
            if "x_pixel_size" not in detector_group or "y_pixel_size" not in detector_group:
                raise ValueError(f"Missing pixel size datasets in water_scattering detector{detector_number}")
            if "beam_center_x" not in detector_group or "beam_center_y" not in detector_group:
                raise ValueError(f"Missing beam center datasets in water_scattering detector{detector_number}")
            pixel_size = (
                float(np.asarray(detector_group["x_pixel_size"][()]).reshape(())),
                float(np.asarray(detector_group["y_pixel_size"][()]).reshape(())),
            )
            beam_center = (
                float(np.asarray(detector_group["beam_center_x"][()]).reshape(())),
                float(np.asarray(detector_group["beam_center_y"][()]).reshape(())),
            )
            corrected_by_detector[detector_number] = normalize_by_solid_angle(
                corrected,
                detector_distance=detector_distance,
                beam_center=beam_center,
                pixel_size=pixel_size,
            )

        if "water_corrected" in references:
            del references["water_corrected"]
        refs_file.copy(f"{entry_path}/references/water_scattering", references, name="water_corrected")

        water_corrected = references["water_corrected"]
        corrected_entry = water_corrected["entry"]
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
