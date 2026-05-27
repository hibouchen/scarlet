from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Union

import h5py
import numpy as np


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
    if mask is not None and detector_number is None:
        raise ValueError("detector_number is required when inserting a mask")
    if detector_number is not None:
        detector_number = int(detector_number)
        if detector_number < 0:
            raise ValueError(f"Detector index must be >= 0, got {detector_number}")

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
            assert detector_number is not None
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
