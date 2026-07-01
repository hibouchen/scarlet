from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from scarlet.io import nexus_reader
from scarlet.reduction.correction import normalize_by_solid_angle, subtract_scattering_references
from scarlet.reduction.transmission import compute_transmission

if TYPE_CHECKING:
    import scipp as sc

    from scarlet.workflow.configuration import Configuration
    from scarlet.workflow.context import WorkflowContext


def _require_scipp():
    """Import Scipp lazily so flatfield metadata can still be handled without it."""
    try:
        import scipp as sc
    except ImportError as exc:
        raise ImportError("scipp is required to load flatfield artifacts as DataArrays") from exc
    return sc


def _select_reference_run(
    workflow: "WorkflowContext",
    *,
    config_id: str,
    entity: str,
    mode: str,
) -> tuple[Any, Path] | None:
    """Return the first matching reference run and warn when duplicates exist."""
    matches = list(workflow.iter_runs(config_id=config_id, entity=entity, mode=mode))
    if not matches:
        return None
    if len(matches) > 1:
        workflow.warn(
            "Multiple reference runs found; using the first one for flatfield preparation",
            where="build_water_flatfield_from_workflow_context",
            key=f"{config_id}:{entity}:{mode}",
            count=len(matches),
        )
    return matches[0]


def _require_reference_run(
    workflow: "WorkflowContext",
    *,
    config_id: str,
    entity: str,
    mode: str,
) -> tuple[Any, Path]:
    selected = _select_reference_run(workflow, config_id=config_id, entity=entity, mode=mode)
    if selected is None:
        raise ValueError(f"Missing {entity} {mode} reference for configuration {config_id!r}")
    return selected


def _compute_or_get_transmission(
    workflow: "WorkflowContext",
    *,
    config_id: str,
    sample_name: str | None,
    transmission_file: Path,
    detector_number: int,
    entity: str,
) -> float:
    if sample_name:
        cached = workflow.get_transmission(sample_name, config_id)
        if cached is not None:
            return float(cached)

    empty_beam_path = workflow.get_empty_beam(config_id, "transmission")
    roi = workflow.get_roi(config_id)
    if empty_beam_path is None or roi is None:
        raise ValueError(
            "Cannot compute transmission without empty-beam transmission and ROI "
            f"for configuration {config_id!r}"
        )

    value = compute_transmission(
        transmission_file,
        empty_beam_path,
        roi,
        detector_number=detector_number,
    )
    if sample_name:
        workflow.set_transmission(sample_name, config_id, value)
    if entity == "empty_cell":
        workflow.set_empty_cell_transmission(config_id, value)
    return value


def _read_detector_beam_center(
    file_path: str | Path,
    *,
    detector_number: int,
) -> tuple[float, float]:
    file_path = Path(file_path).resolve()
    entry_path = nexus_reader.resolve_entry_path(file_path)

    with h5py.File(file_path, "r") as handle:
        detector_root = f"{entry_path}/instrument/detector{detector_number}"
        beam_center_x_path = f"{detector_root}/beam_center_x"
        beam_center_y_path = f"{detector_root}/beam_center_y"
        if beam_center_x_path not in handle or beam_center_y_path not in handle:
            fallback_root = f"{entry_path}/beam_center/detector{detector_number}"
            beam_center_x_path = f"{fallback_root}/beam_center_x"
            beam_center_y_path = f"{fallback_root}/beam_center_y"
        if beam_center_x_path not in handle or beam_center_y_path not in handle:
            raise ValueError(f"Missing beam center for detector{detector_number} in {file_path}")
        return (
            float(np.asarray(handle[beam_center_x_path][()]).reshape(())),
            float(np.asarray(handle[beam_center_y_path][()]).reshape(())),
        )


def _get_workflow_configuration(
    workflow: "WorkflowContext",
    *,
    config_id: str,
    scattering_path: Path,
) -> "Configuration":
    configuration = workflow.configurations.get(config_id)
    if configuration is not None:
        return configuration
    configuration, _issues = nexus_reader.read_configuration(scattering_path)
    workflow.configurations[config_id] = configuration
    return configuration


def _get_detector_distance(configuration: "Configuration", *, detector_number: int) -> float:
    distance = configuration.sample_detector_distance
    if isinstance(distance, list):
        if detector_number >= len(distance):
            raise ValueError(f"Missing sample_detector_distance for detector{detector_number}")
        value = distance[detector_number]
    else:
        value = distance

    distance_value = float(value)
    if not np.isfinite(distance_value) or distance_value <= 0.0:
        raise ValueError(f"Invalid sample_detector_distance for detector{detector_number}: {distance_value!r}")
    return distance_value


def _get_beam_center(
    workflow: "WorkflowContext",
    *,
    config_id: str,
    detector_number: int,
    scattering_path: Path,
) -> tuple[float, float]:
    beam_center = workflow.get_beam_center(config_id, detector_number)
    if beam_center is not None:
        return beam_center
    beam_center = _read_detector_beam_center(scattering_path, detector_number=detector_number)
    workflow.set_beam_center(config_id, detector_number, beam_center)
    return beam_center


def _normalize_detector_by_solid_angle(
    detector: Any,
    *,
    detector_distance: float,
    beam_center: tuple[float, float],
    pixel_size: tuple[float, float],
) -> "sc.DataArray":
    sc = _require_scipp()
    correction = normalize_by_solid_angle(
        np.ones(detector.shape, dtype=np.float64),
        detector_distance=detector_distance,
        beam_center=beam_center,
        pixel_size=pixel_size,
    )
    return detector * sc.array(dims=list(detector.dims), values=np.asarray(correction, dtype=np.float64))


def _extract_flatfield_payload(
    corrected: Any,
    *,
    external_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one corrected water detector into flatfield, errors, and mask arrays."""
    values = np.asarray(corrected.data.values, dtype=np.float64)
    valid_mask = np.isfinite(values) & (values > 0.0)
    if external_mask is not None:
        mask_array = np.asarray(external_mask, dtype=np.uint8)
        if mask_array.shape != values.shape:
            raise ValueError(
                f"Workflow mask shape mismatch: expected {values.shape}, got {mask_array.shape}"
            )
        valid_mask &= mask_array == 0
    if not np.any(valid_mask):
        raise ValueError("Flatfield cannot be constructed because it has no finite positive pixels")

    flatfield = np.ones_like(values, dtype=np.float64)
    flatfield[valid_mask] = values[valid_mask]

    variances = getattr(corrected.data, "variances", None)
    errors = None if variances is None else np.sqrt(np.asarray(variances, dtype=np.float64))
    flatfield_errors = np.zeros(valid_mask.shape, dtype=np.float64)
    if errors is not None:
        flatfield_errors[valid_mask] = errors[valid_mask]
    pixel_mask = (~valid_mask).astype(np.uint8)
    return flatfield, flatfield_errors, pixel_mask


def _copy_optional_scalar(
    source_handle: h5py.File,
    target_group: h5py.Group,
    source_path: str,
    target_name: str,
) -> None:
    if source_path not in source_handle:
        return
    target_group.create_dataset(target_name, data=source_handle[source_path][()])


def _write_text_dataset(group: h5py.Group, name: str, value: str) -> None:
    group.create_dataset(name, data=np.bytes_(value))


def save_flatfield_file(
    flatfields: dict[int, np.ndarray],
    *,
    errors: dict[int, np.ndarray] | None,
    masks: dict[int, np.ndarray] | None,
    file_path: str | Path,
    config_id: str,
    water_scattering_path: Path,
    water_transmission_path: Path,
    dark_path: Path | None,
    empty_cell_path: Path | None,
    mask_file_path: Path | None,
) -> Path:
    """Persist per-detector flatfield arrays to a NeXus/HDF5 artifact."""
    output_path = Path(file_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        _write_text_dataset(entry, "definition", "SCARLET_flatfield")
        _write_text_dataset(entry, "title", f"Water flatfield for {config_id}")
        _write_text_dataset(entry, "config_id", config_id)

        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = np.bytes_("NXinstrument")
        with h5py.File(water_scattering_path, "r") as source_handle:
            entry_path = nexus_reader.resolve_entry_path(water_scattering_path)
            for detector_number, values in sorted(flatfields.items()):
                detector_group = instrument.create_group(f"detector{detector_number}")
                detector_group.attrs["NX_class"] = np.bytes_("NXdetector")
                detector_group.create_dataset("flatfield", data=np.asarray(values, dtype=np.float64))
                detector_group.create_dataset(
                    "flatfield_errors",
                    data=np.asarray((errors or {}).get(detector_number, np.zeros_like(values, dtype=np.float64)), dtype=np.float64),
                )
                detector_group.create_dataset(
                    "pixel_mask",
                    data=np.asarray((masks or {}).get(detector_number, np.zeros_like(values, dtype=np.uint8)), dtype=np.uint8),
                )
                detector_group.create_dataset("flatfield_applied", data=np.bool_(False))

                detector_root = f"{entry_path}/instrument/detector{detector_number}"
                _copy_optional_scalar(source_handle, detector_group, f"{detector_root}/x_pixel_size", "x_pixel_size")
                _copy_optional_scalar(source_handle, detector_group, f"{detector_root}/y_pixel_size", "y_pixel_size")
                _copy_optional_scalar(source_handle, detector_group, f"{detector_root}/beam_center_x", "beam_center_x")
                _copy_optional_scalar(source_handle, detector_group, f"{detector_root}/beam_center_y", "beam_center_y")
                _copy_optional_scalar(source_handle, detector_group, f"{detector_root}/dead_time", "dead_time")
                _copy_optional_scalar(source_handle, detector_group, f"{detector_root}/deadtime", "deadtime")

                data_group = entry.create_group(f"data{detector_number}")
                data_group.attrs["NX_class"] = np.bytes_("NXdata")
                data_group.attrs["signal"] = np.bytes_("flatfield")
                data_group["flatfield"] = h5py.SoftLink(f"/entry/instrument/detector{detector_number}/flatfield")
                data_group["flatfield_errors"] = h5py.SoftLink(
                    f"/entry/instrument/detector{detector_number}/flatfield_errors"
                )
                data_group["pixel_mask"] = h5py.SoftLink(f"/entry/instrument/detector{detector_number}/pixel_mask")

        provenance = entry.create_group("provenance")
        provenance.attrs["NX_class"] = np.bytes_("NXnote")
        _write_text_dataset(provenance, "water_scattering_file", str(water_scattering_path))
        _write_text_dataset(provenance, "water_transmission_file", str(water_transmission_path))
        if dark_path is not None:
            _write_text_dataset(provenance, "dark_file", str(dark_path))
        if empty_cell_path is not None:
            _write_text_dataset(provenance, "empty_cell_scattering_file", str(empty_cell_path))
        if mask_file_path is not None:
            _write_text_dataset(provenance, "mask_file", str(mask_file_path))
    return output_path


def load_flatfield_file(file_path: str | Path) -> dict[int, "sc.DataArray"]:
    """Load per-detector flatfields from a NeXus/HDF5 artifact as Scipp DataArrays."""
    flatfield_path = Path(file_path).resolve()
    sc = _require_scipp()
    with h5py.File(flatfield_path, "r") as handle:
        definition = ""
        if "/entry/definition" in handle:
            raw = np.asarray(handle["/entry/definition"][()]).reshape(()).item()
            if isinstance(raw, (bytes, bytearray)):
                definition = raw.decode(errors="replace")
            else:
                definition = str(raw)
        if definition != "SCARLET_flatfield":
            raise ValueError(f"Unsupported flatfield definition in {flatfield_path}: {definition!r}")

        instrument_path = "/entry/instrument"
        if instrument_path not in handle or not isinstance(handle[instrument_path], h5py.Group):
            raise ValueError(f"Missing instrument group in flatfield file: {flatfield_path}")

        flatfields: dict[int, sc.DataArray] = {}
        instrument = handle[instrument_path]
        for detector_name, detector_group in instrument.items():
            if not isinstance(detector_group, h5py.Group):
                continue
            if not detector_name.startswith("detector") or not detector_name.removeprefix("detector").isdigit():
                continue
            detector_number = int(detector_name.removeprefix("detector"))
            dataset_path = f"{instrument_path}/{detector_name}/flatfield"
            error_path = f"{instrument_path}/{detector_name}/flatfield_errors"
            mask_path = f"{instrument_path}/{detector_name}/pixel_mask"
            if dataset_path not in handle:
                continue
            values = np.asarray(handle[dataset_path][()], dtype=np.float64)
            if values.ndim != 2:
                raise ValueError(
                    f"Flatfield dataset for detector{detector_number} must be 2D, got shape {values.shape}"
                )
            errors = (
                np.asarray(handle[error_path][()], dtype=np.float64)
                if error_path in handle
                else np.zeros_like(values, dtype=np.float64)
            )
            pixel_mask = (
                np.asarray(handle[mask_path][()], dtype=np.uint8)
                if mask_path in handle
                else np.zeros_like(values, dtype=np.uint8)
            )
            if errors.shape != values.shape:
                raise ValueError(
                    f"Flatfield error shape mismatch for detector{detector_number}: "
                    f"expected {values.shape}, got {errors.shape}"
                )
            if pixel_mask.shape != values.shape:
                raise ValueError(
                    f"Flatfield mask shape mismatch for detector{detector_number}: "
                    f"expected {values.shape}, got {pixel_mask.shape}"
                )

            ny, nx = values.shape
            flatfield = sc.DataArray(
                data=sc.array(
                    dims=["y", "x"],
                    values=values,
                    variances=np.square(errors),
                ),
                coords={
                    "y": sc.array(dims=["y"], values=np.arange(ny, dtype=np.float64)),
                    "x": sc.array(dims=["x"], values=np.arange(nx, dtype=np.float64)),
                    "detector_number": sc.scalar(detector_number),
                },
            )
            flatfield.masks["pixel_mask"] = sc.array(
                dims=["y", "x"],
                values=pixel_mask.astype(bool),
            )
            flatfields[detector_number] = flatfield
        return flatfields


def build_water_flatfield_from_workflow_context(
    workflow: "WorkflowContext",
    config_id: str,
    *,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    detector_number_for_transmission: int = 0,
) -> Path:
    """Prepare and persist one water flatfield artifact for a workflow configuration."""
    source_config_id = workflow.resolve_flatfield_config(config_id)
    if output_path is None:
        output_path = workflow.output_dir / f"flatfield_{source_config_id}.nxs"
    output_path = Path(output_path).resolve()

    if output_path.exists() and not overwrite and source_config_id not in workflow.stale_flatfields:
        workflow.set_flatfield(source_config_id, output_path)
        return output_path

    water_scattering_key, water_scattering_path = _require_reference_run(
        workflow,
        config_id=source_config_id,
        entity="water",
        mode="scattering",
    )

    water_scattering = nexus_reader.read_all_detectors(
        water_scattering_path,
        normalize_by_monitor=True,
        correct_deadtime=True,
    )
    dark_path = workflow.get_dark(source_config_id)
    dark = (
        nexus_reader.read_all_detectors(dark_path, normalize_by_monitor=True, correct_deadtime=True)
        if dark_path is not None
        else None
    )
    empty_cell_path = workflow.get_empty_cell(source_config_id, "scattering")
    empty_cell = (
        nexus_reader.read_all_detectors(empty_cell_path, normalize_by_monitor=True, correct_deadtime=True)
        if empty_cell_path is not None
        else None
    )

    water_transmission_path = workflow.get_water(source_config_id, "transmission")
    if water_transmission_path is None:
        selected = _select_reference_run(
            workflow,
            config_id=source_config_id,
            entity="water",
            mode="transmission",
        )
        water_transmission_path = None if selected is None else selected[1]
    if water_transmission_path is None:
        raise ValueError(f"Missing water transmission reference for configuration {source_config_id!r}")

    water_transmission = _compute_or_get_transmission(
        workflow,
        config_id=source_config_id,
        sample_name=water_scattering_key.sample_name,
        transmission_file=water_transmission_path,
        detector_number=detector_number_for_transmission,
        entity="water",
    )

    empty_cell_transmission: float | None = None
    if empty_cell_path is not None:
        empty_cell_transmission = workflow.get_empty_cell_transmission(source_config_id)
        if empty_cell_transmission is None:
            empty_cell_key, empty_cell_transmission_path = _require_reference_run(
                workflow,
                config_id=source_config_id,
                entity="empty_cell",
                mode="transmission",
            )
            empty_cell_transmission = _compute_or_get_transmission(
                workflow,
                config_id=source_config_id,
                sample_name=empty_cell_key.sample_name,
                transmission_file=empty_cell_transmission_path,
                detector_number=detector_number_for_transmission,
                entity="empty_cell",
            )

    flatfields: dict[int, np.ndarray] = {}
    flatfield_errors: dict[int, np.ndarray] = {}
    flatfield_masks: dict[int, np.ndarray] = {}
    invalid_pixels = 0
    configuration = _get_workflow_configuration(
        workflow,
        config_id=source_config_id,
        scattering_path=water_scattering_path,
    )
    for detector_number, water_detector in water_scattering.items():
        dark_detector = None if dark is None else dark.get(detector_number)
        empty_cell_detector = None if empty_cell is None else empty_cell.get(detector_number)
        workflow_mask = workflow.get_mask(source_config_id, detector_number)
        corrected = subtract_scattering_references(
            water_detector,
            water_transmission,
            dark=dark_detector,
            empty_cell=empty_cell_detector,
            empty_cell_transmission=empty_cell_transmission,
        )
        pixel_size = nexus_reader.read_detector_pixel_size(water_scattering_path, detector_number)
        if pixel_size is None:
            raise ValueError(f"pixel_size is required for detector{detector_number} flatfield preparation")
        corrected = _normalize_detector_by_solid_angle(
            corrected,
            detector_distance=_get_detector_distance(configuration, detector_number=detector_number),
            beam_center=_get_beam_center(
                workflow,
                config_id=source_config_id,
                detector_number=detector_number,
                scattering_path=water_scattering_path,
            ),
            pixel_size=pixel_size,
        )
        normalized, normalized_errors, pixel_mask = _extract_flatfield_payload(
            corrected,
            external_mask=workflow_mask,
        )
        replaced = int(np.count_nonzero(pixel_mask))
        flatfields[detector_number] = normalized
        flatfield_errors[detector_number] = normalized_errors
        flatfield_masks[detector_number] = pixel_mask
        invalid_pixels += replaced

    written_path = save_flatfield_file(
        flatfields,
        errors=flatfield_errors,
        masks=flatfield_masks,
        file_path=output_path,
        config_id=source_config_id,
        water_scattering_path=water_scattering_path,
        water_transmission_path=water_transmission_path,
        dark_path=dark_path,
        empty_cell_path=empty_cell_path,
        mask_file_path=workflow.get_mask_file(source_config_id),
    )
    workflow.set_flatfield(source_config_id, written_path)
    workflow.add_artifact(written_path.name, written_path, kind="flatfield")
    if invalid_pixels > 0:
        workflow.warn(
            "Flatfield contains non-positive or non-finite pixels; replaced them with neutral gain",
            where="build_water_flatfield_from_workflow_context",
            key=source_config_id,
            invalid_pixels=invalid_pixels,
        )
    workflow.info(
        "Prepared water flatfield",
        where="build_water_flatfield_from_workflow_context",
        config_id=source_config_id,
        requested_config_id=config_id,
        file_path=str(written_path),
        detectors=sorted(flatfields),
    )
    return written_path
