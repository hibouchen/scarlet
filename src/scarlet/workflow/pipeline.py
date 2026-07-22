from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import h5py
import numpy as np

from scarlet.io import nexus_reader
from scarlet.io.nexus_reader import read_configuration, resolve_entry_path
from scarlet.reduction.correction import normalize_by_solid_angle, subtract_scattering_references
from scarlet.reduction.geometry import compute_q_norm_map
from scarlet.reduction.integration import azimuthal_average
from scarlet.reduction.resolution import compute_beam_divergence, compute_q_resolution_circular, compute_q_resolution_rectangular
from scarlet.workflow.configuration import Configuration
from scarlet.workflow.context import RunKey, WorkflowContext
from scarlet.workflow.normalization import load_flatfield_file

if TYPE_CHECKING:
    import scipp as sc

StepFunction = Callable[["ReductionState"], "ReductionState"]
_F = TypeVar("_F", bound=StepFunction)


@dataclass
class ReductionState:
    sample_name: str
    config_id: str
    workflow: WorkflowContext
    transmission: float = field(default_factory=float)
    azimuthal_n_bins: int = 200
    azimuthal_q_scale: str = "linear"
    detectors: dict[int, Any] = field(default_factory=dict)
    reductions_steps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    file_path: str = field(default_factory=str)

    def __post_init__(self):
        file_path = self.workflow.get_run_path(
            RunKey(
                config_id=self.config_id,
                entity="sample",
                mode="scattering",
                sample_name=self.sample_name,
            )
        )
        if file_path:
            self.detectors = nexus_reader.read_all_detectors(
                file_path,
                normalize_by_monitor=True,
                correct_deadtime=True,
            )
            self.file_path = str(file_path)
            self.notes.append("Loaded data corrected by the deadtime and the monitor")
        if not self.transmission:
            self.transmission = self.workflow.get_transmission(sample_name=self.sample_name, config_id=self.config_id)

def _require_scipp():
    try:
        import scipp as sc
    except ImportError as exc:
        raise ImportError("scipp is required to use workflow pipeline steps on detector DataArrays") from exc
    return sc


def _require_dataarray(value: Any, *, name: str, ndim: int | None = None) -> "sc.DataArray":
    sc = _require_scipp()
    if not isinstance(value, sc.DataArray):
        raise TypeError(f"{name} must be a scipp.DataArray, got {type(value).__name__}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D DataArray, got shape {value.shape}")
    return value


def _read_state_configuration(state: ReductionState) -> Configuration:
    configuration = state.workflow.configurations.get(state.config_id)
    if isinstance(configuration, Configuration):
        return configuration
    if not state.file_path:
        raise ValueError(f"Missing sample scattering file for config {state.config_id!r}")
    configuration, _issues = read_configuration(state.file_path)
    state.workflow.configurations[state.config_id] = configuration
    return configuration


def _get_detector_distance(configuration: Configuration, *, detector_number: int) -> float:
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


def _get_wavelength(configuration: Configuration) -> float:
    wavelength = float(configuration.wavelength)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError(f"Invalid wavelength in configuration {configuration.config_id!r}: {wavelength!r}")
    return wavelength


def _spread_to_wavelength_uncertainty(wavelength: float, spread: float) -> float:
    spread_value = abs(float(spread))
    if spread_value < 1.0:
        return wavelength * spread_value
    if spread_value <= 100.0:
        return wavelength * (spread_value / 100.0)
    return spread_value


def _read_wavelength_uncertainty(
    file_path: str | Path,
    *,
    wavelength: float,
) -> float:
    file_path = Path(file_path).resolve()
    entry_path = resolve_entry_path(file_path)

    with h5py.File(file_path, "r") as handle:
        for dataset_path in (
            f"{entry_path}/instrument/monochromator/wavelength_error",
            f"{entry_path}/instrument/source/incident_wavelength_spread",
            f"{entry_path}/instrument/velocity_selector/wavelength_spread",
        ):
            if dataset_path not in handle:
                continue
            raw_value = float(np.asarray(handle[dataset_path][()]).reshape(()))
            if not np.isfinite(raw_value):
                continue
            if dataset_path.endswith("wavelength_error"):
                return abs(raw_value)
            return _spread_to_wavelength_uncertainty(wavelength, raw_value)
    return 0.0


def _aperture_opening(aperture: Any) -> tuple[float, float] | None:
    aperture_type = getattr(aperture, "type", None)
    if aperture_type == "slit":
        x_gap = getattr(aperture, "x_gap", None)
        y_gap = getattr(aperture, "y_gap", None)
        if x_gap is None or y_gap is None:
            return None
        return float(x_gap), float(y_gap)
    if aperture_type == "pinhole":
        diameter = getattr(aperture, "diameter", None)
        if diameter is None:
            return None
        diameter_value = float(diameter)
        return diameter_value, diameter_value
    return None

def _aperture_type(aperture: Any) -> str | None:
    aperture_type = getattr(aperture, "type", None)
    if aperture_type == "slit":
        return "slit"
    if aperture_type == "pinhole":
        return "pinhole"
    return None


def _get_beam_divergence(configuration: Configuration) -> float:
    collimation = configuration.collimation
    if collimation is None:
        return 0.0

    entry_slit_size = _aperture_opening(collimation.aperture1)
    exit_slit_size = _aperture_opening(collimation.aperture2)
    if entry_slit_size is None or exit_slit_size is None:
        return 0.0

    sigma_div_x, sigma_div_y = compute_beam_divergence(
        entry_slit_size=entry_slit_size,
        exit_slit_size=exit_slit_size,
        collimation_distance=float(collimation.collimation_distance),
    )
    return max(abs(float(sigma_div_x)), abs(float(sigma_div_y)))


def _read_detector_beam_center(
    file_path: str | Path,
    *,
    detector_number: int,
) -> tuple[float, float]:
    file_path = Path(file_path).resolve()
    entry_path = resolve_entry_path(file_path)

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


def _get_beam_center(state: ReductionState, *, detector_number: int) -> tuple[float, float]:
    beam_center = state.workflow.get_beam_center(state.config_id, detector_number)
    if beam_center is not None:
        return beam_center
    if not state.file_path:
        raise ValueError(f"Missing sample scattering file for detector{detector_number} beam center lookup")
    beam_center = _read_detector_beam_center(state.file_path, detector_number=detector_number)
    state.workflow.set_beam_center(state.config_id, detector_number, beam_center)
    return beam_center


def _get_detector_pixel_size(state: ReductionState, *, detector_number: int) -> tuple[float, float]:
    if not state.file_path:
        raise ValueError(f"Missing sample scattering file for detector{detector_number} pixel-size lookup")
    pixel_size = nexus_reader.read_detector_pixel_size(state.file_path, detector_number)
    if pixel_size is None:
        raise ValueError(f"pixel_size is required for detector{detector_number} azimuthal averaging")
    return pixel_size


def _apply_workflow_mask(
    detector: "sc.DataArray",
    *,
    mask: np.ndarray | None,
    name: str = "workflow_config",
) -> "sc.DataArray":
    """Attach one workflow-level boolean mask to a detector DataArray."""
    if mask is None:
        return detector

    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != detector.shape:
        raise ValueError(f"workflow mask shape mismatch: expected {detector.shape}, got {mask_array.shape}")

    sc = _require_scipp()
    masked = detector.copy(deep=False)
    mask_variable = sc.array(dims=list(detector.dims), values=mask_array)
    if name in masked.masks:
        existing = np.asarray(masked.masks[name].values, dtype=bool)
        mask_variable = sc.array(dims=list(detector.dims), values=np.logical_or(existing, mask_array))
    masked.masks[name] = mask_variable
    return masked


def _merge_dataarray_masks(
    target: "sc.DataArray",
    *sources: "sc.DataArray",
) -> "sc.DataArray":
    """Ensure the output carries all masks from the provided source DataArrays."""
    sc = _require_scipp()
    merged = target.copy(deep=False)
    for source in sources:
        for mask_name, mask_variable in source.masks.items():
            if mask_variable.dims != merged.dims or mask_variable.shape != merged.shape:
                raise ValueError(
                    f"mask {mask_name!r} shape mismatch: expected {merged.shape}, got {mask_variable.shape}"
                )
            source_mask = np.asarray(mask_variable.values, dtype=bool)
            if mask_name in merged.masks:
                existing_mask = np.asarray(merged.masks[mask_name].values, dtype=bool)
                source_mask = np.logical_or(existing_mask, source_mask)
            merged.masks[mask_name] = sc.array(dims=list(merged.dims), values=source_mask)
    return merged


def _normalize_detector_by_solid_angle(
    detector: "sc.DataArray",
    *,
    detector_distance: float,
    beam_center: tuple[float, float],
    pixel_size: tuple[float, float],
) -> "sc.DataArray":
    """Apply the standard solid-angle correction while preserving DataArray metadata."""
    sc = _require_scipp()
    correction = normalize_by_solid_angle(
        np.ones(detector.shape, dtype=np.float64),
        detector_distance=detector_distance,
        beam_center=beam_center,
        pixel_size=pixel_size,
    )
    return detector * sc.array(dims=list(detector.dims), values=np.asarray(correction, dtype=np.float64))


def _write_text_dataset(parent: h5py.Group, name: str, value: str) -> None:
    parent.create_dataset(name, data=np.bytes_(value))


def _write_text_list_dataset(parent: h5py.Group, name: str, values: list[str]) -> None:
    parent.create_dataset(name, data=np.asarray([np.bytes_(value) for value in values], dtype="S"))


def _replace_group(parent: h5py.Group | h5py.File, name: str, *, nx_class: str | None = None) -> h5py.Group:
    if name in parent:
        del parent[name]
    group = parent.create_group(name)
    if nx_class is not None:
        group.attrs["NX_class"] = np.bytes_(nx_class)
    return group


def _normalize_processed_entry_name(entry_name: str) -> str:
    normalized = str(entry_name).strip().strip("/")
    if not normalized:
        raise ValueError("processed entry name must not be empty")
    if "/" in normalized:
        raise ValueError(f"processed entry must be a top-level NXentry name, got {entry_name!r}")
    return normalized


def _sanitize_output_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "unknown"


def _build_azimuthal_text_output_path(
    *,
    output_dir: Path,
    sample_name: str,
    config_id: str,
    detector_number: int | None = None,
    suffix: str = ".txt",
) -> Path:
    stem = f"{_sanitize_output_token(sample_name)}_config_{_sanitize_output_token(config_id)}"
    if detector_number is not None:
        stem = f"{stem}_detector{int(detector_number)}"
    return (output_dir / f"{stem}{suffix}").resolve()


def write_azimuthal_text_file(
    file_path: str | Path,
    *,
    q: Any,
    intensity: Any,
    intensity_error: Any | None,
    q_error: Any | None,
    sample_name: str,
    config_id: str,
    transmission: float | None,
    source_nexus_file: str | Path | None = None,
) -> Path:
    """Write one azimuthal average curve to a 4-column text file."""
    output_path = Path(file_path).resolve()
    q_values = np.asarray(q, dtype=np.float64)
    intensity_values = np.asarray(intensity, dtype=np.float64)
    if q_values.shape != intensity_values.shape:
        raise ValueError(f"q/intensity shape mismatch: {q_values.shape} != {intensity_values.shape}")

    if intensity_error is None:
        intensity_error_values = np.full(q_values.shape, np.nan, dtype=np.float64)
    else:
        intensity_error_values = np.asarray(intensity_error, dtype=np.float64)
        if intensity_error_values.shape != q_values.shape:
            raise ValueError(f"intensity_error shape mismatch: expected {q_values.shape}, got {intensity_error_values.shape}")

    if q_error is None:
        q_error_values = np.full(q_values.shape, np.nan, dtype=np.float64)
    else:
        q_error_values = np.asarray(q_error, dtype=np.float64)
        if q_error_values.shape != q_values.shape:
            raise ValueError(f"q_error shape mismatch: expected {q_values.shape}, got {q_error_values.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.column_stack((q_values, intensity_values, intensity_error_values, q_error_values))
    transmission_text = "nan" if transmission is None else f"{float(transmission):.6g}"
    source_nexus_file_text = "unknown" if source_nexus_file is None else Path(source_nexus_file).name
    header = "\n".join(
        (
            f"sample_name: {sample_name}",
            f"config_id: {config_id}",
            f"transmission: {transmission_text}",
            f"source_nexus_file: {source_nexus_file_text}",
            "q I I_error q_error",
        )
    )
    np.savetxt(output_path, matrix, header=header)
    return output_path


def write_azimuthal_average_text_outputs(
    state: "ReductionState",
    *,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Write one 4-column text file per detector from azimuthally averaged data."""
    target_dir = state.workflow.output_dir if output_dir is None else Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    detector_numbers = sorted(state.detectors)
    if not detector_numbers:
        raise ValueError("No detector data available to write azimuthal average text output")

    output_paths: list[Path] = []
    multiple_detectors = len(detector_numbers) > 1
    for detector_number in detector_numbers:
        detector = _require_dataarray(state.detectors[detector_number], name=f"detector{detector_number}", ndim=1)
        if "q" not in detector.coords:
            raise ValueError(f"Missing q coordinate for detector{detector_number} azimuthal output")
        q_coord = detector.coords["q"]
        q_error_coord = detector.coords.get("q_error")
        q_error = None if q_error_coord is None else np.asarray(q_error_coord.values, dtype=np.float64)
        file_path = _build_azimuthal_text_output_path(
            output_dir=target_dir,
            sample_name=state.sample_name,
            config_id=state.config_id,
            detector_number=detector_number if multiple_detectors else None,
        )
        output_paths.append(
            write_azimuthal_text_file(
                file_path,
                q=q_coord.values,
                intensity=detector.data.values,
                intensity_error=None if detector.data.variances is None else np.sqrt(np.asarray(detector.data.variances, dtype=np.float64)),
                q_error=q_error,
                sample_name=state.sample_name,
                config_id=state.config_id,
                transmission=state.transmission,
                source_nexus_file=state.file_path,
            )
        )
    return output_paths


def _write_dataarray_dataset(parent: h5py.Group, name: str, detector: Any) -> None:
    data_array = _require_dataarray(detector, name=name)
    group = _replace_group(parent, name, nx_class="NXdata")
    group.attrs["signal"] = np.bytes_("data")
    group.attrs["dims"] = np.asarray([np.bytes_(dim) for dim in data_array.dims], dtype="S")

    axis_names = [dim for dim in data_array.dims if dim in data_array.coords]
    if axis_names:
        group.attrs["axes"] = np.asarray([np.bytes_(dim) for dim in axis_names], dtype="S")

    group.create_dataset("data", data=np.asarray(data_array.data.values))
    if data_array.data.variances is not None:
        errors = np.sqrt(np.clip(np.asarray(data_array.data.variances, dtype=np.float64), 0.0, None))
        group.create_dataset("errors", data=errors)

    for coord_name, coord_value in data_array.coords.items():
        group.create_dataset(coord_name, data=np.asarray(coord_value.values))

    if data_array.masks:
        masks_group = group.create_group("masks")
        masks_group.attrs["NX_class"] = np.bytes_("NXcollection")
        masks_group.attrs["convention"] = np.bytes_("1=masked, 0=valid")
        for mask_name, mask_value in data_array.masks.items():
            masks_group.create_dataset(mask_name, data=np.asarray(mask_value.values, dtype=np.uint8))


def _write_nxdata_view(
    entry: h5py.Group,
    *,
    detector_number: int,
    detector_group_name: str,
    detector: Any,
) -> None:
    data_array = _require_dataarray(detector, name=detector_group_name)
    view_group = _replace_group(entry, f"data{detector_number}", nx_class="NXdata")
    view_group.attrs["signal"] = np.bytes_("data")

    axis_names = [dim for dim in data_array.dims if dim in data_array.coords]
    if axis_names:
        view_group.attrs["axes"] = np.asarray([np.bytes_(dim) for dim in axis_names], dtype="S")

    detector_root = f"/{entry.name.strip('/')}/data/{detector_group_name}"
    view_group["data"] = h5py.SoftLink(f"{detector_root}/data")
    if data_array.data.variances is not None:
        view_group["errors"] = h5py.SoftLink(f"{detector_root}/errors")
    for coord_name in data_array.coords:
        view_group[coord_name] = h5py.SoftLink(f"{detector_root}/{coord_name}")


def write_processed_detectors(
    state: ReductionState,
    *,
    entry_name: str = "processed",
) -> Path:
    """Persist ``state.detectors`` into the source NeXus file under one top-level NXentry."""
    if not state.file_path:
        raise ValueError("Cannot save processed detectors without a source sample file")

    normalized_entry_name = _normalize_processed_entry_name(entry_name)
    source_path = Path(state.file_path).resolve()
    source_entry_path = resolve_entry_path(source_path)

    with h5py.File(source_path, "a") as handle:
        entry = _replace_group(handle, normalized_entry_name, nx_class="NXentry")
        _write_text_dataset(entry, "definition", "SCARLET_processed")
        handle.attrs["default"] = np.bytes_(normalized_entry_name)

        meta = _replace_group(entry, "meta", nx_class="NXcollection")
        _write_text_dataset(meta, "source_file", str(source_path))
        _write_text_dataset(meta, "source_entry", source_entry_path)
        _write_text_dataset(meta, "sample_name", state.sample_name)
        _write_text_dataset(meta, "config_id", state.config_id)
        if state.reductions_steps:
            _write_text_list_dataset(meta, "reduction_steps", state.reductions_steps)
        if state.notes:
            _write_text_list_dataset(meta, "notes", state.notes)

        data_group = _replace_group(entry, "data", nx_class="NXcollection")
        default_view_name: str | None = None
        for detector_number, detector in sorted(state.detectors.items()):
            detector_group_name = f"detector{detector_number}"
            _write_dataarray_dataset(data_group, detector_group_name, detector)
            _write_nxdata_view(
                entry,
                detector_number=detector_number,
                detector_group_name=detector_group_name,
                detector=detector,
            )
            if default_view_name is None or detector_number == 0:
                default_view_name = f"data{detector_number}"
        if default_view_name is not None:
            entry.attrs["default"] = np.bytes_(default_view_name)

    return source_path


@dataclass(frozen=True)
class ReductionStep:
    name: str
    fn: StepFunction


def reduction_step(name: str) -> Callable[[_F], _F]:
    def decorator(fn: _F) -> _F:
        setattr(fn, "_reduction_step_name", name)
        return fn

    return decorator


def as_reduction_step(fn: StepFunction) -> ReductionStep:
    name = getattr(fn, "_reduction_step_name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"Function {fn.__name__} is not decorated with @reduction_step")
    return ReductionStep(name=name, fn=cast(StepFunction, fn))


@reduction_step("subtract references")
def subtract_references_step(state: ReductionState) -> ReductionState:
    dark_data_dict: dict[int, Any] = {}
    dark_file = state.workflow.get_dark(state.config_id)
    if dark_file:
        dark_data_dict = nexus_reader.read_all_detectors(dark_file, normalize_by_monitor=True, correct_deadtime=True)
    else:
        for key in state.detectors:
            dark_data_dict[key] = None

    ec_data_dict: dict[int, Any] = {}
    ec_file = state.workflow.get_empty_cell(state.config_id, "scattering")
    if ec_file:
        tr_ec = state.workflow.get_empty_cell_transmission(state.config_id)
        ec_data_dict = nexus_reader.read_all_detectors(ec_file, normalize_by_monitor=True, correct_deadtime=True)
    else:
        tr_ec = None
        for key in state.detectors:
            ec_data_dict[key] = None

    subtracted_data_dict = {}
    for key in state.detectors:
        corrected = subtract_scattering_references(
            state.detectors[key],
            state.transmission,
            dark=dark_data_dict[key],
            empty_cell=ec_data_dict[key],
            empty_cell_transmission=tr_ec,
        )
        corrected = _apply_workflow_mask(
            corrected,
            mask=state.workflow.get_mask(state.config_id, key),
        )
        subtracted_data_dict[key] = corrected
    state.detectors = subtracted_data_dict
    return state


@reduction_step("water normalization")
def normalization_step(state: ReductionState) -> ReductionState:
    configuration = _read_state_configuration(state)
    effective_config_id = state.workflow.resolve_flatfield_config(state.config_id)
    flatfield_path = state.workflow.get_flatfield(state.config_id)
    if flatfield_path is None:
        flatfield_path = state.workflow.build_water_flatfield(state.config_id)
        if effective_config_id == state.config_id:
            state.notes.append(f"Prepared water flatfield for {state.config_id}")
        else:
            state.notes.append(
                f"Prepared water flatfield for source config {effective_config_id} used by {state.config_id}"
            )

    flatfields = load_flatfield_file(flatfield_path)
    normalized: dict[int, Any] = {}
    for detector_number, detector in state.detectors.items():
        detector_array = _require_dataarray(detector, name=f"detector{detector_number}", ndim=2)
        detector_array = _apply_workflow_mask(
            detector_array,
            mask=state.workflow.get_mask(state.config_id, detector_number),
        )
        flatfield = flatfields.get(detector_number)
        if flatfield is None:
            raise ValueError(
                f"Missing detector{detector_number} flatfield in {flatfield_path} for configuration {state.config_id!r}"
            )
        flatfield_array = _require_dataarray(flatfield, name=f"flatfield detector{detector_number}", ndim=2)
        if detector_array.dims != flatfield_array.dims:
            raise ValueError(
                f"Flatfield dimension mismatch for detector{detector_number}: "
                f"expected {detector_array.dims}, got {flatfield_array.dims}"
            )
        if detector_array.shape != flatfield_array.shape:
            raise ValueError(
                f"Flatfield shape mismatch for detector{detector_number}: "
                f"expected {detector_array.shape}, got {flatfield_array.shape}"
            )
        corrected = detector_array / flatfield_array
        corrected = _merge_dataarray_masks(corrected, detector_array, flatfield_array)
        normalized[detector_number] = _normalize_detector_by_solid_angle(
            corrected,
            detector_distance=_get_detector_distance(configuration, detector_number=detector_number),
            beam_center=_get_beam_center(state, detector_number=detector_number),
            pixel_size=_get_detector_pixel_size(state, detector_number=detector_number),
        )

    state.detectors = normalized
    if effective_config_id == state.config_id:
        state.notes.append(f"Applied water flatfield and solid-angle correction from {Path(flatfield_path).name}")
    else:
        state.notes.append(
            f"Applied water flatfield and solid-angle correction from {Path(flatfield_path).name} "
            f"(source config {effective_config_id} for {state.config_id})"
        )
    return state


@reduction_step("azimuthal averaging")
def azimuthal_averaging_step(state: ReductionState) -> ReductionState:
    configuration = _read_state_configuration(state)
    wavelength = _get_wavelength(configuration)
    wavelength_uncertainty = 0.0 if not state.file_path else _read_wavelength_uncertainty(state.file_path, wavelength=wavelength)
    aperture_type = _aperture_type(configuration.collimation.aperture2)
    aperture2_opening = _aperture_opening(configuration.collimation.aperture2)
    aperture1_opening = _aperture_opening(configuration.collimation.aperture1)
    integrated: dict[int, Any] = {}
    for detector_number, detector in state.detectors.items():
        detector_array = _require_dataarray(detector, name=f"detector{detector_number}", ndim=2)
        beam_center = _get_beam_center(state, detector_number=detector_number)
        detector_distance = _get_detector_distance(configuration, detector_number=detector_number)
        pixel_size = _get_detector_pixel_size(state, detector_number=detector_number)
        q_map = compute_q_norm_map(
            np.asarray(detector_array.data.values, dtype=np.float64),
            beam_center=beam_center,
            detector_distance=detector_distance,
            pixel_size=pixel_size,
            wavelength=wavelength,
        )
        if aperture_type == "circular":
            q_error = compute_q_resolution_circular(
                q_map,
                r1=aperture1_opening[0]/2,
                r2=aperture2_opening[0]/2,
                collimation_distance=configuration.collimation.collimation_distance,
                distance=detector_distance,
                wavelength_spread=wavelength_uncertainty,
                wavelength=wavelength,
                pixel_size=pixel_size,
            )
        elif aperture_type == "slit":
            q_error = compute_q_resolution_rectangular(
                q_map,
                x1=aperture1_opening[0],
                y1=aperture1_opening[1],
                x2=aperture2_opening[0],
                y2=aperture2_opening[1],
                collimation_distance=configuration.collimation.collimation_distance,
                distance=detector_distance,     
                wavelength_spread=wavelength_uncertainty,
                wavelength=wavelength,
                pixel_size=pixel_size,
            )

        mask = state.workflow.get_mask(state.config_id, detector_number)
        if mask is not None and mask.shape != detector_array.shape:
            raise ValueError(
                f"Workflow mask shape mismatch for detector{detector_number}: "
                f"expected {detector_array.shape}, got {mask.shape}"
            )

        result = azimuthal_average(
            detector_array,
            q_map,
            mask=mask,
            q_error=q_error,
            n_bins=state.azimuthal_n_bins,
            q_scale=state.azimuthal_q_scale,
        )
        integrated[detector_number] = result.to_data_array()

    state.detectors = integrated
    state.notes.append(
        f"Computed azimuthal average with {state.azimuthal_n_bins} bins ({state.azimuthal_q_scale})"
    )
    return state


@reduction_step("save processed detectors")
def save_processed_detectors_step(state: ReductionState) -> ReductionState:
    output_path = write_processed_detectors(state, entry_name="processed")
    state.notes.append(f"Saved processed detectors into {output_path.name}:/processed")
    return state


@reduction_step("save azimuthal text")
def save_azimuthal_text_step(state: ReductionState) -> ReductionState:
    output_paths = write_azimuthal_average_text_outputs(state)
    for output_path in output_paths:
        state.workflow.add_artifact(output_path.name, output_path, kind="txt")
    state.notes.append(
        "Saved azimuthal average text output to "
        + ", ".join(path.name for path in output_paths)
    )
    return state

@dataclass(frozen=True)
class ReductionPipeline:
    steps: tuple[ReductionStep, ...] = field(default_factory=tuple)

    @classmethod
    def with_azimuthal_text_output(cls) -> "ReductionPipeline":
        return cls(
            steps=(
                as_reduction_step(subtract_references_step),
                as_reduction_step(normalization_step),
                as_reduction_step(azimuthal_averaging_step),
                as_reduction_step(save_processed_detectors_step),
                as_reduction_step(save_azimuthal_text_step),
            )
        )

    @classmethod
    def default(cls) -> "ReductionPipeline":
        return cls(
            steps=(
                as_reduction_step(subtract_references_step),
                as_reduction_step(normalization_step),
                as_reduction_step(azimuthal_averaging_step),
                as_reduction_step(save_processed_detectors_step),
                as_reduction_step(save_azimuthal_text_step),
            )
        )

    @property
    def step_names(self) -> tuple[str, ...]:
        return tuple(step.name for step in self.steps)

    def run(self, state: ReductionState) -> ReductionState:
        for step in self.steps:
            state = step.fn(state)
            state.reductions_steps.append(step.name)
        return state
    
    def run_all(self, workflow: WorkflowContext):
        for run in workflow.runs:
            if run.entity=="sample" and run.mode=="scattering":
                state = ReductionState(sample_name=run.sample_name, config_id=run.config_id,workflow=workflow)
                self.run(state)

    def run_for_sample(self, workflow: WorkflowContext, sample_name: str, config_id: str) -> ReductionState:
        state = ReductionState(sample_name=sample_name, config_id=config_id, workflow=workflow)
        return self.run(state)
