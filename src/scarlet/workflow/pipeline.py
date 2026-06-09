from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, TypeVar, cast

import h5py
import numpy as np

from scarlet.io.nexus_reader import (
    get_roi,
    read_all_detectors,
    read_configuration,
    read_detector_pixel_size,
    read_empty_beam_transmission_source_file,
    resolve_entry_path,
)
from scarlet.reduction.correction import normalize_by_solid_angle, subtract_scattering_references
from scarlet.reduction.geometry import compute_q_norm_map
from scarlet.reduction.integration import azimuthal_average
from scarlet.reduction.transmission import compute_transmission
from scarlet.workflow.reference import (
    compute_corrected_water_scattering,
    compute_reference_transmissions,
)


StepFunction = Callable[["ReductionState"], "ReductionState"]
_F = TypeVar("_F", bound=StepFunction)


@dataclass(frozen=True)
class PipelineData:
    data: np.ndarray
    data_error: np.ndarray | None
    x: np.ndarray | None
    x_error: float | None
    y: np.ndarray | None
    y_error: float | None
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class ReductionInputs:
    sample_file_scattering: str
    sample_file_transmission: str
    sample_transmission: float | None = None
    ref_sub_file: str = ""
    ref_norm_file: str = ""


@dataclass
class ReductionState:
    inputs: ReductionInputs
    detectors: dict[int, PipelineData] = field(default_factory=dict)
    water_corrected_detectors: dict[int, np.ndarray] = field(default_factory=dict)
    data: np.ndarray | None = None
    data_error: np.ndarray | None = None
    x: np.ndarray | None = None
    x_error: np.ndarray | None = None
    y: np.ndarray | None = None
    y_error: np.ndarray | None = None
    reductions_steps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReductionStep:
    name: str
    fn: StepFunction


def reduction_step(name: str) -> Callable[[_F], _F]:
    """Attach a pipeline step name to a state-transform function."""
    def decorator(fn: _F) -> _F:
        setattr(fn, "_reduction_step_name", name)
        return fn

    return decorator


def as_reduction_step(fn: StepFunction) -> ReductionStep:
    """Convert a decorated state-transform function into a ``ReductionStep``."""
    name = getattr(fn, "_reduction_step_name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"Function {fn.__name__} is not decorated with @reduction_step")
    return ReductionStep(name=name, fn=cast(StepFunction, fn))


@dataclass(frozen=True)
class ReductionPipeline:
    steps: tuple[ReductionStep, ...] = field(default_factory=tuple)

    @classmethod
    def default(cls) -> "ReductionPipeline":
        return cls(
            steps=(
                as_reduction_step(check_ref_sub_file),
                as_reduction_step(check_ref_norm_file),
                as_reduction_step(compute_transmission_step),
                as_reduction_step(subtract_references_step),
                as_reduction_step(normalize_by_water_step),
                as_reduction_step(azimutal_averaging_step),
            )
        )

    @property
    def step_names(self) -> tuple[str, ...]:
        return tuple(step.name for step in self.steps)

    def run(self, inputs: ReductionInputs) -> ReductionState:
        state = ReductionState(inputs=inputs)
        for step in self.steps:
            state = step.fn(state)
            state.reductions_steps.append(step.name)
        return state


@reduction_step("check reference subtraction file")
def check_ref_sub_file(state: ReductionState) -> ReductionState:
    if not state.inputs.ref_sub_file:
        raise ValueError("ref_sub_file is required for reference subtraction")
    compute_reference_transmissions(state.inputs.ref_sub_file)
    return state


@reduction_step("check reference normalization file")
def check_ref_norm_file(state: ReductionState) -> ReductionState:
    if not state.inputs.ref_norm_file:
        raise ValueError("ref_norm_file is required for normalization")
    compute_reference_transmissions(state.inputs.ref_norm_file)
    state.water_corrected_detectors = compute_corrected_water_scattering(state.inputs.ref_norm_file)

    return state


@reduction_step("compute transmission")
def compute_transmission_step(state: ReductionState) -> ReductionState:
    if state.inputs.sample_transmission is not None:
        return state
    eb_tr_path = read_empty_beam_transmission_source_file(state.inputs.ref_sub_file)
    roi, detnum = get_roi(state.inputs.ref_sub_file)
    transmission = compute_transmission(
        state.inputs.sample_file_transmission,
        eb_tr_path,
        roi,
        detector_number=detnum,
    )
    state.inputs = replace(state.inputs, sample_transmission=transmission)
    return state


@reduction_step("subtract references")
def subtract_references_step(state: ReductionState) -> ReductionState:
    if state.inputs.sample_transmission is None:
        raise ValueError("sample_transmission must be defined before subtracting references")

    sample_detectors = read_all_detectors(
        state.inputs.sample_file_scattering,
        normalize_by_monitor=True,
    )
    empty_cell_transmission = _read_reference_transmission(
        state.inputs.ref_sub_file,
        "empty_cell_scattering",
    )
    if empty_cell_transmission is None:
        empty_cell_transmission = _read_reference_transmission(
            state.inputs.ref_sub_file,
            "empty_cell_transmission",
        )

    corrected_detectors: dict[int, PipelineData] = {}
    for detector_number, sample_detector in sorted(sample_detectors.items()):
        dark = _read_reference_detector_data(state.inputs.ref_sub_file, "dark", detector_number=detector_number)
        empty_cell = _read_reference_detector_data(
            state.inputs.ref_sub_file,
            "empty_cell_scattering",
            detector_number=detector_number,
        )
        mask = _read_reference_mask(
            state.inputs.ref_sub_file,
            detector_number=detector_number,
        )

        if empty_cell is not None and empty_cell_transmission is None:
            raise ValueError("empty_cell transmission is required when empty_cell_scattering is present")

        corrected = subtract_scattering_references(
            sample_detector.data,
            state.inputs.sample_transmission,
            dark=None if dark is None else dark.data,
            empty_cell=None if empty_cell is None else empty_cell.data,
            empty_cell_transmission=empty_cell_transmission,
        )
        if mask is not None and mask.shape != corrected.shape:
            raise ValueError(
                f"Reference mask shape mismatch for detector{detector_number}: "
                f"expected {corrected.shape}, got {mask.shape}"
            )

        corrected_variance = np.square(sample_detector.error) / (state.inputs.sample_transmission ** 2)
        if dark is not None and empty_cell is None:
            corrected_variance += np.square(dark.data_error) / (state.inputs.sample_transmission ** 2)
        elif dark is not None and empty_cell is not None:
            corrected_variance += np.square(dark.data_error) * np.square(
                (1.0 / empty_cell_transmission) - (1.0 / state.inputs.sample_transmission)
            )
        if empty_cell is not None:
            corrected_variance += np.square(empty_cell.data_error) / (empty_cell_transmission ** 2)

        beam_center = _read_detector_beam_center(
            state.inputs.ref_sub_file,
            detector_number=detector_number,
        )
        pixel_size = read_detector_pixel_size(
            state.inputs.sample_file_scattering,
            detector_number=detector_number,
        )
        if pixel_size is None:
            raise ValueError("pixel_size is required to normalize the subtracted data by solid angle")
        detector_distance = _read_sample_detector_distance(
            state.inputs.ref_sub_file,
            detector_number=detector_number,
        )
        solid_angle_correction = normalize_by_solid_angle(
            np.ones_like(corrected, dtype=np.float64),
            detector_distance=detector_distance,
            beam_center=beam_center,
            pixel_size=pixel_size,
        )
        corrected_detectors[detector_number] = PipelineData(
            data=corrected * solid_angle_correction,
            data_error=np.sqrt(corrected_variance) * solid_angle_correction,
            x=None,
            x_error=None,
            y=None,
            y_error=None,
            mask=mask,
        )

    state.detectors = corrected_detectors
    primary_detector_number = 0 if 0 in corrected_detectors else next(iter(corrected_detectors), None)
    if primary_detector_number is not None:
        primary = corrected_detectors[primary_detector_number]
        state.data = primary.data
        state.data_error = primary.data_error
        state.x = primary.x
        state.x_error = primary.x_error
        state.y = primary.y
        state.y_error = primary.y_error
    return state


@reduction_step("flatfield correction")
def normalize_by_water_step(state: ReductionState) -> ReductionState:
    water_corrected_detectors = state.water_corrected_detectors
    if not water_corrected_detectors:
        water_corrected_detectors = compute_corrected_water_scattering(state.inputs.ref_norm_file)
        state.water_corrected_detectors = water_corrected_detectors

    normalized_detectors: dict[int, PipelineData] = {}
    for detector_number, detector in sorted(state.detectors.items()):
        water_corrected = water_corrected_detectors.get(detector_number)
        if water_corrected is None:
            raise ValueError(f"Missing water_corrected for detector{detector_number} in {state.inputs.ref_norm_file}")
        if water_corrected.shape != detector.data.shape:
            raise ValueError(
                f"water_corrected shape mismatch for detector{detector_number}: "
                f"expected {detector.data.shape}, got {water_corrected.shape}"
            )

        ref_norm_mask = _read_reference_mask(
            state.inputs.ref_norm_file,
            detector_number=detector_number,
        )
        merged_mask = _combine_masks(detector.mask, ref_norm_mask)
        invalid_water = ~np.isfinite(water_corrected) | (water_corrected <= 0.0)
        merged_mask = _combine_masks(merged_mask, invalid_water.astype(np.uint8))
        merged_mask_bool = None if merged_mask is None else np.asarray(merged_mask, dtype=bool)

        safe_water_corrected = water_corrected
        if merged_mask_bool is not None:
            safe_water_corrected = np.array(water_corrected, copy=True)
            safe_water_corrected[merged_mask_bool] = 1.0

        normalized_data = detector.data / safe_water_corrected
        normalized_error = None
        if detector.data_error is not None:
            normalized_error = detector.data_error / np.abs(safe_water_corrected)

        if merged_mask_bool is not None:
            normalized_data = np.array(normalized_data, copy=True)
            normalized_data[merged_mask_bool] = np.nan
            if normalized_error is not None:
                normalized_error = np.array(normalized_error, copy=True)
                normalized_error[merged_mask_bool] = np.nan

        normalized_detectors[detector_number] = PipelineData(
            data=normalized_data,
            data_error=normalized_error,
            x=detector.x,
            x_error=detector.x_error,
            y=detector.y,
            y_error=detector.y_error,
            mask=merged_mask,
        )

    state.detectors = normalized_detectors
    primary_detector_number = 0 if 0 in normalized_detectors else next(iter(normalized_detectors), None)
    if primary_detector_number is not None:
        primary = normalized_detectors[primary_detector_number]
        state.data = primary.data
        state.data_error = primary.data_error
        state.x = primary.x
        state.x_error = primary.x_error
        state.y = primary.y
        state.y_error = primary.y_error
    return state


@reduction_step("azimuthal averaging")
def azimutal_averaging_step(state: ReductionState) -> ReductionState:
    integrated_detectors: dict[int, PipelineData] = {}
    wavelength = _read_wavelength(state.inputs.ref_sub_file)

    for detector_number, detector in sorted(state.detectors.items()):
        beam_center = _read_detector_beam_center(
            state.inputs.ref_sub_file,
            detector_number=detector_number,
        )
        pixel_size = read_detector_pixel_size(
            state.inputs.sample_file_scattering,
            detector_number=detector_number,
        )
        if pixel_size is None:
            raise ValueError(f"pixel_size is required for detector{detector_number} azimuthal averaging")
        detector_distance = _read_sample_detector_distance(
            state.inputs.ref_sub_file,
            detector_number=detector_number,
        )
        q_map = compute_q_norm_map(
            detector.data,
            beam_center=beam_center,
            detector_distance=detector_distance,
            pixel_size=pixel_size,
            wavelength=wavelength,
        )
        integration = azimuthal_average(
            detector.data,
            q_map,
            mask=detector.mask,
            intensity_error=detector.data_error,
        )
        integrated_detectors[detector_number] = PipelineData(
            data=integration.intensity,
            data_error=integration.intensity_error,
            x=integration.q,
            x_error=integration.q_error,
            y=None,
            y_error=None,
            mask=None,
        )

    state.detectors = integrated_detectors
    primary_detector_number = 0 if 0 in integrated_detectors else next(iter(integrated_detectors), None)
    if primary_detector_number is not None:
        primary = integrated_detectors[primary_detector_number]
        state.data = primary.data
        state.data_error = primary.data_error
        state.x = primary.x
        state.x_error = primary.x_error
        state.y = primary.y
        state.y_error = primary.y_error
    return state


def _combine_masks(
    base_mask: np.ndarray | None,
    extra_mask: np.ndarray | None,
) -> np.ndarray | None:
    if base_mask is None and extra_mask is None:
        return None
    if base_mask is None:
        return np.asarray(extra_mask, dtype=np.uint8)
    if extra_mask is None:
        return np.asarray(base_mask, dtype=np.uint8)

    base = np.asarray(base_mask, dtype=bool)
    extra = np.asarray(extra_mask, dtype=bool)
    if base.shape != extra.shape:
        raise ValueError(f"Mask shape mismatch: expected {base.shape}, got {extra.shape}")
    return np.logical_or(base, extra).astype(np.uint8)


def _read_reference_detector_data(
    refs_file_path: str | Path,
    reference_name: str,
    *,
    detector_number: int,
) -> PipelineData | None:
    refs_file_path = Path(refs_file_path).resolve()
    refs_entry_path = resolve_entry_path(refs_file_path)

    with h5py.File(refs_file_path, "r") as handle:
        reference_entry_path = f"{refs_entry_path}/references/{reference_name}/entry"
        data_path = f"{reference_entry_path}/instrument/detector{detector_number}/data"
        monitor_path = f"{reference_entry_path}/control/integral"
        if data_path not in handle:
            return None
        if monitor_path not in handle:
            raise ValueError(f"Missing monitor integral for reference {reference_name!r}: {monitor_path}")

        raw_data = np.asarray(handle[data_path][()], dtype=np.float64)
        if raw_data.ndim != 2:
            raise ValueError(f"Reference detector data must be 2D at {data_path}, got shape {raw_data.shape}")

        monitor = float(np.asarray(handle[monitor_path][()]).reshape(()))
        if not np.isfinite(monitor) or monitor <= 0.0:
            raise ValueError(f"Reference monitor integral must be > 0 at {monitor_path}")

        data = raw_data / monitor
        error = np.sqrt(np.clip(raw_data, 0.0, None)) / monitor
        return PipelineData(data=data, data_error=error, x=None, x_error=None, y=None, y_error=None)


def _read_reference_transmission(refs_file_path: str | Path, reference_name: str) -> float | None:
    refs_file_path = Path(refs_file_path).resolve()
    refs_entry_path = resolve_entry_path(refs_file_path)

    with h5py.File(refs_file_path, "r") as handle:
        dataset_path = f"{refs_entry_path}/references/{reference_name}/entry/sample/transmission"
        if dataset_path not in handle:
            return None
        value = float(np.asarray(handle[dataset_path][()]).reshape(()))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Reference transmission must be > 0 at {dataset_path}")
        return value


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


def _read_reference_mask(
    refs_file_path: str | Path,
    *,
    detector_number: int,
) -> np.ndarray | None:
    refs_file_path = Path(refs_file_path).resolve()
    refs_entry_path = resolve_entry_path(refs_file_path)

    with h5py.File(refs_file_path, "r") as handle:
        dataset_path = f"{refs_entry_path}/mask/mask_detector{detector_number}"
        if dataset_path not in handle:
            return None

        mask = np.asarray(handle[dataset_path][()], dtype=np.uint8)
        if mask.ndim != 2:
            raise ValueError(f"Reference mask must be 2D at {dataset_path}, got shape {mask.shape}")
        return mask


def _read_sample_detector_distance(
    file_path: str | Path,
    *,
    detector_number: int,
) -> float:
    file_path = Path(file_path).resolve()
    configuration, _issues = read_configuration(file_path)
    distance = configuration.sample_detector_distance

    if isinstance(distance, list):
        if detector_number >= len(distance):
            raise ValueError(f"Missing sample_detector_distance for detector{detector_number} in {file_path}")
        distance_value = distance[detector_number]
    else:
        distance_value = distance

    distance_value = float(distance_value)
    if not np.isfinite(distance_value) or distance_value <= 0.0:
        raise ValueError(f"Invalid sample_detector_distance for detector{detector_number} in {file_path}")
    return distance_value


def _read_wavelength(file_path: str | Path) -> float:
    file_path = Path(file_path).resolve()
    configuration, _issues = read_configuration(file_path)
    wavelength = float(configuration.wavelength)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError(f"Invalid wavelength in {file_path}: {wavelength!r}")
    return wavelength
