from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from scarlet.workflow.configuration import Configuration
from scarlet.reduction import (
    AzimuthalAverageResult,
    azimuthal_average,
    correct_detector_dead_time,
    compute_q_norm_map,
    compute_q_resolution_circular,
    normalize_by_solid_angle,
    subtract_scattering_references,
)
@dataclass(frozen=True)
class InstrumentConfig:
    detector_distance: float
    beam_center: tuple[float, float]
    pixel_size: tuple[float, float]
    wavelength: float
    wavelength_spread: Optional[float] = None
    collimation: Optional[Any] = None
    deadtime: Optional[float] = None

@dataclass(frozen=True)
class PipelineData:
    image: np.ndarray
    transmission: float
    monitor: float
    acquisition_time: float
    deadtime: Optional[float] = None
    configuration: Optional[Configuration] = None
    deadtime_corrected_image: np.ndarray = field(init=False, repr=False)
    deadtime_corrected_error: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        image = np.asarray(self.image, dtype=np.float64)
        if image.ndim != 2:
            raise ValueError(f"PipelineData.image must be 2D, got shape {image.shape}")
        transmission = float(self.transmission)
        if not np.isfinite(transmission) or transmission <= 0.0:
            raise ValueError(f"PipelineData.transmission must be > 0, got {self.transmission!r}")
        monitor = float(self.monitor)
        if not np.isfinite(monitor) or monitor <= 0.0:
            raise ValueError(f"PipelineData.monitor must be > 0, got {self.monitor!r}")
        acquisition_time = float(self.acquisition_time)
        if not np.isfinite(acquisition_time) or acquisition_time <= 0.0:
            raise ValueError(f"PipelineData.acquisition_time must be > 0, got {self.acquisition_time!r}")
        deadtime = self.deadtime
        if deadtime is not None:
            deadtime = float(deadtime)
            if not np.isfinite(deadtime):
                deadtime = None
            elif deadtime < 0.0:
                raise ValueError(f"PipelineData.deadtime must be >= 0, got {self.deadtime!r}")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "transmission", transmission)
        object.__setattr__(self, "monitor", monitor)
        object.__setattr__(self, "acquisition_time", acquisition_time)
        object.__setattr__(self, "deadtime", deadtime)
        raw_image = image * monitor
        applied_deadtime = 0.0 if deadtime is None else deadtime
        deadtime_corrected_image = correct_detector_dead_time(
            raw_image,
            acq_time=acquisition_time,
            deadtime=applied_deadtime,
        ) / monitor
        rate = raw_image / acquisition_time
        denominator = 1.0 - (rate * applied_deadtime)
        deadtime_corrected_error = np.sqrt(np.clip(raw_image, 0.0, None)) / (
            monitor * np.square(denominator)
        )
        object.__setattr__(self, "deadtime_corrected_image", deadtime_corrected_image)
        object.__setattr__(self, "deadtime_corrected_error", deadtime_corrected_error)

    @property
    def error(self) -> np.ndarray:
        return np.sqrt(np.clip(self.image, 0.0, None) / self.monitor)


@dataclass(frozen=True)
class ReductionInputs:
    detector_distance: float
    beam_center: tuple[float, float]
    pixel_size: tuple[float, float]
    wavelength: float
    sample_image: Optional[np.ndarray] = None
    sample_error: Optional[np.ndarray] = None
    sample_transmission: Optional[float] = None
    n_bins: int = 200
    dark_image: Optional[np.ndarray] = None
    dark_error: Optional[np.ndarray] = None
    empty_cell: Optional[PipelineData] = None
    empty_beam_image: Optional[np.ndarray] = None
    empty_beam_error: Optional[np.ndarray] = None
    water_corrected_image: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    wavelength_spread: Optional[float] = None
    collimation: Any = None
    sample_data: Optional[PipelineData] = None
    dark_data: Optional[PipelineData] = None
    empty_cell_data: Optional[PipelineData] = None
    empty_beam_data: Optional[PipelineData] = None

    def __post_init__(self) -> None:
        sample_data = self.sample_data
        if sample_data is not None:
            if self.sample_image is None:
                object.__setattr__(self, "sample_image", sample_data.image)
            if self.sample_error is None:
                object.__setattr__(self, "sample_error", sample_data.error)
            if self.sample_transmission is None:
                object.__setattr__(self, "sample_transmission", sample_data.transmission)

        dark_data = self.dark_data
        if dark_data is not None:
            if self.dark_image is None:
                object.__setattr__(self, "dark_image", dark_data.image)
            if self.dark_error is None:
                object.__setattr__(self, "dark_error", dark_data.error)

        if self.empty_cell is None and self.empty_cell_data is not None:
            object.__setattr__(self, "empty_cell", self.empty_cell_data)

        empty_beam_data = self.empty_beam_data
        if empty_beam_data is not None:
            if self.empty_beam_image is None:
                object.__setattr__(self, "empty_beam_image", empty_beam_data.image)
            if self.empty_beam_error is None:
                object.__setattr__(self, "empty_beam_error", empty_beam_data.error)


@dataclass
class ReductionState:
    inputs: ReductionInputs
    mask: Optional[np.ndarray] = None
    corrected: Optional[np.ndarray] = None
    corrected_error: Optional[np.ndarray] = None
    normalized_image: Optional[np.ndarray] = None
    normalized_error: Optional[np.ndarray] = None
    solid_angle_corrected: Optional[np.ndarray] = None
    solid_angle_error: Optional[np.ndarray] = None
    q_map: Optional[np.ndarray] = None
    q_error_map: Optional[np.ndarray] = None
    integration: Optional[AzimuthalAverageResult] = None

    def __post_init__(self) -> None:
        if self.mask is None and self.inputs.mask is not None:
            self.mask = np.asarray(self.inputs.mask, dtype=np.uint8)


@dataclass(frozen=True)
class ReductionStep:
    name: str
    fn: Callable[[ReductionState], ReductionState]


def _combine_masks(base_mask: Optional[np.ndarray], invalid_mask: np.ndarray) -> np.ndarray:
    invalid = np.asarray(invalid_mask, dtype=bool)
    if base_mask is None:
        return invalid.astype(np.uint8)
    base = np.asarray(base_mask, dtype=bool)
    if base.shape != invalid.shape:
        raise ValueError(f"Mask shape mismatch: expected {invalid.shape}, got {base.shape}")
    return np.logical_or(base, invalid).astype(np.uint8)


def _require_array(name: str, value: Optional[np.ndarray]) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing pipeline value: {name}")
    return value


def _subtract_references_step(state: ReductionState) -> ReductionState:
    inputs = state.inputs
    empty_cell = inputs.empty_cell
    empty_cell_image = None if empty_cell is None else empty_cell.image
    empty_cell_transmission = None if empty_cell is None else empty_cell.transmission
    corrected = subtract_scattering_references(
        inputs.sample_image,
        inputs.sample_transmission,
        dark=inputs.dark_image,
        empty_cell=empty_cell_image,
        empty_cell_transmission=empty_cell_transmission,
        empty_beam=inputs.empty_beam_image,
        empty_beam_transmission=1.0 if inputs.empty_beam_image is not None else None,
        distance=inputs.detector_distance,
        beam_center=inputs.beam_center,
    )

    corrected_variance = np.square(inputs.sample_error) / (inputs.sample_transmission * inputs.sample_transmission)
    if inputs.dark_error is not None and empty_cell is None:
        corrected_variance += np.square(inputs.dark_error) / (inputs.sample_transmission * inputs.sample_transmission)
    elif (
        inputs.dark_error is not None
        and empty_cell is not None
    ):
        corrected_variance += np.square(inputs.dark_error) * np.square(
            (1.0 / empty_cell.transmission) - (1.0 / inputs.sample_transmission)
        )
    if empty_cell is not None:
        corrected_variance += np.square(empty_cell.error) / (
            empty_cell.transmission * empty_cell.transmission
        )

    state.corrected = corrected
    state.corrected_error = np.sqrt(corrected_variance)
    return state


def _normalize_by_water_step(state: ReductionState) -> ReductionState:
    inputs = state.inputs
    corrected = _require_array("corrected", state.corrected)
    corrected_error = _require_array("corrected_error", state.corrected_error)
    water_corrected_image = inputs.water_corrected_image
    if water_corrected_image is None:
        raise ValueError("Missing pipeline input: water_corrected_image")
    if water_corrected_image.shape != corrected.shape:
        raise ValueError(
            "water_corrected shape mismatch: "
            f"expected {corrected.shape}, got {water_corrected_image.shape}"
        )

    invalid_water_mask = ~np.isfinite(water_corrected_image) | (water_corrected_image <= 0.0)
    state.normalized_image = corrected / water_corrected_image
    state.normalized_error = corrected_error / np.abs(water_corrected_image)
    state.mask = _combine_masks(state.mask, invalid_water_mask)
    return state


def _normalize_by_solid_angle_step(state: ReductionState) -> ReductionState:
    inputs = state.inputs
    normalized_image = _require_array("normalized_image", state.normalized_image)
    normalized_error = _require_array("normalized_error", state.normalized_error)
    state.solid_angle_corrected = normalize_by_solid_angle(
        normalized_image,
        detector_distance=inputs.detector_distance,
        beam_center=inputs.beam_center,
        pixel_size=inputs.pixel_size,
    )
    solid_angle_correction = normalize_by_solid_angle(
        np.ones_like(normalized_image, dtype=np.float64),
        detector_distance=inputs.detector_distance,
        beam_center=inputs.beam_center,
        pixel_size=inputs.pixel_size,
    )
    state.solid_angle_error = normalized_error * solid_angle_correction
    return state


def _compute_q_step(state: ReductionState) -> ReductionState:
    inputs = state.inputs
    image = _require_array("normalized_image", state.normalized_image)
    state.q_map = compute_q_norm_map(
        image,
        beam_center=inputs.beam_center,
        detector_distance=inputs.detector_distance,
        pixel_size=inputs.pixel_size,
        wavelength=inputs.wavelength,
    )

    wavelength_spread = inputs.wavelength_spread
    collimation = inputs.collimation
    if (
        wavelength_spread is not None
        and collimation is not None
        and getattr(collimation.aperture1, "diameter", None) is not None
        and getattr(collimation.aperture2, "diameter", None) is not None
    ):
        state.q_error_map = compute_q_resolution_circular(
            state.q_map,
            r1=float(collimation.aperture1.diameter),
            r2=float(collimation.aperture2.diameter),
            collimation_distance=float(collimation.collimation_distance),
            distance=inputs.detector_distance,
            wavelength_spread=float(wavelength_spread),
            wavelength=inputs.wavelength,
            pixel_size=inputs.pixel_size,
        )
    else:
        state.q_error_map = None
    return state


def _integrate_step(state: ReductionState) -> ReductionState:
    state.integration = azimuthal_average(
        _require_array("solid_angle_corrected", state.solid_angle_corrected),
        _require_array("q_map", state.q_map),
        mask=state.mask,
        intensity_error=state.solid_angle_error,
        q_error=state.q_error_map,
        n_bins=int(state.inputs.n_bins),
    )
    return state


@dataclass(frozen=True)
class ReductionPipeline:
    steps: tuple[ReductionStep, ...] = field(default_factory=tuple)

    @classmethod
    def default(cls) -> "ReductionPipeline":
        return cls(
            steps=(
                ReductionStep("subtract_references", _subtract_references_step),
                ReductionStep("normalize_by_water", _normalize_by_water_step),
                ReductionStep("normalize_by_solid_angle", _normalize_by_solid_angle_step),
                ReductionStep("compute_q", _compute_q_step),
                ReductionStep("integrate_azimuthally", _integrate_step),
            )
        )

    @property
    def step_names(self) -> tuple[str, ...]:
        return tuple(step.name for step in self.steps)

    def run(self, inputs: ReductionInputs) -> ReductionState:
        state = ReductionState(inputs=inputs)
        for step in self.steps:
            state = step.fn(state)
        return state
