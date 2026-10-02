from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import h5py
import numpy as np

from scarlet.io.mode_inference import guess_measurement_mode_from_image, guess_measurement_mode_from_nexus_image
from scarlet.workflow.configuration import Aperture, Collimation, Configuration

from .converters._hdf import (
    as_float_scalar,
    as_str,
    pick_entry,
    safe_get,
    safe_get_dataset,
)
from .converters._units import MM_TO_M, length_dataset_to_m, length_dataset_to_mm


MeasurementMode = Literal["scattering", "transmission", "unknown"]


@dataclass(frozen=True)
class RawRunMetadata:
    """Metadata required to register a raw acquisition without converting it."""

    sample_name: str
    sample_thickness_mm: Optional[float]
    configuration: Configuration
    mode: MeasurementMode


def _normalized_apparatus(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "")


def _dataset_text(handle: h5py.File, path: str) -> Optional[str]:
    value = safe_get(handle, path)
    if value is None:
        return None
    text = as_str(value).strip()
    return text or None


def _sample_thickness_mm(handle: h5py.File, entry: str) -> Optional[float]:
    return length_dataset_to_mm(safe_get_dataset(handle, f"{entry}/sample/thickness"))


def _guess_from_image(image: np.ndarray) -> MeasurementMode:
    array = np.asarray(image)
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3:
        array = np.sum(array, axis=0)
    if array.ndim != 2:
        return "unknown"
    return guess_measurement_mode_from_image(array).mode


def _is_standard_nxsas_input(path: Path) -> bool:
    """Return whether the file already exposes the SCARLET detector layout."""
    try:
        with h5py.File(path, "r") as handle:
            for entry in ("/raw_data", "/entry", "/entry0", "/entry1"):
                if (
                    entry in handle
                    and f"{entry}/instrument/detector0/data" in handle
                    and f"{entry}/control" in handle
                ):
                    return True
    except OSError:
        return False
    return False


def is_pipeline_ready_nexus(path: str | Path) -> bool:
    """Return whether a run can be consumed directly by the reduction pipeline."""
    return _is_standard_nxsas_input(Path(path))


def _inspect_standard(path: Path) -> RawRunMetadata:
    # Import lazily so tests and callers can patch the public reader.
    from scarlet.workflow.configuration import configuration_from_nexus

    configuration, _issues = configuration_from_nexus(path)
    with h5py.File(path, "r") as handle:
        entry = pick_entry(handle)
        sample_name = (
            _dataset_text(handle, f"{entry}/sample/name")
            or _dataset_text(handle, f"{entry}/title")
            or path.stem
        )
        thickness = _sample_thickness_mm(handle, entry)
    mode = guess_measurement_mode_from_nexus_image(path, entry_path=None).mode
    return RawRunMetadata(sample_name, thickness, configuration, mode)


def _inspect_d11(path: Path) -> RawRunMetadata:
    from .converters.d11 import (
        _collimation_distance_m,
        _d11_instrument_path,
        _mm_field_to_m,
        _read_detector_counts,
        _sample_name,
    )

    with h5py.File(path, "r") as handle:
        entry = pick_entry(handle)
        instrument = _d11_instrument_path(handle, entry)
        wavelength_value = safe_get(handle, f"{instrument}/selector/wavelength")
        wavelength = as_float_scalar(wavelength_value) if wavelength_value is not None else float("nan")
        distance = length_dataset_to_m(safe_get_dataset(handle, f"{instrument}/detector/det_actual"))
        detector_distance = float(distance) if distance is not None else float("nan")
        collimation_distance = _collimation_distance_m(handle, instrument)

        guide_width = _mm_field_to_m(handle, f"{instrument}/collimation/guide_exit_cross_section_width")
        guide_height = _mm_field_to_m(handle, f"{instrument}/collimation/guide_exit_cross_section_height")
        sample_x = _mm_field_to_m(handle, f"{instrument}/Beam/sample_ap_x_or_diam")
        sample_y = _mm_field_to_m(handle, f"{instrument}/Beam/sample_ap_y")
        aperture1 = Aperture(type="slit", x_gap=guide_width, y_gap=guide_height)
        if sample_y is not None and sample_y == 0.0:
            aperture2 = Aperture(type="pinhole", diameter=sample_x)
        else:
            aperture2 = Aperture(type="slit", x_gap=sample_x, y_gap=sample_y)
        configuration = Configuration(
            wavelength=float(wavelength),
            sample_detector_distance=detector_distance,
            collimation=Collimation(
                aperture1=aperture1,
                aperture2=aperture2,
                collimation_distance=float(collimation_distance),
                last_aperture_to_sample_distance=0.0,
            ),
        )
        mode = _guess_from_image(_read_detector_counts(handle, entry, instrument))
        return RawRunMetadata(
            sample_name=_sample_name(handle, entry),
            sample_thickness_mm=_sample_thickness_mm(handle, entry),
            configuration=configuration,
            mode=mode,
        )


def _sam_slit_widths_mm(handle: h5py.File, instrument: str, index: int) -> tuple[Optional[float], Optional[float]]:
    from .converters.sam import _sam_sample_aperture_mm

    if index == 4:
        sample_x, sample_y = _sam_sample_aperture_mm(handle, instrument)
        if sample_x is not None or sample_y is not None:
            return sample_x, sample_y

    virtual_slits = f"{instrument}/VirtualSlitAxis"

    def width(axis: str) -> Optional[float]:
        value = safe_get(handle, f"{virtual_slits}/s{index}{axis}_actual_width")
        if value is None:
            value = safe_get(handle, f"{virtual_slits}/s{index}{axis}_wanted_width")
        return None if value is None else float(as_float_scalar(value))

    return width("w"), width("h")


def _inspect_sam(path: Path) -> RawRunMetadata:
    from .converters.sam import (
        _sam_detector_distance_m,
        _sam_instrument_path,
        _sam_read_detector_counts,
        _sam_sample_name,
        _sam_sample_thickness_mm,
    )

    with h5py.File(path, "r") as handle:
        entry = pick_entry(handle)
        instrument = _sam_instrument_path(handle, entry)
        wavelength_value = safe_get(handle, f"{instrument}/Selector/wavelength")
        wavelength = as_float_scalar(wavelength_value) if wavelength_value is not None else float("nan")
        detector_distance = _sam_detector_distance_m(handle, instrument)

        collimation_value = safe_get(handle, f"{instrument}/collimation/position")
        if collimation_value is None:
            collimation_value = safe_get(handle, f"{instrument}/collimation/sourceDistance")
        collimation_distance = as_float_scalar(collimation_value) if collimation_value is not None else 1.0

        first_in_guide: Optional[int] = None
        for index in (1, 2, 3):
            state = _dataset_text(handle, f"{instrument}/collimation/col{index}_state")
            if state is None or state.lower() == "in":
                first_in_guide = index
                break
        aperture1_index = 1 if first_in_guide is None else max(1, first_in_guide - 1)
        ap1_x_mm, ap1_y_mm = _sam_slit_widths_mm(handle, instrument, aperture1_index)
        aperture1 = Aperture(
            type="slit",
            x_gap=None if ap1_x_mm is None else ap1_x_mm * MM_TO_M,
            y_gap=None if ap1_y_mm is None else ap1_y_mm * MM_TO_M,
        )

        ap2_x_mm, ap2_y_mm = _sam_slit_widths_mm(handle, instrument, 4)
        if ap2_y_mm is not None and ap2_y_mm == 0.0:
            aperture2 = Aperture(
                type="pinhole",
                diameter=None if ap2_x_mm is None else ap2_x_mm * MM_TO_M,
            )
        else:
            aperture2 = Aperture(
                type="slit",
                x_gap=None if ap2_x_mm is None else ap2_x_mm * MM_TO_M,
                y_gap=None if ap2_y_mm is None else ap2_y_mm * MM_TO_M,
            )

        configuration = Configuration(
            wavelength=float(wavelength),
            sample_detector_distance=float(detector_distance),
            collimation=Collimation(
                aperture1=aperture1,
                aperture2=aperture2,
                collimation_distance=float(max(collimation_distance, 0.0)),
                last_aperture_to_sample_distance=0.1,
            ),
        )

        measurement_type = _dataset_text(handle, f"{instrument}/MeasurementType/typeOfMeasure")
        if measurement_type and "transmission" in measurement_type.lower():
            mode: MeasurementMode = "transmission"
        elif measurement_type and "scattering" in measurement_type.lower():
            mode = "scattering"
        else:
            mode = _guess_from_image(_sam_read_detector_counts(handle, entry, []))

        return RawRunMetadata(
            sample_name=_sam_sample_name(handle, entry),
            sample_thickness_mm=_sam_sample_thickness_mm(handle, entry),
            configuration=configuration,
            mode=mode,
        )


def _sansllb_detector_names(handle: h5py.File, instrument: str) -> list[str]:
    group = handle[instrument]
    names = [name for name, obj in group.items() if name.startswith("detector") and isinstance(obj, h5py.Group)]
    if names:
        return sorted(names, key=lambda name: int(name.replace("detector", "")))
    preferred = {
        "central_detector": 0,
        "left_detector": 1,
        "bottom_detector": 2,
        "right_detector": 3,
        "top_detector": 4,
    }
    names = [
        name
        for name, obj in group.items()
        if name.endswith("_detector")
        and isinstance(obj, h5py.Group)
        and as_str(obj.attrs.get("NX_class")) == "NXdetector"
    ]
    return sorted(names, key=lambda name: (preferred.get(name, 99), name))


def _sansllb_aperture1(handle: h5py.File, instrument: str) -> Aperture:
    collimator_path = f"{instrument}/collimator"
    if collimator_path not in handle:
        x_gap = length_dataset_to_m(safe_get_dataset(handle, f"{instrument}/aperture/x_gap"))
        y_gap = length_dataset_to_m(safe_get_dataset(handle, f"{instrument}/aperture/y_gap"))
        return Aperture(type="slit", x_gap=x_gap, y_gap=y_gap)

    collimator = handle[collimator_path]
    slit_indices = sorted(
        int(name[4:]) for name in collimator if name.startswith("slit") and name[4:].isdigit()
    )
    guide_indices = sorted(
        int(name[5:]) for name in collimator if name.startswith("guide") and name[5:].isdigit()
    )
    max_guide = guide_indices[-1] if guide_indices else None
    if max_guide is not None:
        slit_indices = [index for index in slit_indices if index <= max_guide + 1]

    ordered: list[tuple[str, int]] = []
    if max_guide is not None:
        for index in range(max_guide + 1):
            if index in slit_indices:
                ordered.append(("slit", index))
            if index in guide_indices:
                ordered.append(("guide", index))
        if max_guide + 1 in slit_indices:
            ordered.append(("slit", max_guide + 1))

    first_in_position: Optional[int] = None
    for position, (kind, index) in enumerate(ordered):
        if kind != "guide":
            continue
        selection = _dataset_text(handle, f"{collimator_path}/guide{index}/selection")
        if selection is None or selection.lower() != "ft":
            first_in_position = position
            break

    slit_positions = [position for position, item in enumerate(ordered) if item[0] == "slit"]
    chosen_position: Optional[int] = None
    if first_in_position is not None:
        before = [position for position in slit_positions if position < first_in_position]
        after = [position for position in slit_positions if position > first_in_position]
        chosen_position = before[-1] if before else (after[0] if after else None)
    if chosen_position is None and slit_positions:
        chosen_position = slit_positions[0]
    if chosen_position is None:
        return Aperture(type="pinhole")

    _, slit_index = ordered[chosen_position]
    x_gap = length_dataset_to_m(safe_get_dataset(handle, f"{collimator_path}/slit{slit_index}/x_gap"))
    y_gap = length_dataset_to_m(safe_get_dataset(handle, f"{collimator_path}/slit{slit_index}/y_gap"))
    return Aperture(type="slit", x_gap=x_gap, y_gap=y_gap)


def _sansllb_aperture2(handle: h5py.File, instrument: str) -> Aperture:
    sample_mask = f"{instrument}/sample_mask"
    shape = (_dataset_text(handle, f"{sample_mask}/shape") or "").lower()
    size = length_dataset_to_m(safe_get_dataset(handle, f"{sample_mask}/size"))
    size_y = length_dataset_to_m(safe_get_dataset(handle, f"{sample_mask}/size_y"))
    if shape == "circle" and size is not None:
        return Aperture(type="pinhole", diameter=size)
    if size is not None and size_y is not None:
        return Aperture(type="slit", x_gap=size, y_gap=size_y)
    return Aperture(type="slit", x_gap=0.01, y_gap=0.01)


def _sansllb_mode_from_instrument_state(
    handle: h5py.File,
    instrument: str,
) -> MeasurementMode:
    """Infer the SANS-LLB mode from the physical beam-stop position.

    The semi-transparent beam stop leaves a direct-beam-like signal in
    scattering images, so image morphology alone is not a reliable mode
    discriminator.  SANS-LLB stores the beam-stop centre and size in the raw
    file.  When its centre overlaps the beam axis the run is scattering; when
    it is at least one beam-stop diameter away the beam is clear and the run
    is transmission.  An inserted attenuator is a secondary transmission
    signal when beam-stop metadata is missing.  Intermediate beam-stop
    positions are deliberately left unknown for the image fallback (they
    commonly occur during commissioning scans).
    """
    beam_stop = f"{instrument}/beam_stop"
    x_mm = length_dataset_to_mm(safe_get_dataset(handle, f"{beam_stop}/x"))
    y_mm = length_dataset_to_mm(safe_get_dataset(handle, f"{beam_stop}/y"))
    size_mm = length_dataset_to_mm(safe_get_dataset(handle, f"{beam_stop}/size"))
    if x_mm is not None and y_mm is not None and size_mm is not None:
        if all(np.isfinite(value) for value in (x_mm, y_mm, size_mm)) and size_mm > 0.0:
            distance_from_axis_mm = float(np.hypot(x_mm, y_mm))
            if distance_from_axis_mm <= 0.5 * size_mm:
                return "scattering"
            if distance_from_axis_mm >= size_mm:
                return "transmission"

    attenuator_selection = _dataset_text(handle, f"{instrument}/attenuator/selection")
    if attenuator_selection is not None:
        normalized_selection = attenuator_selection.strip().lower()
        try:
            if float(normalized_selection) > 0.0:
                return "transmission"
        except ValueError:
            if normalized_selection in {"in", "inserted", "attenuated", "closed"}:
                return "transmission"
    return "unknown"


def _inspect_sansllb(path: Path) -> RawRunMetadata:
    from .converters.sansllb import _wavelength_dataset_to_angstrom

    with h5py.File(path, "r") as handle:
        entry = pick_entry(handle)
        instrument = f"{entry}/instrument"
        if instrument not in handle:
            for name, obj in handle[entry].items():
                if isinstance(obj, h5py.Group) and as_str(obj.attrs.get("NX_class")) == "NXinstrument":
                    instrument = f"{entry}/{name}"
                    break
        if instrument not in handle:
            raise ValueError(f"No NXinstrument group found in {path}")

        wavelength_dataset = safe_get_dataset(handle, f"{instrument}/source/incident_wavelength")
        if wavelength_dataset is None:
            wavelength_dataset = safe_get_dataset(handle, f"{instrument}/velocity_selector/wavelength")
        wavelength = _wavelength_dataset_to_angstrom(wavelength_dataset)

        detector_names = _sansllb_detector_names(handle, instrument)
        distances = [
            length_dataset_to_m(safe_get_dataset(handle, f"{instrument}/{name}/distance")) or 0.0
            for name in detector_names
        ]
        sample_detector_distance: float | list[float]
        if len(distances) == 1:
            sample_detector_distance = float(distances[0])
        else:
            sample_detector_distance = [float(value) for value in distances]

        collimator = f"{instrument}/collimator"
        collimation_distance = length_dataset_to_m(safe_get_dataset(handle, f"{collimator}/length"))
        if collimation_distance is None:
            collimation_distance = length_dataset_to_m(safe_get_dataset(handle, f"{collimator}/geometry/size"))
        if collimation_distance is None:
            collimation_distance = 1.0
        last_aperture_distance = length_dataset_to_m(safe_get_dataset(handle, f"{collimator}/distance"))
        if last_aperture_distance is None:
            slit_names = (
                sorted(
                    (name for name in handle[collimator] if name.startswith("slit")),
                    reverse=True,
                )
                if collimator in handle
                else ()
            )
            for name in slit_names:
                last_aperture_distance = length_dataset_to_m(
                    safe_get_dataset(handle, f"{collimator}/{name}/distance")
                )
                if last_aperture_distance is not None:
                    break
        if last_aperture_distance is None:
            last_aperture_distance = 0.1

        configuration = Configuration(
            wavelength=float(wavelength) if wavelength is not None else float("nan"),
            sample_detector_distance=sample_detector_distance,
            collimation=Collimation(
                aperture1=_sansllb_aperture1(handle, instrument),
                aperture2=_sansllb_aperture2(handle, instrument),
                collimation_distance=float(collimation_distance),
                last_aperture_to_sample_distance=float(last_aperture_distance),
            ),
        )
        sample_name = _dataset_text(handle, f"{entry}/sample/name") or path.stem
        mode = _sansllb_mode_from_instrument_state(handle, instrument)
        if mode == "unknown" and detector_names:
            detector = handle[f"{instrument}/{detector_names[0]}"]
            if "data" in detector:
                mode = _guess_from_image(detector["data"][()])
        return RawRunMetadata(
            sample_name=sample_name,
            sample_thickness_mm=_sample_thickness_mm(handle, entry),
            configuration=configuration,
            mode=mode,
        )


def inspect_raw_run(apparatus: str, path: str | Path) -> RawRunMetadata:
    """Inspect one acquisition without creating a converted NeXus file."""
    input_path = Path(path).resolve()
    if _is_standard_nxsas_input(input_path):
        return _inspect_standard(input_path)

    key = _normalized_apparatus(apparatus)
    if key == "d11":
        return _inspect_d11(input_path)
    if key == "sam":
        return _inspect_sam(input_path)
    if key == "sansllb":
        return _inspect_sansllb(input_path)
    raise ValueError(f"Unknown apparatus {apparatus!r}; cannot inspect raw acquisition")


__all__ = ["RawRunMetadata", "inspect_raw_run", "is_pipeline_ready_nexus"]
