from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from .azimuthal import DetectorAzimuthalCurve, azimuthal_average_from_arrays, flatten_detector_iq, resolve_azimuthal_q_range
from .corrections import (
    ROI,
    TransmissionComputation,
    apply_mask as apply_pixel_mask,
    background_correct_scattering,
    compute_transmission,
    normalize_by_water,
    require_same_shape,
    roi_view,
    subtract_empty_cell,
    zeros_like,
)
from .nexus import (
    PROCESSED_ENTRY_PATH,
    RAW_ENTRY_PATH,
    REFERENCE_BUNDLE_ENTRY_PATH,
    NormalizeBy,
    compute_q_axes,
    copy_raw_file_for_processing,
    create_processed_entry,
    list_detector_indices_in_file,
    read_combined_mask,
    read_frame_from_file,
    read_reference_frame,
    read_transmission_roi,
    read_transmission_roi_detector,
    write_dataset,
)


@dataclass(frozen=True)
class TransmissionResult:
    """Transmission computed from a rectangular direct-beam ROI."""

    value: float
    sample_roi_sum: float
    empty_beam_roi_sum: float
    roi: ROI
    detector_index: int
    method: str

    @classmethod
    def from_computation(cls, computation: TransmissionComputation, *, detector_index: int) -> "TransmissionResult":
        return cls(
            value=computation.value,
            sample_roi_sum=computation.sample_roi_sum,
            empty_beam_roi_sum=computation.empty_beam_roi_sum,
            roi=computation.roi,
            detector_index=detector_index,
            method=computation.method,
        )


@dataclass(frozen=True)
class DetectorReduction2DResult:
    """Reduced 2D arrays for one detector."""

    intensity: np.ndarray
    sample_corrected: np.ndarray
    water_corrected: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    detector_index: int


@dataclass(frozen=True)
class Reduction2DResult:
    """Result of the first deterministic SCARLET 2D reduction pass."""

    detector_results: dict[int, DetectorReduction2DResult]
    detector_q_axes: dict[int, tuple[np.ndarray, np.ndarray]]
    azimuthal_results: dict[int, DetectorAzimuthalCurve]
    sample_transmission: TransmissionResult
    water_transmission: Optional[TransmissionResult]
    normalize_by: NormalizeBy
    sample_scattering: Path
    sample_transmission_file: Optional[Path]
    refs_sub: Path
    refs_norm: Optional[Path]
    raw_entry: str
    processed_entry: str
    q_beam_centers: dict[int, tuple[float, float]]
    azimuthal_q_min: float
    azimuthal_q_max: float

    @property
    def detector_indices(self) -> list[int]:
        return sorted(self.detector_results)

    @property
    def primary_detector_index(self) -> int:
        return self.detector_indices[0]

    @property
    def intensity(self) -> np.ndarray:
        return self.detector_results[self.primary_detector_index].intensity

    @property
    def sample_corrected(self) -> np.ndarray:
        return self.detector_results[self.primary_detector_index].sample_corrected

    @property
    def water_corrected(self) -> Optional[np.ndarray]:
        return self.detector_results[self.primary_detector_index].water_corrected

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self.detector_results[self.primary_detector_index].mask

    @property
    def detector_index(self) -> int:
        return self.primary_detector_index


def compute_transmission_from_frames(
    sample_transmission: np.ndarray,
    empty_beam_transmission: np.ndarray,
    *,
    dark: Optional[np.ndarray],
    roi: ROI,
    detector_index: int,
    method: str,
) -> TransmissionResult:
    """
    Compute a transmission from normalized 2D detector images.

    This compatibility wrapper returns the historical ``TransmissionResult``
    dataclass while delegating the numerical work to
    :func:`scarlet.reduction.corrections.compute_transmission`.
    """
    return TransmissionResult.from_computation(
        compute_transmission(
            sample_transmission,
            empty_beam_transmission,
            dark=dark,
            roi=roi,
            method=method,
        ),
        detector_index=detector_index,
    )


def _assumed_unit_transmission(detector_index: int, roi: ROI) -> TransmissionResult:
    return TransmissionResult(
        value=1.0,
        sample_roi_sum=float("nan"),
        empty_beam_roi_sum=float("nan"),
        roi=roi,
        detector_index=detector_index,
        method="assumed_1_no_sample_transmission_file",
    )


def _correct_reference_scattering(
    reference_data: Optional[np.ndarray],
    *,
    dark: np.ndarray,
    empty_beam_scattering: Optional[np.ndarray],
    shape_like: np.ndarray,
    label: str,
) -> np.ndarray:
    if reference_data is None:
        return zeros_like(shape_like)
    require_same_shape(label, shape_like, reference_data)
    return background_correct_scattering(
        reference_data,
        dark=dark,
        empty_beam_scattering=empty_beam_scattering,
    )


def _center_of_mass_in_roi(
    data: np.ndarray,
    *,
    roi: ROI,
) -> tuple[float, float]:
    """Compute the intensity-weighted beam center inside a half-open ROI."""
    window = np.asarray(roi_view(data, roi), dtype=np.float64)
    finite = np.isfinite(window)
    if not np.any(finite):
        raise ValueError("Cannot compute center of mass: ROI contains no finite pixels")

    values = np.where(finite, window, 0.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Cannot compute center of mass: ROI sum is not positive")

    x0, _x1, y0, _y1 = roi
    ys, xs = np.indices(values.shape, dtype=np.float64)
    center_x = float(x0 + (values * xs).sum() / total)
    center_y = float(y0 + (values * ys).sum() / total)
    return center_x, center_y


def _resolve_detector_indices(
    sample_scattering: Path,
    *,
    raw_entry: str,
    detector_index: Optional[int],
) -> tuple[str, list[int]]:
    resolved_entry, available = list_detector_indices_in_file(sample_scattering, entry_path=raw_entry)
    if detector_index is not None:
        if detector_index not in available:
            available_text = ", ".join(f"detector{i}" for i in available) or "<none>"
            raise ValueError(
                f"Requested detector{detector_index} is missing from {sample_scattering}. "
                f"Available detectors: {available_text}"
            )
        return resolved_entry, [detector_index]
    if not available:
        raise ValueError(f"No detector datasets found in {sample_scattering}")
    return resolved_entry, available


def reduce_2d(
    sample_scattering: Union[str, Path],
    refs_sub: Union[str, Path],
    *,
    sample_transmission: Optional[Union[str, Path]] = None,
    refs_norm: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    detector_index: Optional[int] = None,
    normalize_by: NormalizeBy = "monitor",
    apply_mask: bool = True,
    overwrite: bool = False,
    raw_entry: str = RAW_ENTRY_PATH,
    processed_entry: str = PROCESSED_ENTRY_PATH,
    refs_entry: str = REFERENCE_BUNDLE_ENTRY_PATH,
    azimuthal_bins: int = 200,
    azimuthal_q_min: Optional[float] = None,
    azimuthal_q_max: Optional[float] = None,
) -> Reduction2DResult:
    """
    Run SCARLET's first deterministic 2D reduction pass.

    The raw input is read from ``/raw_data`` by default, with compatibility
    fallback to the older ``/entry`` convention. If ``output_path`` is given,
    the sample file is copied to that output path and the corrected image is
    written into a reduced-data NXentry, ``/processed_data`` by default.

    If ``detector_index`` is omitted, every detector present in the sample file
    is reduced. The transmission factor is computed once on the first selected
    detector and then applied to all reduced detector images. The output entry
    stores one azimuthal ``NXdata`` per detector.
    """
    if azimuthal_bins <= 0:
        raise ValueError(f"azimuthal_bins must be positive, got {azimuthal_bins}")

    sample_scattering = Path(sample_scattering)
    refs_sub = Path(refs_sub)
    refs_norm = None if refs_norm is None else Path(refs_norm)
    sample_transmission_path = None if sample_transmission is None else Path(sample_transmission)
    resolved_raw_entry, detector_indices = _resolve_detector_indices(
        sample_scattering,
        raw_entry=raw_entry,
        detector_index=detector_index,
    )
    transmission_detector_index = detector_indices[0]

    with h5py.File(refs_sub, "r") as refs_sub_file:
        roi = read_transmission_roi(refs_sub_file, refs_entry_path=refs_entry)
        transmission_detector_index = read_transmission_roi_detector(
            refs_sub_file,
            refs_entry_path=refs_entry,
        )
        if transmission_detector_index not in detector_indices:
            available_text = ", ".join(f"detector{i}" for i in detector_indices)
            raise ValueError(
                f"Transmission ROI targets detector{transmission_detector_index}, "
                f"but the sample file only exposes {available_text}"
            )

        empty_beam_transmission_frame = read_reference_frame(
            refs_sub_file,
            "empty_beam_transmission",
            detector_index=transmission_detector_index,
            normalize_by=normalize_by,
            refs_entry_path=refs_entry,
        )
        if empty_beam_transmission_frame is None:
            raise ValueError("refs_sub must contain an empty_beam_transmission reference")

        transmission_dark_frame = read_reference_frame(
            refs_sub_file,
            "dark",
            detector_index=transmission_detector_index,
            normalize_by=normalize_by,
            refs_entry_path=refs_entry,
        )
        transmission_dark = (
            zeros_like(empty_beam_transmission_frame.data)
            if transmission_dark_frame is None
            else transmission_dark_frame.data
        )
        require_same_shape("transmission dark", empty_beam_transmission_frame.data, transmission_dark)

        q_beam_centers: dict[int, tuple[float, float]] = {}
        transmission_center_data = empty_beam_transmission_frame.data - transmission_dark
        q_beam_centers[transmission_detector_index] = _center_of_mass_in_roi(
            transmission_center_data,
            roi=roi,
        )

        if sample_transmission_path is None:
            sample_tr = _assumed_unit_transmission(transmission_detector_index, roi)
        else:
            sample_transmission_frame = read_frame_from_file(
                sample_transmission_path,
                entry_path=raw_entry,
                detector_index=transmission_detector_index,
                normalize_by=normalize_by,
            )
            require_same_shape(
                "sample_transmission",
                empty_beam_transmission_frame.data,
                sample_transmission_frame.data,
            )
            sample_tr = compute_transmission_from_frames(
                sample_transmission_frame.data,
                empty_beam_transmission_frame.data,
                dark=transmission_dark,
                roi=roi,
                detector_index=transmission_detector_index,
                method="sample_transmission_over_empty_beam_transmission",
            )

    water_tr: Optional[TransmissionResult] = None
    if refs_norm is not None:
        with h5py.File(refs_norm, "r") as refs_norm_file:
            norm_roi = read_transmission_roi(refs_norm_file, refs_entry_path=refs_entry)
            water_transmission_frame = read_reference_frame(
                refs_norm_file,
                "water_transmission",
                detector_index=transmission_detector_index,
                normalize_by=normalize_by,
                refs_entry_path=refs_entry,
            )
            if water_transmission_frame is None:
                raise ValueError("refs_norm must contain a water_transmission reference")
            require_same_shape(
                "water_transmission",
                empty_beam_transmission_frame.data,
                water_transmission_frame.data,
            )

            norm_dark_frame = read_reference_frame(
                refs_norm_file,
                "dark",
                detector_index=transmission_detector_index,
                normalize_by=normalize_by,
                refs_entry_path=refs_entry,
            )
            norm_dark = transmission_dark if norm_dark_frame is None else norm_dark_frame.data
            require_same_shape("normalization dark", empty_beam_transmission_frame.data, norm_dark)

            norm_empty_beam_transmission_frame = read_reference_frame(
                refs_norm_file,
                "empty_beam_transmission",
                detector_index=transmission_detector_index,
                normalize_by=normalize_by,
                refs_entry_path=refs_entry,
            )
            if norm_empty_beam_transmission_frame is None:
                norm_empty_beam_transmission = empty_beam_transmission_frame.data
            else:
                norm_empty_beam_transmission = norm_empty_beam_transmission_frame.data
            require_same_shape(
                "normalization empty_beam_transmission",
                empty_beam_transmission_frame.data,
                norm_empty_beam_transmission,
            )

            water_tr = compute_transmission_from_frames(
                water_transmission_frame.data,
                norm_empty_beam_transmission,
                dark=norm_dark,
                roi=norm_roi,
                detector_index=transmission_detector_index,
                method="water_transmission_over_empty_beam_transmission",
            )

    detector_results: dict[int, DetectorReduction2DResult] = {}
    with h5py.File(refs_sub, "r") as refs_sub_file:
        refs_norm_file = h5py.File(refs_norm, "r") if refs_norm is not None else None
        try:
            for index in detector_indices:
                sample_frame = read_frame_from_file(
                    sample_scattering,
                    entry_path=raw_entry,
                    detector_index=index,
                    normalize_by=normalize_by,
                )
                shape = sample_frame.data.shape

                dark_frame = read_reference_frame(
                    refs_sub_file,
                    "dark",
                    detector_index=index,
                    normalize_by=normalize_by,
                    refs_entry_path=refs_entry,
                )
                dark = zeros_like(sample_frame.data) if dark_frame is None else dark_frame.data
                require_same_shape("dark", sample_frame.data, dark)

                empty_beam_scattering_frame = read_reference_frame(
                    refs_sub_file,
                    "empty_beam_scattering",
                    detector_index=index,
                    normalize_by=normalize_by,
                    refs_entry_path=refs_entry,
                )
                empty_beam_scattering_raw = None if empty_beam_scattering_frame is None else empty_beam_scattering_frame.data

                empty_cell_scattering_frame = read_reference_frame(
                    refs_sub_file,
                    "empty_cell_scattering",
                    detector_index=index,
                    normalize_by=normalize_by,
                    refs_entry_path=refs_entry,
                )
                empty_cell_raw = None if empty_cell_scattering_frame is None else empty_cell_scattering_frame.data
                empty_cell_corrected = _correct_reference_scattering(
                    empty_cell_raw,
                    dark=dark,
                    empty_beam_scattering=empty_beam_scattering_raw,
                    shape_like=sample_frame.data,
                    label="empty_cell_scattering",
                )

                sample_background_corrected = background_correct_scattering(
                    sample_frame.data,
                    dark=dark,
                    empty_beam_scattering=empty_beam_scattering_raw,
                )
                sample_corrected = subtract_empty_cell(
                    sample_background_corrected,
                    empty_cell_corrected,
                    transmission=sample_tr.value,
                )
                intensity = sample_corrected
                water_corrected = None

                mask = read_combined_mask(
                    refs_sub_file,
                    index,
                    shape,
                    refs_entry_path=refs_entry,
                ) if apply_mask else None

                if refs_norm_file is not None:
                    water_scattering_frame = read_reference_frame(
                        refs_norm_file,
                        "water_scattering",
                        detector_index=index,
                        normalize_by=normalize_by,
                        refs_entry_path=refs_entry,
                    )
                    if water_scattering_frame is None:
                        raise ValueError("refs_norm must contain a water_scattering reference")
                    require_same_shape("water_scattering", sample_frame.data, water_scattering_frame.data)

                    norm_dark_frame = read_reference_frame(
                        refs_norm_file,
                        "dark",
                        detector_index=index,
                        normalize_by=normalize_by,
                        refs_entry_path=refs_entry,
                    )
                    norm_dark = dark if norm_dark_frame is None else norm_dark_frame.data
                    require_same_shape("normalization dark", sample_frame.data, norm_dark)

                    norm_empty_beam_scattering_frame = read_reference_frame(
                        refs_norm_file,
                        "empty_beam_scattering",
                        detector_index=index,
                        normalize_by=normalize_by,
                        refs_entry_path=refs_entry,
                    )
                    norm_empty_beam_raw = None if norm_empty_beam_scattering_frame is None else norm_empty_beam_scattering_frame.data

                    norm_empty_cell_scattering_frame = read_reference_frame(
                        refs_norm_file,
                        "empty_cell_scattering",
                        detector_index=index,
                        normalize_by=normalize_by,
                        refs_entry_path=refs_entry,
                    )
                    norm_empty_cell_raw = None if norm_empty_cell_scattering_frame is None else norm_empty_cell_scattering_frame.data
                    norm_empty_cell_corrected = _correct_reference_scattering(
                        norm_empty_cell_raw,
                        dark=norm_dark,
                        empty_beam_scattering=norm_empty_beam_raw,
                        shape_like=sample_frame.data,
                        label="normalization empty_cell_scattering",
                    )

                    water_background_corrected = background_correct_scattering(
                        water_scattering_frame.data,
                        dark=norm_dark,
                        empty_beam_scattering=norm_empty_beam_raw,
                    )
                    water_corrected = subtract_empty_cell(
                        water_background_corrected,
                        norm_empty_cell_corrected,
                        transmission=water_tr.value,
                    )
                    intensity = normalize_by_water(sample_corrected, water_corrected)

                    if apply_mask:
                        norm_mask = read_combined_mask(
                            refs_norm_file,
                            index,
                            shape,
                            refs_entry_path=refs_entry,
                        )
                        if norm_mask is not None:
                            mask = norm_mask if mask is None else (mask | norm_mask)

                intensity = apply_pixel_mask(intensity, mask)
                sample_corrected = apply_pixel_mask(sample_corrected, mask)
                if water_corrected is not None:
                    water_corrected = apply_pixel_mask(water_corrected, mask)

                detector_results[index] = DetectorReduction2DResult(
                    intensity=intensity,
                    sample_corrected=sample_corrected,
                    water_corrected=water_corrected,
                    mask=mask,
                    detector_index=index,
                )
        finally:
            if refs_norm_file is not None:
                refs_norm_file.close()

    detector_q_axes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    all_q_values: list[np.ndarray] = []
    with h5py.File(sample_scattering, "r") as sample_file:
        for index in detector_indices:
            det_result = detector_results[index]
            qx, qy = compute_q_axes(
                sample_file,
                resolved_raw_entry,
                detector_index=index,
                shape=det_result.intensity.shape,
                beam_center=q_beam_centers.get(index),
            )
            detector_q_axes[index] = (qx, qy)
            q_values, _intensity_values = flatten_detector_iq(det_result.intensity, qx, qy)
            if q_values.size:
                all_q_values.append(q_values)

    if not all_q_values:
        raise ValueError("No finite reduced pixels available for azimuthal averaging")

    resolved_azimuthal_q_min, resolved_azimuthal_q_max = resolve_azimuthal_q_range(
        np.concatenate(all_q_values),
        q_min=azimuthal_q_min,
        q_max=azimuthal_q_max,
    )
    azimuthal_results: dict[int, DetectorAzimuthalCurve] = {}
    for index in detector_indices:
        det_result = detector_results[index]
        qx, qy = detector_q_axes[index]
        azimuthal_results[index] = azimuthal_average_from_arrays(
            det_result.intensity,
            qx,
            qy,
            detector_index=index,
            n_bins=azimuthal_bins,
            q_min=resolved_azimuthal_q_min,
            q_max=resolved_azimuthal_q_max,
        )

    result = Reduction2DResult(
        detector_results=detector_results,
        detector_q_axes=detector_q_axes,
        azimuthal_results=azimuthal_results,
        sample_transmission=sample_tr,
        water_transmission=water_tr,
        normalize_by=normalize_by,
        sample_scattering=sample_scattering.resolve(),
        sample_transmission_file=None if sample_transmission_path is None else sample_transmission_path.resolve(),
        refs_sub=refs_sub.resolve(),
        refs_norm=None if refs_norm is None else refs_norm.resolve(),
        raw_entry=resolved_raw_entry,
        processed_entry=processed_entry,
        q_beam_centers=q_beam_centers,
        azimuthal_q_min=resolved_azimuthal_q_min,
        azimuthal_q_max=resolved_azimuthal_q_max,
    )

    if output_path is not None:
        write_processed_2d_entry(
            output_path,
            result,
            overwrite=overwrite,
            processed_entry=processed_entry,
        )

    return result


def write_processed_2d_entry(
    output_path: Union[str, Path],
    result: Reduction2DResult,
    *,
    overwrite: bool = False,
    processed_entry: str = PROCESSED_ENTRY_PATH,
) -> Path:
    """
    Write a reduced-data NXentry into a NeXus file.

    If ``output_path`` differs from ``result.sample_scattering``, the raw sample
    file is copied first. The reduced output is then added under
    ``/processed_data`` by default. Raw data are kept unchanged, while the
    final ``NXdata`` groups are written as per-detector azimuthal ``I(Q)``
    curves.
    """
    output_path = copy_raw_file_for_processing(
        result.sample_scattering,
        output_path,
        overwrite=overwrite,
    )

    with h5py.File(output_path, "r+") as f:
        entry = create_processed_entry(f, processed_entry_path=processed_entry, overwrite=overwrite)
        write_dataset(entry, "definition", "SCARLET_azimuthal_iq")
        write_dataset(entry, "schema_version", "0.2")
        write_dataset(entry, "created_utc", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        write_dataset(entry, "raw_entry", result.raw_entry)

        primary_name = f"data{result.primary_detector_index}"
        for index in result.detector_indices:
            det_result = result.detector_results[index]
            azimuthal = result.azimuthal_results[index]
            data = entry.create_group(f"data{index}")
            data.attrs["NX_class"] = np.bytes_("NXdata")
            data.attrs["signal"] = np.bytes_("I")
            data.attrs["axes"] = np.asarray([np.bytes_("Q")])
            write_dataset(data, "I", azimuthal.intensity)
            q_ds = write_dataset(data, "Q", azimuthal.q)
            q_edges_ds = write_dataset(data, "Q_edges", azimuthal.q_edges)
            write_dataset(data, "n_pixels", azimuthal.n_pixels)
            write_dataset(data, "detector_index", int(index))
            q_ds.attrs["units"] = np.bytes_("1/angstrom")
            q_edges_ds.attrs["units"] = np.bytes_("1/angstrom")

            detail = entry.create_group(f"detector{index}")
            detail.attrs["NX_class"] = np.bytes_("NXcollection")
            write_dataset(detail, "I_2d", det_result.intensity)
            qx, qy = result.detector_q_axes[index]
            qx_ds = write_dataset(detail, "Qx", qx)
            qy_ds = write_dataset(detail, "Qy", qy)
            qx_ds.attrs["units"] = np.bytes_("1/angstrom")
            qy_ds.attrs["units"] = np.bytes_("1/angstrom")
            if index in result.q_beam_centers:
                center = entry.create_group(f"beam_center_detector{index}")
                center.attrs["NX_class"] = np.bytes_("NXcollection")
                write_dataset(center, "x", float(result.q_beam_centers[index][0]))
                write_dataset(center, "y", float(result.q_beam_centers[index][1]))
                write_dataset(center, "method", "center_of_mass_on_empty_beam_transmission_roi")
            write_dataset(detail, "sample_corrected", det_result.sample_corrected)
            if det_result.water_corrected is not None:
                write_dataset(detail, "water_corrected", det_result.water_corrected)
            if det_result.mask is not None:
                write_dataset(detail, "mask", det_result.mask.astype(np.uint8))

        entry["data"] = entry[primary_name]

        red = entry.create_group("reduction")
        red.attrs["NX_class"] = np.bytes_("NXprocess")
        write_dataset(red, "program", "scarlet")
        write_dataset(red, "stage", "azimuthal_iq")
        write_dataset(red, "detector_index", int(result.detector_index))
        write_dataset(red, "detector_indices", np.asarray(result.detector_indices, dtype=np.int64))
        write_dataset(red, "normalize_by", result.normalize_by)
        write_dataset(red, "azimuthal_bins", int(result.azimuthal_results[result.primary_detector_index].q.size))
        write_dataset(red, "azimuthal_q_min", float(result.azimuthal_q_min))
        write_dataset(red, "azimuthal_q_max", float(result.azimuthal_q_max))
        write_dataset(red, "formula_sample_corrected", "(sample - dark - (empty_beam - dark)) - T_sample * (empty_cell - dark - (empty_beam - dark))")
        if result.water_corrected is not None:
            write_dataset(red, "formula_I", "sample_corrected / water_corrected")
        else:
            write_dataset(red, "formula_I", "sample_corrected")

        _write_transmission_group(red, "sample_transmission", result.sample_transmission)
        if result.water_transmission is not None:
            _write_transmission_group(red, "water_transmission", result.water_transmission)

        inputs = red.create_group("inputs")
        inputs.attrs["NX_class"] = np.bytes_("NXcollection")
        write_dataset(inputs, "sample_scattering", result.sample_scattering)
        if result.sample_transmission_file is not None:
            write_dataset(inputs, "sample_transmission", result.sample_transmission_file)
        write_dataset(inputs, "refs_sub", result.refs_sub)
        if result.refs_norm is not None:
            write_dataset(inputs, "refs_norm", result.refs_norm)

    return output_path


def _write_transmission_group(parent: h5py.Group, name: str, transmission: TransmissionResult) -> h5py.Group:
    group = parent.create_group(name)
    group.attrs["NX_class"] = np.bytes_("NXcollection")
    write_dataset(group, "value", float(transmission.value))
    write_dataset(group, "sample_roi_sum", float(transmission.sample_roi_sum))
    write_dataset(group, "empty_beam_roi_sum", float(transmission.empty_beam_roi_sum))
    write_dataset(group, "roi", np.asarray(transmission.roi, dtype=np.int64))
    write_dataset(group, "method", transmission.method)
    write_dataset(group, "detector_index", int(transmission.detector_index))
    return group


def write_reduced_2d_file(
    output_path: Union[str, Path],
    result: Reduction2DResult,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Backward-compatible alias for :func:`write_processed_2d_entry`.
    """
    return write_processed_2d_entry(output_path, result, overwrite=overwrite)
