from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from scarlet.io.nexus_reader import read_detector_data, resolve_entry_path


ROI = tuple[int, int, int, int]
BeamCenter = tuple[float, float]


def _read_attenuator_transmission(file_path: Path) -> float:
    """Return ``I_attenuated / I_incident`` from a SCARLET NeXus file.

    A missing ``NXattenuator`` means that no attenuator is present and therefore
    has a neutral transmission of one.  An attenuator group without a usable
    transmission is rejected because silently assuming one would overestimate
    or underestimate the direct flux.
    """
    entry_path = resolve_entry_path(file_path)
    attenuator_path = f"{entry_path}/instrument/attenuator"
    transmission_path = f"{attenuator_path}/attenuator_transmission"
    with h5py.File(file_path, "r") as handle:
        if attenuator_path not in handle:
            return 1.0
        if transmission_path not in handle:
            raise ValueError(
                f"Missing attenuator transmission in {file_path}: {transmission_path}"
            )
        raw_value = np.asarray(handle[transmission_path][()])

    if raw_value.size != 1:
        raise ValueError(
            f"Attenuator transmission must be scalar in {file_path}: {transmission_path}"
        )
    try:
        transmission = float(raw_value.reshape(()))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid attenuator transmission in {file_path}: {transmission_path}"
        ) from exc
    if not np.isfinite(transmission) or not (0.0 < transmission <= 1.0):
        raise ValueError(
            "Attenuator transmission must be finite and in (0, 1] in "
            f"{file_path}: got {transmission!r}"
        )
    return transmission


def compute_direct_flux(
    direct_beam_file: str | Path,
    *,
    detector_number: int = 0,
    roi: ROI | None = None,
) -> float:
    """Compute the attenuation-corrected direct flux per monitor unit.

    Detector counts are corrected for dead time, normalized by the monitor,
    summed over the complete detector by default, and divided by the
    ``attenuator_transmission`` stored in the file.  When ``roi`` is provided,
    only that ``(x0, x1, y0, y1)`` rectangle is integrated.

    Files without an ``NXattenuator`` group are treated as unattenuated
    measurements (transmission equal to one).
    """
    direct_beam_file = Path(direct_beam_file).resolve()
    detector_number = int(detector_number)
    if detector_number < 0:
        raise ValueError(f"Detector index must be >= 0, got {detector_number}")

    data = np.asarray(
        read_detector_data(
            direct_beam_file,
            detector_number,
            normalize_by_monitor=True,
            correct_deadtime=True,
        ),
        dtype=np.float64,
    )
    if data.ndim != 2:
        raise ValueError(f"Direct-beam image must be 2D, got shape {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("Direct-beam image must contain only finite values")

    if roi is not None:
        x0, x1, y0, y1 = (int(value) for value in roi)
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid ROI: {(x0, x1, y0, y1)}")
        if x1 > data.shape[1] or y1 > data.shape[0]:
            raise ValueError(
                f"ROI {(x0, x1, y0, y1)} is outside detector{detector_number} "
                f"data shape {data.shape}"
            )
        data = data[y0:y1, x0:x1]

    normalized_sum = float(np.sum(data, dtype=np.float64))
    if normalized_sum <= 0.0:
        raise ValueError(
            "Monitor-normalized direct-beam detector sum must be > 0, "
            f"got {normalized_sum!r}"
        )
    return normalized_sum / _read_attenuator_transmission(direct_beam_file)


def compute_beam_center(
    transmission_file: str | Path,
    *,
    detector_number: int = 0,
) -> BeamCenter:
    """
    Compute the beam center from an empty-beam transmission file.

    The returned coordinates follow SCARLET's detector convention: ``(x, y)``
    in pixel units.
    """
    transmission_file = Path(transmission_file).resolve()
    detector_number = int(detector_number)
    if detector_number < 0:
        raise ValueError(f"Detector index must be >= 0, got {detector_number}")

    data = read_detector_data(
        transmission_file,
        detector_number,
        normalize_by_monitor=True,
    )
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Transmission image must be 2D, got shape {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("Transmission image must contain only finite values")

    total = float(np.sum(data, dtype=np.float64))
    if total <= 0.0:
        raise ValueError(f"Transmission image sum must be > 0 to compute beam center, got {total!r}")

    y_coords = np.arange(data.shape[0], dtype=np.float64)
    x_coords = np.arange(data.shape[1], dtype=np.float64)
    beam_center_x = float(np.sum(data * x_coords[np.newaxis, :], dtype=np.float64) / total)
    beam_center_y = float(np.sum(data * y_coords[:, np.newaxis], dtype=np.float64) / total)
    return beam_center_x, beam_center_y


def compute_transmission_roi(
    transmission_file: str | Path,
    *,
    detector_number: int = 0,
    padding_pixels: int = 1,
    padding_fraction: float = 0.05,
    threshold_fraction: float = 0.10,
) -> ROI:
    """
    Estimate a transmission ROI from an empty-beam transmission file.

    The returned ROI follows SCARLET's rectangle convention: ``(x0, x1, y0, y1)``
    with ``x1`` and ``y1`` exclusive.
    """
    transmission_file = Path(transmission_file).resolve()
    detector_number = int(detector_number)
    if detector_number < 0:
        raise ValueError(f"Detector index must be >= 0, got {detector_number}")
    if padding_pixels < 0:
        raise ValueError(f"padding_pixels must be >= 0, got {padding_pixels}")
    if not (0.0 <= threshold_fraction <= 1.0):
        raise ValueError(f"threshold_fraction must be in [0, 1], got {threshold_fraction}")
    if padding_fraction < 0.0:
        raise ValueError(f"padding_fraction must be >= 0, got {padding_fraction}")

    data = read_detector_data(
        transmission_file,
        detector_number,
        normalize_by_monitor=True,
    )
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Transmission image must be 2D, got shape {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("Transmission image must contain only finite values")

    ny, nx = data.shape
    peak_y, peak_x = np.unravel_index(int(np.argmax(data)), data.shape)
    peak_value = float(data[peak_y, peak_x])
    background = float(np.median(data))
    threshold = background + threshold_fraction * max(0.0, peak_value - background)
    bright_mask = data >= threshold

    component_mask = np.zeros_like(bright_mask, dtype=bool)
    stack = [(int(peak_y), int(peak_x))]
    component_mask[peak_y, peak_x] = True
    while stack:
        y, x = stack.pop()
        for yy in range(max(0, y - 1), min(ny, y + 2)):
            for xx in range(max(0, x - 1), min(nx, x + 2)):
                if component_mask[yy, xx] or not bright_mask[yy, xx]:
                    continue
                component_mask[yy, xx] = True
                stack.append((yy, xx))

    ys, xs = np.nonzero(component_mask)
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    width = x1 - x0
    height = y1 - y0
    pad_x = max(int(padding_pixels), int(np.ceil(padding_fraction * width)))
    pad_y = max(int(padding_pixels), int(np.ceil(padding_fraction * height)))
    return (
        max(0, x0 - pad_x),
        min(nx, x1 + pad_x),
        max(0, y0 - pad_y),
        min(ny, y1 + pad_y),
    )


def _read_normalized_roi_sum(file_path: Path, *, detector_number: int, roi: ROI) -> float:
    x0, x1, y0, y1 = (int(value) for value in roi)
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid ROI: {(x0, x1, y0, y1)}")

    data = read_detector_data(
        file_path,
        detector_number,
        normalize_by_monitor=True,
    )
    if x1 > data.shape[1] or y1 > data.shape[0]:
        raise ValueError(
            f"ROI {(x0, x1, y0, y1)} is outside detector{detector_number} data shape {data.shape}"
        )
    return float(np.sum(data[y0:y1, x0:x1], dtype=np.float64))


def compute_transmission(
    transmission_file: str | Path,
    empty_beam_transmission_file: str | Path,
    roi: ROI,
    *,
    detector_number: int = 0,
) -> float:
    """
    Compute sample transmission from two transmission images normalized by monitor.

    The ROI follows SCARLET's rectangle convention: ``(x0, x1, y0, y1)``
    with ``x1`` and ``y1`` exclusive.
    """
    transmission_file = Path(transmission_file).resolve()
    empty_beam_transmission_file = Path(empty_beam_transmission_file).resolve()
    detector_number = int(detector_number)
    if detector_number < 0:
        raise ValueError(f"Detector index must be >= 0, got {detector_number}")

    sample_value = _read_normalized_roi_sum(
        transmission_file,
        detector_number=detector_number,
        roi=roi,
    )
    empty_beam_value = _read_normalized_roi_sum(
        empty_beam_transmission_file,
        detector_number=detector_number,
        roi=roi,
    )
    if empty_beam_value <= 0.0:
        raise ValueError("Normalized empty-beam transmission ROI sum must be > 0")
    return sample_value / empty_beam_value


def compute_reference_transmissions(
    refs_file_path: str | Path,
    *,
    refs_entry_path: str = "/entry",
) -> dict[str, float]:
    """Reference-transmission workflow is currently disabled."""
    del refs_file_path, refs_entry_path
    raise NotImplementedError("compute_reference_transmissions is unavailable because scarlet.workflow.reference was removed")
