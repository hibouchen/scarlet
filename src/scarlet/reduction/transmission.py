from __future__ import annotations

from pathlib import Path
import numpy as np

from scarlet.io.nexus_reader import read_detector_data


ROI = tuple[int, int, int, int]


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
    """Backward-compatible wrapper around :mod:`scarlet.workflow.reference`."""
    from scarlet.workflow.reference import compute_reference_transmissions as _compute_reference_transmissions

    return _compute_reference_transmissions(refs_file_path, refs_entry_path=refs_entry_path)
