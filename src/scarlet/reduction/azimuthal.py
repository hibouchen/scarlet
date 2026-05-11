from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import re
from typing import Optional, Sequence, Union

import h5py
import numpy as np

from .nexus import PROCESSED_ENTRY_PATH, as_hdf5_path


@dataclass(frozen=True)
class AzimuthalAverageResult:
    """Azimuthally averaged I(Q) from one reduced 2D NeXus file."""

    q: np.ndarray
    intensity: np.ndarray
    n_pixels: np.ndarray
    q_edges: np.ndarray
    detector_indices: list[int]
    input_file: Path
    processed_entry: str


def _available_detector_indices(entry: h5py.Group) -> list[int]:
    indices: list[int] = []
    for name in entry.keys():
        match = re.fullmatch(r"data(\d+)", name)
        if match is None:
            continue
        group = entry[name]
        if isinstance(group, h5py.Group) and "I" in group and "Qx" in group and "Qy" in group:
            indices.append(int(match.group(1)))
    return sorted(indices)


def _resolve_detector_indices(
    entry: h5py.Group,
    detector_indices: Optional[Sequence[int]],
) -> list[int]:
    available = _available_detector_indices(entry)
    if detector_indices is None:
        if not available:
            raise ValueError("No reduced detector groups found under the processed entry")
        return available

    requested = sorted(dict.fromkeys(int(i) for i in detector_indices))
    missing = [i for i in requested if i not in available]
    if missing:
        available_text = ", ".join(f"data{i}" for i in available) or "<none>"
        missing_text = ", ".join(f"data{i}" for i in missing)
        raise ValueError(f"Missing reduced detector groups: {missing_text}. Available groups: {available_text}")
    return requested


def _flatten_detector_iq(data_group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    intensity = np.asarray(data_group["I"][()], dtype=np.float64)
    qx = np.asarray(data_group["Qx"][()], dtype=np.float64)
    qy = np.asarray(data_group["Qy"][()], dtype=np.float64)
    if intensity.ndim != 2:
        raise ValueError(f"Reduced detector image must be 2D, got ndim={intensity.ndim}")
    if qx.ndim != 1 or qy.ndim != 1:
        raise ValueError("Qx and Qy axes must be 1D")
    if intensity.shape != (qy.size, qx.size):
        raise ValueError(
            f"Inconsistent reduced detector geometry: I shape {intensity.shape}, "
            f"Qx size {qx.size}, Qy size {qy.size}"
        )

    q = np.sqrt(qy[:, None] * qy[:, None] + qx[None, :] * qx[None, :])
    valid = np.isfinite(intensity) & np.isfinite(q)
    return q[valid].reshape(-1), intensity[valid].reshape(-1)


def azimuthal_average(
    file_path: Union[str, Path],
    *,
    detector_indices: Optional[Sequence[int]] = None,
    processed_entry: str = PROCESSED_ENTRY_PATH,
    n_bins: int = 200,
    q_min: Optional[float] = None,
    q_max: Optional[float] = None,
) -> AzimuthalAverageResult:
    """
    Compute a simple azimuthal average from one reduced SCARLET 2D file.

    Intensities are grouped by ``Q = sqrt(Qx^2 + Qy^2)`` into linear bins.
    Pixels with non-finite intensity or Q are ignored.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    file_path = Path(file_path)
    processed_entry = as_hdf5_path(processed_entry)

    with h5py.File(file_path, "r") as f:
        if processed_entry not in f or not isinstance(f[processed_entry], h5py.Group):
            raise ValueError(f"Processed entry not found: {processed_entry}")
        entry = f[processed_entry]
        selected = _resolve_detector_indices(entry, detector_indices)

        all_q: list[np.ndarray] = []
        all_i: list[np.ndarray] = []
        for detector_index in selected:
            q_flat, i_flat = _flatten_detector_iq(entry[f"data{detector_index}"])
            all_q.append(q_flat)
            all_i.append(i_flat)

    q_values = np.concatenate(all_q) if all_q else np.empty(0, dtype=np.float64)
    intensity_values = np.concatenate(all_i) if all_i else np.empty(0, dtype=np.float64)
    if q_values.size == 0:
        raise ValueError("No finite reduced pixels available for azimuthal averaging")

    if q_min is None:
        q_min = float(np.nanmin(q_values))
    if q_max is None:
        q_max = float(np.nanmax(q_values))
    q_min = float(q_min)
    q_max = float(q_max)
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max <= q_min:
        raise ValueError(f"Invalid q range: q_min={q_min!r}, q_max={q_max!r}")

    in_range = (q_values >= q_min) & (q_values <= q_max)
    q_values = q_values[in_range]
    intensity_values = intensity_values[in_range]
    if q_values.size == 0:
        raise ValueError("No pixels fall within the requested q range")

    q_edges = np.linspace(q_min, q_max, int(n_bins) + 1, dtype=np.float64)
    # searchsorted(..., side="right") puts exact-edge values in the bin on the right;
    # subtract 1 to convert edges to 0-based bin indices and clamp q == q_max into the last bin.
    bin_index = np.searchsorted(q_edges, q_values, side="right") - 1
    bin_index = np.clip(bin_index, 0, int(n_bins) - 1)

    n_pixels = np.bincount(bin_index, minlength=int(n_bins)).astype(np.int64)
    summed_intensity = np.bincount(bin_index, weights=intensity_values, minlength=int(n_bins))
    intensity = np.full(int(n_bins), np.nan, dtype=np.float64)
    valid_bins = n_pixels > 0
    intensity[valid_bins] = summed_intensity[valid_bins] / n_pixels[valid_bins]
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])

    return AzimuthalAverageResult(
        q=q_centers,
        intensity=intensity,
        n_pixels=n_pixels,
        q_edges=q_edges,
        detector_indices=selected,
        input_file=file_path.resolve(),
        processed_entry=processed_entry,
    )


def write_azimuthal_average_csv(
    output_path: Union[str, Path],
    result: AzimuthalAverageResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Write an azimuthal-average result as a simple CSV table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}")

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["q_A^-1", "I", "n_pixels"])
        for q_value, intensity, n_pixels in zip(result.q, result.intensity, result.n_pixels):
            writer.writerow([f"{float(q_value):.16g}", f"{float(intensity):.16g}", int(n_pixels)])

    return output_path
