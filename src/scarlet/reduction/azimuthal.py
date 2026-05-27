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
class DetectorAzimuthalCurve:
    """Azimuthally averaged I(Q) for one detector."""

    detector_index: int
    q: np.ndarray
    intensity: np.ndarray
    n_pixels: np.ndarray
    q_edges: np.ndarray


@dataclass(frozen=True)
class AzimuthalAverageResult:
    """Azimuthally averaged I(Q) loaded or computed from one reduced NeXus file."""

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
        if not isinstance(group, h5py.Group):
            continue
        if ("I" in group and "Qx" in group and "Qy" in group) or ("I" in group and "Q" in group):
            indices.append(int(match.group(1)))
    return sorted(indices)


def _detector_group_kind(data_group: h5py.Group) -> Optional[str]:
    if "I" in data_group and "Q" in data_group:
        return "azimuthal_1d"
    if "I" in data_group and "Qx" in data_group and "Qy" in data_group:
        return "legacy_2d"
    return None


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


def flatten_detector_iq(intensity: np.ndarray, qx: np.ndarray, qy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a reduced detector image into per-pixel (Q, I) samples."""
    intensity = np.asarray(intensity, dtype=np.float64)
    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)
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


def _flatten_detector_iq(data_group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    return flatten_detector_iq(
        np.asarray(data_group["I"][()], dtype=np.float64),
        np.asarray(data_group["Qx"][()], dtype=np.float64),
        np.asarray(data_group["Qy"][()], dtype=np.float64),
    )


def resolve_azimuthal_q_range(
    q_values: np.ndarray,
    *,
    q_min: Optional[float] = None,
    q_max: Optional[float] = None,
) -> tuple[float, float]:
    """Resolve a valid Q range, expanding a degenerate auto-range if needed."""
    q_values = np.asarray(q_values, dtype=np.float64)
    if q_values.size == 0:
        raise ValueError("No finite reduced pixels available for azimuthal averaging")

    auto_min = q_min is None
    auto_max = q_max is None
    if auto_min:
        q_min = float(np.nanmin(q_values))
    if auto_max:
        q_max = float(np.nanmax(q_values))

    q_min = float(q_min)
    q_max = float(q_max)
    if auto_min and auto_max and q_max == q_min:
        half_width = max(abs(q_min) * 1.0e-6, 1.0e-12)
        q_min -= half_width
        q_max += half_width

    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max <= q_min:
        raise ValueError(f"Invalid q range: q_min={q_min!r}, q_max={q_max!r}")
    return q_min, q_max


def _bin_azimuthal_samples(
    q_values: np.ndarray,
    intensity_values: np.ndarray,
    *,
    n_bins: int,
    q_min: Optional[float] = None,
    q_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_values = np.asarray(q_values, dtype=np.float64)
    intensity_values = np.asarray(intensity_values, dtype=np.float64)
    if q_values.shape != intensity_values.shape:
        raise ValueError(f"Inconsistent azimuthal samples: q shape {q_values.shape}, I shape {intensity_values.shape}")

    q_min, q_max = resolve_azimuthal_q_range(q_values, q_min=q_min, q_max=q_max)
    in_range = (q_values >= q_min) & (q_values <= q_max)
    q_values = q_values[in_range]
    intensity_values = intensity_values[in_range]
    if q_values.size == 0:
        raise ValueError("No pixels fall within the requested q range")

    q_edges = np.linspace(q_min, q_max, int(n_bins) + 1, dtype=np.float64)
    bin_index = np.searchsorted(q_edges, q_values, side="right") - 1
    bin_index = np.clip(bin_index, 0, int(n_bins) - 1)

    n_pixels = np.bincount(bin_index, minlength=int(n_bins)).astype(np.int64)
    summed_intensity = np.bincount(bin_index, weights=intensity_values, minlength=int(n_bins))
    intensity = np.full(int(n_bins), np.nan, dtype=np.float64)
    valid_bins = n_pixels > 0
    intensity[valid_bins] = summed_intensity[valid_bins] / n_pixels[valid_bins]
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    return q_centers, intensity, n_pixels, q_edges


def azimuthal_average_from_arrays(
    intensity: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    *,
    detector_index: int,
    n_bins: int = 200,
    q_min: Optional[float] = None,
    q_max: Optional[float] = None,
) -> DetectorAzimuthalCurve:
    """Compute a binned azimuthal curve from one reduced 2D detector image."""
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    q_values, intensity_values = flatten_detector_iq(intensity, qx, qy)
    q, averaged, n_pixels, q_edges = _bin_azimuthal_samples(
        q_values,
        intensity_values,
        n_bins=n_bins,
        q_min=q_min,
        q_max=q_max,
    )
    return DetectorAzimuthalCurve(
        detector_index=int(detector_index),
        q=q,
        intensity=averaged,
        n_pixels=n_pixels,
        q_edges=q_edges,
    )


def _infer_q_edges(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError(f"Stored azimuthal Q axis must be 1D, got ndim={q.ndim}")
    if q.size == 0:
        raise ValueError("Stored azimuthal Q axis is empty")
    if q.size == 1:
        half_width = max(abs(float(q[0])) * 1.0e-6, 1.0e-12)
        return np.array([float(q[0]) - half_width, float(q[0]) + half_width], dtype=np.float64)

    midpoints = 0.5 * (q[:-1] + q[1:])
    edges = np.empty(q.size + 1, dtype=np.float64)
    edges[1:-1] = midpoints
    edges[0] = q[0] - (midpoints[0] - q[0])
    edges[-1] = q[-1] + (q[-1] - midpoints[-1])
    return edges


def _read_azimuthal_curve(data_group: h5py.Group, *, detector_index: int) -> DetectorAzimuthalCurve:
    q = np.asarray(data_group["Q"][()], dtype=np.float64)
    intensity = np.asarray(data_group["I"][()], dtype=np.float64)
    if q.ndim != 1 or intensity.ndim != 1:
        raise ValueError("Stored azimuthal NXdata must contain 1D I and Q datasets")
    if q.shape != intensity.shape:
        raise ValueError(f"Inconsistent stored azimuthal curve: Q shape {q.shape}, I shape {intensity.shape}")

    if "n_pixels" in data_group:
        n_pixels = np.asarray(data_group["n_pixels"][()], dtype=np.int64)
    else:
        n_pixels = np.ones(q.shape, dtype=np.int64)
    if n_pixels.ndim != 1 or n_pixels.shape != q.shape:
        raise ValueError(f"Inconsistent stored n_pixels shape: expected {q.shape}, got {n_pixels.shape}")

    if "Q_edges" in data_group:
        q_edges = np.asarray(data_group["Q_edges"][()], dtype=np.float64)
        if q_edges.ndim != 1 or q_edges.size != q.size + 1:
            raise ValueError(f"Inconsistent stored Q_edges shape: expected {(q.size + 1,)}, got {q_edges.shape}")
    else:
        q_edges = _infer_q_edges(q)

    return DetectorAzimuthalCurve(
        detector_index=int(detector_index),
        q=q,
        intensity=intensity,
        n_pixels=n_pixels,
        q_edges=q_edges,
    )


def _merge_azimuthal_curves(curves: Sequence[DetectorAzimuthalCurve]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not curves:
        raise ValueError("No detector curves available for merging")

    ref_q = curves[0].q
    ref_q_edges = curves[0].q_edges
    total_pixels = np.zeros(ref_q.shape, dtype=np.int64)
    weighted_sum = np.zeros(ref_q.shape, dtype=np.float64)

    for curve in curves:
        if curve.q.shape != ref_q.shape or not np.allclose(curve.q, ref_q, rtol=0.0, atol=1.0e-12):
            raise ValueError("Selected detector curves do not share the same Q bins yet")
        if curve.q_edges.shape != ref_q_edges.shape or not np.allclose(curve.q_edges, ref_q_edges, rtol=0.0, atol=1.0e-12):
            raise ValueError("Selected detector curves do not share the same Q edges yet")

        valid = (curve.n_pixels > 0) & np.isfinite(curve.intensity)
        total_pixels += np.where(valid, curve.n_pixels, 0)
        weighted_sum += np.where(valid, curve.intensity * curve.n_pixels, 0.0)

    merged_intensity = np.full(ref_q.shape, np.nan, dtype=np.float64)
    valid_bins = total_pixels > 0
    merged_intensity[valid_bins] = weighted_sum[valid_bins] / total_pixels[valid_bins]
    return ref_q, merged_intensity, total_pixels, ref_q_edges


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
    Load or compute an azimuthal average from one reduced SCARLET file.

    Legacy reduced files are rebinned from ``(I, Qx, Qy)`` detector images.
    Newer reduced files can be read directly from per-detector azimuthal
    ``NXdata`` groups containing ``(I, Q)`` curves.
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

        detector_kinds = {
            detector_index: _detector_group_kind(entry[f"data{detector_index}"])
            for detector_index in selected
        }
        kinds = {kind for kind in detector_kinds.values() if kind is not None}
        if len(kinds) != 1:
            raise ValueError("Selected detector groups do not share a consistent reduced-data format")

        kind = kinds.pop()
        if kind == "azimuthal_1d":
            curves = [
                _read_azimuthal_curve(entry[f"data{detector_index}"], detector_index=detector_index)
                for detector_index in selected
            ]
            q_centers, intensity, n_pixels, q_edges = _merge_azimuthal_curves(curves)
            return AzimuthalAverageResult(
                q=q_centers,
                intensity=intensity,
                n_pixels=n_pixels,
                q_edges=q_edges,
                detector_indices=selected,
                input_file=file_path.resolve(),
                processed_entry=processed_entry,
            )

        all_q: list[np.ndarray] = []
        all_i: list[np.ndarray] = []
        for detector_index in selected:
            q_flat, i_flat = _flatten_detector_iq(entry[f"data{detector_index}"])
            all_q.append(q_flat)
            all_i.append(i_flat)

    q_values = np.concatenate(all_q) if all_q else np.empty(0, dtype=np.float64)
    intensity_values = np.concatenate(all_i) if all_i else np.empty(0, dtype=np.float64)
    q_centers, intensity, n_pixels, q_edges = _bin_azimuthal_samples(
        q_values,
        intensity_values,
        n_bins=int(n_bins),
        q_min=q_min,
        q_max=q_max,
    )

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
