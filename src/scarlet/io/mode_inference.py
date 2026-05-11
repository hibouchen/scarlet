from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import h5py
import numpy as np


MeasurementMode = Literal["transmission", "scattering", "unknown"]


@dataclass(frozen=True)
class ImageModeGuess:
    mode: MeasurementMode
    confidence: float
    scores: Dict[str, float]
    reasons: List[str]


def guess_measurement_mode_from_image(
    data: np.ndarray,
    *,
    center: Optional[Tuple[float, float]] = None,
    center_half_size: int = 1,
    ring_half_size: int = 4,
    local_half_size: int = 10,
) -> ImageModeGuess:
    """
    Guess measurement mode using only a 2D detector image.

    Strategy:
    - use the geometrical image center unless an explicit center is provided
    - compare a tiny center ROI to a surrounding ring
    - measure the offset of the brightest local peak

    Interpretation:
    - dark center + off-centered bright halo -> scattering
    - bright center + peak near center -> transmission
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("guess_measurement_mode_from_image expects a 2D array")

    ny, nx = data.shape

    if center is None:
        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0
    else:
        cx, cy = center

    cx = int(round(cx))
    cy = int(round(cy))

    if not (0 <= cx < nx and 0 <= cy < ny):
        raise ValueError("Center is outside the image")

    scores = {"transmission": 0.0, "scattering": 0.0}
    reasons: List[str] = []

    # --- tiny center ROI
    c0x = max(0, cx - center_half_size)
    c1x = min(nx, cx + center_half_size + 1)
    c0y = max(0, cy - center_half_size)
    c1y = min(ny, cy + center_half_size + 1)
    center_roi = data[c0y:c1y, c0x:c1x]

    # --- ring box around the center
    r0x = max(0, cx - ring_half_size)
    r1x = min(nx, cx + ring_half_size + 1)
    r0y = max(0, cy - ring_half_size)
    r1y = min(ny, cy + ring_half_size + 1)
    ring_box = data[r0y:r1y, r0x:r1x].copy()

    inner_y0 = c0y - r0y
    inner_y1 = inner_y0 + center_roi.shape[0]
    inner_x0 = c0x - r0x
    inner_x1 = inner_x0 + center_roi.shape[1]
    ring_box[inner_y0:inner_y1, inner_x0:inner_x1] = np.nan

    # --- local box for peak search
    h = max(local_half_size, ring_half_size + 2)
    b0x = max(0, cx - h)
    b1x = min(nx, cx + h + 1)
    b0y = max(0, cy - h)
    b1y = min(ny, cy + h + 1)
    local_box = data[b0y:b1y, b0x:b1x]

    center_mean = float(np.nanmean(center_roi))
    ring_mean = float(np.nanmean(ring_box))
    peak_idx = np.unravel_index(np.nanargmax(local_box), local_box.shape)
    peak_y = b0y + int(peak_idx[0])
    peak_x = b0x + int(peak_idx[1])
    peak_value = float(np.nanmax(local_box))
    peak_dist = float(np.hypot(peak_x - cx, peak_y - cy))

    eps = 1e-12
    center_vs_ring = (center_mean + eps) / (ring_mean + eps)
    center_vs_peak = (center_mean + eps) / (peak_value + eps)

    # --------------------------------------------------
    # Scattering:
    # very dark center + peak shifted away from center
    # --------------------------------------------------
    if center_vs_ring < 0.45 and peak_dist >= 2.5:
        scores["scattering"] += 4.0
        reasons.append(
            f"dark center with offset bright region "
            f"(center/ring={center_vs_ring:.2f}, peak offset={peak_dist:.1f} px)"
        )

    if center_vs_peak < 0.20 and peak_dist >= 3.0:
        scores["scattering"] += 2.0
        reasons.append(
            f"center strongly suppressed relative to local peak "
            f"(center/peak={center_vs_peak:.2f}, peak offset={peak_dist:.1f} px)"
        )

    # --------------------------------------------------
    # Transmission:
    # bright center + peak close to center
    # --------------------------------------------------
    if center_vs_ring > 1.15 and peak_dist <= 5.0:
        scores["transmission"] += 4.0
        reasons.append(
            f"bright center compatible with direct beam "
            f"(center/ring={center_vs_ring:.2f}, peak offset={peak_dist:.1f} px)"
        )

    if center_vs_peak > 0.45 and peak_dist <= 6.0:
        scores["transmission"] += 1.5
        reasons.append(
            f"center remains bright compared to peak "
            f"(center/peak={center_vs_peak:.2f}, peak offset={peak_dist:.1f} px)"
        )

    mode, confidence = _finalize_image_guess(scores)

    return ImageModeGuess(
        mode=mode,
        confidence=confidence,
        scores=scores,
        reasons=reasons,
    )


def guess_measurement_mode_from_nexus_image(
    file_path: str | Path,
    *,
    entry_path: str | None = "/raw_data",
    detector_index: int = 0,
    center_half_size: int = 1,
    ring_half_size: int = 4,
    local_half_size: int = 10,
) -> ImageModeGuess:
    """
    Guess measurement mode using only the detector image stored in a NeXus file.

    This function:
    1. resolves the entry path (/raw_data, with /entry, /entry0, /entry1 fallbacks),
    2. reads <entry>/instrument/detectorN/data,
    3. calls guess_measurement_mode_from_image(data).
    """
    file_path = Path(file_path)

    with h5py.File(file_path, "r") as h5:
        resolved_entry = _resolve_entry_path(h5, entry_path)
        data_path = f"{resolved_entry}/instrument/detector{detector_index}/data"

        if data_path not in h5:
            raise ValueError(f"Missing detector image dataset: {data_path}")

        data = np.asarray(h5[data_path][()], dtype=float)

    return guess_measurement_mode_from_image(
        data,
        center=None,
        center_half_size=center_half_size,
        ring_half_size=ring_half_size,
        local_half_size=local_half_size,
    )


def _resolve_entry_path(h5: h5py.File, preferred: str | None) -> str:
    if preferred and preferred in h5:
        return preferred
    for cand in ("/raw_data", "/entry", "/entry0", "/entry1"):
        if cand in h5:
            return cand
    raise ValueError("No entry group found in file.")


def _finalize_image_guess(scores: Dict[str, float]) -> Tuple[MeasurementMode, float]:
    t = scores["transmission"]
    s = scores["scattering"]

    total = t + s
    if total <= 0.0:
        return "unknown", 0.0

    delta = abs(t - s)
    confidence = max(0.0, min(1.0, delta / total))

    if delta < 1.0:
        return "unknown", confidence

    if t > s:
        return "transmission", confidence
    if s > t:
        return "scattering", confidence
    return "unknown", confidence