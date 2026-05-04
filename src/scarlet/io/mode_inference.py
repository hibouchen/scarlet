from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import h5py
import numpy as np


MeasurementMode = Literal["transmission", "scattering", "unknown"]


@dataclass(frozen=True)
class MeasurementModeGuess:
    mode: MeasurementMode
    confidence: float
    entry_path: str
    scores: Dict[str, float]
    reasons: List[str]


def guess_measurement_mode(
    file_path: str | Path,
    *,
    entry_path: str | None = "/entry",
    detector_index: int = 0,
    roi_half_size: int = 8,
) -> MeasurementModeGuess:
    """
    Guess whether a NeXus file corresponds to a transmission or scattering measurement.

    Heuristics used:
    1. explicit textual metadata if available
    2. filename hints
    3. attenuator hints
    4. beamstop hints
    5. image-based central ROI heuristic on detector0 (or chosen detector)

    Parameters
    ----------
    file_path:
        Path to the NeXus/HDF5 file.
    entry_path:
        Preferred entry path (default: /entry). Falls back to /entry0 or /entry1.
    detector_index:
        Detector index used for image-based heuristic.
    roi_half_size:
        Half-size of the square ROI around the beam center for the image heuristic.

    Returns
    -------
    MeasurementModeGuess
    """
    file_path = Path(file_path)

    scores = {"transmission": 0.0, "scattering": 0.0}
    reasons: List[str] = []

    with h5py.File(file_path, "r") as h5:
        resolved_entry = _resolve_entry_path(h5, entry_path)

        _apply_textual_metadata_heuristics(h5, resolved_entry, scores, reasons)
        _apply_filename_heuristics(file_path, scores, reasons)
        _apply_attenuator_heuristics(h5, resolved_entry, scores, reasons)
        _apply_beamstop_heuristics(h5, resolved_entry, scores, reasons)
        _apply_image_heuristics(
            h5,
            resolved_entry,
            detector_index=detector_index,
            roi_half_size=roi_half_size,
            scores=scores,
            reasons=reasons,
        )

    mode, confidence = _finalize_guess(scores)

    return MeasurementModeGuess(
        mode=mode,
        confidence=confidence,
        entry_path=resolved_entry,
        scores=dict(scores),
        reasons=reasons,
    )


def _resolve_entry_path(h5: h5py.File, preferred: str | None) -> str:
    if preferred and preferred in h5:
        return preferred
    for cand in ("/entry", "/entry0", "/entry1"):
        if cand in h5:
            return cand
    raise ValueError("No entry group found in file.")


def _as_str(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _as_str(value.item())
    return str(value)


def _read_scalar(h5: h5py.File, path: str):
    if path not in h5:
        return None
    try:
        return h5[path][()]
    except Exception:
        return None


def _text_contains_transmission(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("trans", "transmission", "attenuat"))


def _text_contains_scattering(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("scatt", "scatter", "diffus", "diffraction"))


def _apply_textual_metadata_heuristics(
    h5: h5py.File,
    entry: str,
    scores: Dict[str, float],
    reasons: List[str],
) -> None:
    candidate_paths = [
        f"{entry}/measurement_mode",
        f"{entry}/mode",
        f"{entry}/instrument/mode",
        f"{entry}/instrument/measurement_mode",
        f"{entry}/title",
        f"{entry}/sample/name",
    ]

    for path in candidate_paths:
        value = _read_scalar(h5, path)
        if value is None:
            continue
        text = _as_str(value)
        if _text_contains_transmission(text):
            scores["transmission"] += 6.0
            reasons.append(f"text hint at {path}: {text!r} -> transmission")
        elif _text_contains_scattering(text):
            scores["scattering"] += 6.0
            reasons.append(f"text hint at {path}: {text!r} -> scattering")


def _apply_filename_heuristics(
    file_path: Path,
    scores: Dict[str, float],
    reasons: List[str],
) -> None:
    name = file_path.name.lower()

    if any(k in name for k in ("trans", "transmission", "_tr_", "_tr.", "_t.")):
        scores["transmission"] += 2.0
        reasons.append(f"filename hint: {file_path.name!r} -> transmission")

    if any(k in name for k in ("scatt", "scatter", "diff", "sans")):
        scores["scattering"] += 1.5
        reasons.append(f"filename hint: {file_path.name!r} -> scattering")


def _apply_attenuator_heuristics(
    h5: h5py.File,
    entry: str,
    scores: Dict[str, float],
    reasons: List[str],
) -> None:
    candidate_paths = [
        f"{entry}/transmission_setup/attenuator/attenuation_factor",
        f"{entry}/instrument/attenuator/attenuation_factor",
        f"{entry}/instrument/attenuator/transmission",
    ]

    for path in candidate_paths:
        value = _read_scalar(h5, path)
        if value is None:
            continue
        try:
            x = float(value)
        except Exception:
            continue

        if path.endswith("attenuation_factor") and x > 1.0:
            scores["transmission"] += 3.0
            reasons.append(f"attenuator hint at {path}: factor={x:g} -> transmission")
        elif path.endswith("/transmission") and 0.0 < x < 1.0:
            scores["transmission"] += 3.0
            reasons.append(f"attenuator hint at {path}: transmission={x:g} -> transmission")


def _apply_beamstop_heuristics(
    h5: h5py.File,
    entry: str,
    scores: Dict[str, float],
    reasons: List[str],
) -> None:
    beamstop_group = f"{entry}/instrument/beamstop"
    if beamstop_group not in h5:
        return

    state_paths = [
        f"{beamstop_group}/state",
        f"{beamstop_group}/in_use",
        f"{beamstop_group}/inserted",
    ]

    found_state = False
    for path in state_paths:
        value = _read_scalar(h5, path)
        if value is None:
            continue
        found_state = True
        text = _as_str(value).strip().lower()

        if text in {"in", "inserted", "true", "1", "yes"}:
            scores["scattering"] += 2.5
            reasons.append(f"beamstop hint at {path}: {text!r} -> scattering")
        elif text in {"out", "removed", "false", "0", "no"}:
            scores["transmission"] += 2.5
            reasons.append(f"beamstop hint at {path}: {text!r} -> transmission")

    if not found_state:
        scores["scattering"] += 0.5
        reasons.append("beamstop group present but no explicit state -> weak scattering hint")


def _apply_image_heuristics(
    h5: h5py.File,
    entry: str,
    *,
    detector_index: int,
    roi_half_size: int,
    scores: Dict[str, float],
    reasons: List[str],
) -> None:
    det = f"{entry}/instrument/detector{detector_index}"
    data_path = f"{det}/data"
    bcx_path = f"{det}/beam_center_x"
    bcy_path = f"{det}/beam_center_y"

    if data_path not in h5 or bcx_path not in h5 or bcy_path not in h5:
        return

    try:
        data = np.asarray(h5[data_path][()], dtype=float)
        if data.ndim != 2:
            return

        bcx = float(h5[bcx_path][()])
        bcy = float(h5[bcy_path][()])
    except Exception:
        return

    ny, nx = data.shape
    cx = int(round(bcx))
    cy = int(round(bcy))

    if not (0 <= cx < nx and 0 <= cy < ny):
        return

    h = max(2, int(roi_half_size))
    x0 = max(0, cx - h)
    x1 = min(nx, cx + h + 1)
    y0 = max(0, cy - h)
    y1 = min(ny, cy + h + 1)

    center = data[y0:y1, x0:x1]
    if center.size == 0:
        return

    h2 = min(max(2 * h, h + 2), max(nx, ny))
    xx0 = max(0, cx - h2)
    xx1 = min(nx, cx + h2 + 1)
    yy0 = max(0, cy - h2)
    yy1 = min(ny, cy + h2 + 1)

    outer = data[yy0:yy1, xx0:xx1].copy()
    inner_y0 = y0 - yy0
    inner_y1 = inner_y0 + center.shape[0]
    inner_x0 = x0 - xx0
    inner_x1 = inner_x0 + center.shape[1]
    outer[inner_y0:inner_y1, inner_x0:inner_x1] = np.nan

    center_mean = float(np.nanmean(center))
    outer_mean = float(np.nanmean(outer))
    center_sum = float(np.nansum(center))
    total_sum = float(np.nansum(data))

    eps = 1e-12
    ratio = (center_mean + eps) / (outer_mean + eps)
    fraction = (center_sum + eps) / (total_sum + eps)

    if ratio > 5.0 and fraction > 0.02:
        scores["transmission"] += 4.0
        reasons.append(
            f"image hint: strong central beam on detector{detector_index} "
            f"(center/background ratio={ratio:.2f}, central fraction={fraction:.3f}) -> transmission"
        )
    elif ratio < 0.7:
        scores["scattering"] += 2.5
        reasons.append(
            f"image hint: central ROI suppressed on detector{detector_index} "
            f"(center/background ratio={ratio:.2f}) -> scattering"
        )
    elif ratio > 2.0:
        scores["transmission"] += 1.5
        reasons.append(
            f"image hint: moderately enhanced central ROI on detector{detector_index} "
            f"(center/background ratio={ratio:.2f}) -> weak transmission hint"
        )


def _finalize_guess(scores: Dict[str, float]) -> tuple[MeasurementMode, float]:
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