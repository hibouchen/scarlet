from __future__ import annotations

from pathlib import Path
import re
import h5py
import numpy as np


ROI = tuple[int, int, int, int]


def _resolve_entry_path(f: h5py.File) -> str:
    for entry_path in ("/raw_data", "/entry", "/entry0", "/entry1"):
        if entry_path in f and isinstance(f[entry_path], h5py.Group):
            return entry_path
    raise ValueError("No raw-data entry group found")


def _read_monitor_value(f: h5py.File, entry_path: str) -> float:
    dataset_path = f"{entry_path}/control/integral"
    if dataset_path not in f:
        raise ValueError(f"Missing monitor integral: {dataset_path}")
    try:
        monitor_value = float(np.asarray(f[dataset_path][()]).reshape(()))
    except Exception as exc:
        raise ValueError(f"Cannot read monitor integral: {dataset_path}") from exc
    if not np.isfinite(monitor_value) or monitor_value <= 0:
        raise ValueError(f"Monitor integral must be > 0: {dataset_path}")
    return monitor_value


def _require_same_shape(label: str, reference: np.ndarray, other: np.ndarray) -> None:
    if reference.shape != other.shape:
        raise ValueError(f"Shape mismatch for {label}: expected {reference.shape}, got {other.shape}")


def _read_reference_detector_image(
    refs_file: h5py.File,
    reference_name: str,
    *,
    detector_number: int,
    refs_entry_path: str = "/entry",
) -> Optional[np.ndarray]:
    reference_root = f"{refs_entry_path}/references/{reference_name}/entry"
    if reference_root not in refs_file or not isinstance(refs_file[reference_root], h5py.Group):
        return None

    data_path = f"{reference_root}/instrument/detector{detector_number}/data"
    if data_path not in refs_file:
        return None

    monitor_value = _read_monitor_value(refs_file, reference_root)
    data = np.asarray(refs_file[data_path][()], dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Reference detector data must be 2D, got shape {data.shape}")
    return data / monitor_value


def _read_reference_entry_path(
    refs_file: h5py.File,
    reference_name: str,
    *,
    refs_entry_path: str = "/entry",
) -> Optional[str]:
    entry_path = f"{refs_entry_path}/references/{reference_name}/entry"
    if entry_path in refs_file and isinstance(refs_file[entry_path], h5py.Group):
        return entry_path
    return None


def _read_transmission_roi(refs_file: h5py.File, *, refs_entry_path: str = "/entry") -> ROI:
    roi_root = f"{refs_entry_path}/transmission_roi"
    required_fields = ("x0", "x1", "y0", "y1")
    values: list[int] = []
    for field in required_fields:
        dataset_path = f"{roi_root}/{field}"
        if dataset_path not in refs_file:
            raise ValueError(f"Missing transmission ROI field: {dataset_path}")
        values.append(int(np.asarray(refs_file[dataset_path][()]).reshape(())))
    roi = tuple(values)
    x0, x1, y0, y1 = roi
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid transmission ROI in refs_norm: {roi}")
    return roi


def _read_transmission_roi_detector(refs_file: h5py.File, *, refs_entry_path: str = "/entry") -> int:
    dataset_path = f"{refs_entry_path}/transmission_roi/detector"
    if dataset_path not in refs_file:
        raise ValueError(f"Missing transmission ROI detector: {dataset_path}")

    raw = np.asarray(refs_file[dataset_path][()]).reshape(()).item()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode(errors="replace")
    if isinstance(raw, str):
        text = raw.strip()
        match = re.fullmatch(r"detector(\d+)", text, flags=re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
        return int(text)
    return int(raw)


def _roi_sum(data: np.ndarray, roi: ROI) -> float:
    x0, x1, y0, y1 = roi
    if data.ndim != 2:
        raise ValueError(f"Transmission ROI requires a 2D detector image, got ndim={data.ndim}")
    ny, nx = data.shape
    if not (0 <= x0 < x1 <= nx and 0 <= y0 < y1 <= ny):
        raise ValueError(f"Invalid ROI {roi} for detector image shape {data.shape}")
    value = float(np.nansum(data[y0:y1, x0:x1]))
    if not np.isfinite(value):
        raise ValueError(f"Non-finite ROI sum for ROI={roi}")
    return value


def _read_normalized_roi_sum(file_path: Path, *, detector_number: int, roi: ROI) -> float:
    x0, x1, y0, y1 = (int(value) for value in roi)
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid ROI: {(x0, x1, y0, y1)}")

    with h5py.File(file_path, "r") as f:
        entry_path = _resolve_entry_path(f)
        data_path = f"{entry_path}/instrument/detector{detector_number}/data"
        if data_path not in f:
            raise ValueError(f"Missing detector data: {data_path}")

        data = np.asarray(f[data_path][()], dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Detector data must be 2D, got shape {data.shape}")
        if x1 > data.shape[1] or y1 > data.shape[0]:
            raise ValueError(
                f"ROI {(x0, x1, y0, y1)} is outside detector{detector_number} data shape {data.shape}"
            )

        monitor_value = _read_monitor_value(f, entry_path)
        roi_sum = float(np.sum(data[y0:y1, x0:x1], dtype=np.float64))
        return roi_sum / monitor_value


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


def compute_corrected_water_scattering(
    ref_norm_file: str | Path,
    *,
    detector_number: int = 0,
    refs_entry_path: str = "/entry",
) -> np.ndarray:
    """
    Compute the corrected water-scattering image stored in a SCARLET refs_norm file.

    The computation uses monitor-normalized reference images copied in the bundle:

    ``water_scattering - dark - (empty_beam_scattering - dark)
       - T_water * (empty_cell_scattering - dark - (empty_beam_scattering - dark))``

    where ``T_water`` is computed from ``water_transmission`` and
    ``empty_beam_transmission`` using the transmission ROI stored in the refs file.
    """
    ref_norm_file = Path(ref_norm_file).resolve()
    detector_number = int(detector_number)
    if detector_number < 0:
        raise ValueError(f"Detector index must be >= 0, got {detector_number}")

    with h5py.File(ref_norm_file, "r") as f:
        definition_path = f"{refs_entry_path}/definition"
        if definition_path not in f:
            raise ValueError(f"Missing refs_norm definition: {definition_path}")
        definition = np.asarray(f[definition_path][()]).reshape(()).item()
        if isinstance(definition, (bytes, bytearray)):
            definition = definition.decode(errors="replace")
        if str(definition) != "SCARLET_refs_norm":
            raise ValueError(f"Unsupported reference bundle definition: {definition!r}")

        water_scattering = _read_reference_detector_image(
            f,
            "water_scattering",
            detector_number=detector_number,
            refs_entry_path=refs_entry_path,
        )
        if water_scattering is None:
            raise ValueError("refs_norm must contain a water_scattering reference")

        dark = _read_reference_detector_image(
            f,
            "dark",
            detector_number=detector_number,
            refs_entry_path=refs_entry_path,
        )
        if dark is None:
            dark = np.zeros_like(water_scattering, dtype=np.float64)
        _require_same_shape("dark", water_scattering, dark)

        empty_beam_scattering = _read_reference_detector_image(
            f,
            "empty_beam_scattering",
            detector_number=detector_number,
            refs_entry_path=refs_entry_path,
        )
        if empty_beam_scattering is not None:
            _require_same_shape("empty_beam_scattering", water_scattering, empty_beam_scattering)

        empty_cell_scattering = _read_reference_detector_image(
            f,
            "empty_cell_scattering",
            detector_number=detector_number,
            refs_entry_path=refs_entry_path,
        )
        if empty_cell_scattering is not None:
            _require_same_shape("empty_cell_scattering", water_scattering, empty_cell_scattering)

        water_transmission = _read_reference_detector_image(
            f,
            "water_transmission",
            detector_number=detector_number,
            refs_entry_path=refs_entry_path,
        )
        if water_transmission is None:
            raise ValueError("refs_norm must contain a water_transmission reference")
        _require_same_shape("water_transmission", water_scattering, water_transmission)

        empty_beam_transmission = _read_reference_detector_image(
            f,
            "empty_beam_transmission",
            detector_number=detector_number,
            refs_entry_path=refs_entry_path,
        )
        if empty_beam_transmission is None:
            raise ValueError("refs_norm must contain an empty_beam_transmission reference")
        _require_same_shape("empty_beam_transmission", water_scattering, empty_beam_transmission)

        roi = _read_transmission_roi(f, refs_entry_path=refs_entry_path)
        water_transmission_value = _roi_sum(water_transmission - dark, roi) / _roi_sum(
            empty_beam_transmission - dark,
            roi,
        )

        water_background_corrected = water_scattering - dark
        if empty_beam_scattering is not None:
            water_background_corrected = water_background_corrected - (empty_beam_scattering - dark)

        if empty_cell_scattering is None:
            return water_background_corrected

        empty_cell_corrected = empty_cell_scattering - dark
        if empty_beam_scattering is not None:
            empty_cell_corrected = empty_cell_corrected - (empty_beam_scattering - dark)
        return water_background_corrected - water_transmission_value * empty_cell_corrected


def _write_sample_transmission(entry_group: h5py.Group, value: float) -> None:
    sample_group = entry_group.require_group("sample")
    sample_group.attrs["NX_class"] = np.bytes_("NXsample")
    if "transmission" in sample_group:
        del sample_group["transmission"]
    sample_group.create_dataset("transmission", data=float(value))


def _compute_reference_transmission_value(
    refs_file: h5py.File,
    transmission_reference_name: str,
    *,
    detector_number: int,
    roi: ROI,
    refs_entry_path: str,
) -> float:
    transmission_image = _read_reference_detector_image(
        refs_file,
        transmission_reference_name,
        detector_number=detector_number,
        refs_entry_path=refs_entry_path,
    )
    if transmission_image is None:
        raise ValueError(f"Missing {transmission_reference_name} reference")

    empty_beam_transmission = _read_reference_detector_image(
        refs_file,
        "empty_beam_transmission",
        detector_number=detector_number,
        refs_entry_path=refs_entry_path,
    )
    if empty_beam_transmission is None:
        raise ValueError("Missing empty_beam_transmission reference")
    _require_same_shape("empty_beam_transmission", transmission_image, empty_beam_transmission)

    dark = _read_reference_detector_image(
        refs_file,
        "dark",
        detector_number=detector_number,
        refs_entry_path=refs_entry_path,
    )
    if dark is None:
        dark = np.zeros_like(transmission_image, dtype=np.float64)
    _require_same_shape("dark", transmission_image, dark)

    empty_beam_roi_sum = _roi_sum(empty_beam_transmission - dark, roi)
    if empty_beam_roi_sum == 0.0:
        raise ValueError("Cannot compute transmission: empty-beam ROI sum is zero")
    return _roi_sum(transmission_image - dark, roi) / empty_beam_roi_sum


def compute_reference_transmissions(
    refs_file_path: str | Path,
    *,
    refs_entry_path: str = "/entry",
) -> dict[str, float]:
    """
    Compute and store the transmission of reference measurements embedded in a refs bundle.

    The computed scalar is written in-place to ``/sample/transmission`` inside the
    copied reference entries. For each transmission reference, the same scalar is also
    written to the associated scattering reference when present.
    """
    refs_file_path = Path(refs_file_path).resolve()
    updated: dict[str, float] = {}

    with h5py.File(refs_file_path, "r+") as refs_file:
        definition_path = f"{refs_entry_path}/definition"
        if definition_path not in refs_file:
            raise ValueError(f"Missing refs definition: {definition_path}")
        definition = np.asarray(refs_file[definition_path][()]).reshape(()).item()
        if isinstance(definition, (bytes, bytearray)):
            definition = definition.decode(errors="replace")
        definition = str(definition)
        if definition not in {"SCARLET_refs_sub", "SCARLET_refs_norm"}:
            raise ValueError(f"Unsupported reference bundle definition: {definition!r}")

        roi = _read_transmission_roi(refs_file, refs_entry_path=refs_entry_path)
        detector_number = _read_transmission_roi_detector(refs_file, refs_entry_path=refs_entry_path)

        if _read_reference_entry_path(refs_file, "empty_beam_transmission", refs_entry_path=refs_entry_path) is None:
            raise ValueError("refs bundle must contain empty_beam_transmission to compute reference transmissions")

        reference_specs: list[tuple[str, Optional[str], float]] = [
            ("empty_beam_transmission", "empty_beam_scattering", 1.0),
        ]
        for transmission_name, scattering_name in (
            ("empty_cell_transmission", "empty_cell_scattering"),
            ("water_transmission", "water_scattering"),
        ):
            if _read_reference_entry_path(refs_file, transmission_name, refs_entry_path=refs_entry_path) is None:
                continue
            value = _compute_reference_transmission_value(
                refs_file,
                transmission_name,
                detector_number=detector_number,
                roi=roi,
                refs_entry_path=refs_entry_path,
            )
            reference_specs.append((transmission_name, scattering_name, value))

        for transmission_name, scattering_name, value in reference_specs:
            transmission_entry_path = _read_reference_entry_path(
                refs_file,
                transmission_name,
                refs_entry_path=refs_entry_path,
            )
            if transmission_entry_path is not None:
                _write_sample_transmission(refs_file[transmission_entry_path], value)
                updated[transmission_name] = value

            if scattering_name is None:
                continue
            scattering_entry_path = _read_reference_entry_path(
                refs_file,
                scattering_name,
                refs_entry_path=refs_entry_path,
            )
            if scattering_entry_path is not None:
                _write_sample_transmission(refs_file[scattering_entry_path], value)
                updated[scattering_name] = value

    return updated
