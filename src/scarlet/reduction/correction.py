from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import scipp as sc


def _as_image(image: Any, *, name: str) -> np.ndarray:
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"{name} must be a 2D image, got shape {data.shape}")
    return data


def _require_positive_scalar(value: float, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def _require_same_shape(reference: np.ndarray, other: np.ndarray, *, name: str) -> None:
    if reference.shape != other.shape:
        raise ValueError(f"{name} shape mismatch: expected {reference.shape}, got {other.shape}")


def _require_scipp():
    """Import Scipp lazily so callers only pay the dependency when needed."""
    try:
        import scipp as sc
    except ImportError as exc:
        raise ImportError("scipp is required to use DataArray-based correction helpers") from exc
    return sc


def _require_dataarray(value: Any, *, name: str, ndim: int | None = None) -> "sc.DataArray":
    sc = _require_scipp()
    if not isinstance(value, sc.DataArray):
        raise TypeError(f"{name} must be a scipp.DataArray, got {type(value).__name__}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D DataArray, got shape {value.shape}")
    return value


def _require_positive_scalar_dataarray(value: Any, *, name: str) -> "sc.DataArray":
    data_array = _require_dataarray(value, name=name, ndim=0)
    scalar = float(np.asarray(data_array.data.values, dtype=np.float64).reshape(()))
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be > 0, got {scalar!r}")
    return data_array


def _require_same_dataarray_shape(reference: "sc.DataArray", other: "sc.DataArray", *, name: str) -> None:
    if reference.dims != other.dims or reference.shape != other.shape:
        raise ValueError(f"{name} shape mismatch: expected {reference.shape}, got {other.shape}")


def _scalar_data_without_variance(data_array: "sc.DataArray", *, name: str):
    sc = _require_scipp()
    scalar = float(np.asarray(data_array.data.values, dtype=np.float64).reshape(()))
    return sc.scalar(scalar, unit=data_array.data.unit)


def _normalize_scalar_transmission(value: Any, *, name: str):
    sc = _require_scipp()
    if isinstance(value, sc.DataArray):
        return _scalar_data_without_variance(
            _require_positive_scalar_dataarray(value, name=name),
            name=name,
        )

    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be > 0, got {scalar!r}")
    return sc.scalar(scalar)


def normalize_by_monitor(
    image: Any,
    monitor: float,
) -> np.ndarray:
    """Normalize a detector image by its monitor integral."""
    data = np.asarray(image, dtype=np.float64)
    monitor = float(monitor)

    if not np.isfinite(monitor) or monitor <= 0.0:
        raise ValueError(f"monitor must be > 0, got {monitor!r}")

    return data / monitor


def normalize_by_solid_angle(
    image: Any,
    detector_distance: float,
    beam_center: tuple[float, float],
    pixel_size: float | tuple[float, float],
) -> np.ndarray:
    """
    Normalize a detector image by the solid angle subtended by each pixel.

    The detector is assumed to be a flat plane perpendicular to the direct beam.
    The returned image is in counts per steradian.
    """
    data = _as_image(image, name="image")
    distance = _require_positive_scalar(detector_distance, name="detector_distance")

    beam_center_x = float(beam_center[0])
    beam_center_y = float(beam_center[1])
    if not np.isfinite(beam_center_x) or not np.isfinite(beam_center_y):
        raise ValueError(f"beam_center must contain finite coordinates, got {beam_center!r}")

    if isinstance(pixel_size, tuple):
        pixel_size_x = _require_positive_scalar(pixel_size[0], name="pixel_size_x")
        pixel_size_y = _require_positive_scalar(pixel_size[1], name="pixel_size_y")
    else:
        pixel_size_x = _require_positive_scalar(pixel_size, name="pixel_size")
        pixel_size_y = pixel_size_x

    ny, nx = data.shape
    x = (np.arange(nx, dtype=np.float64) - beam_center_x) * pixel_size_x
    y = (np.arange(ny, dtype=np.float64) - beam_center_y) * pixel_size_y
    xx, yy = np.meshgrid(x, y)
    tth = np.arctan2(np.sqrt(xx * xx + yy * yy), distance)
    solid_angle = (pixel_size_x * pixel_size_y) / np.power(distance, 2) * np.cos(tth) ** 3
    return data / solid_angle


def correct_detector_dead_time(
    image: Any,
    acq_time: float,
    deadtime: float,
) -> np.ndarray:
    """
    Correct a detector image for non-paralyzable detector dead time.

    The correction is applied on the count rate:

    ``I_corrected = (I) / (1 - (I / acq_time) * deadtime)``

    where ``I`` is the measured image in counts, ``acq_time`` the acquisition
    time in seconds, and ``deadtime`` the detector dead time in seconds.
    """
    data = np.asarray(image, dtype=np.float64)
    acq_time = float(acq_time)
    deadtime = float(deadtime)

    if not np.isfinite(acq_time) or acq_time <= 0.0:
        raise ValueError(f"acq_time must be > 0, got {acq_time!r}")
    if not np.isfinite(deadtime) or deadtime < 0.0:
        raise ValueError(f"deadtime must be >= 0, got {deadtime!r}")

    rate = data / acq_time
    denominator = 1.0 - (rate * deadtime)
    if np.any(denominator <= 0.0):
        raise ValueError("dead-time correction is undefined when 1 - rate * deadtime <= 0")

    return rate / denominator


def subtract_scattering_references(
    sample: Any,
    transmission: Any,
    *,
    dark: Any | None = None,
    empty_cell: Any | None = None,
    empty_cell_transmission: Any | None = None,
    empty_beam: Any | None = None,
    empty_beam_transmission: Any | None = None,
    distance: float | None = None,
    beam_center: tuple[float, float] | None = None,
) -> "sc.DataArray":
    """
    Apply the current reference subtraction model:

    ``I = (sample - dark) / T_sample - (empty_cell - dark) / T_empty_cell``

    `empty_beam`, `empty_beam_transmission`, `distance` and `beam_center` are
    intentionally kept in the API for future extensions and are not used yet.
    """
    del distance, beam_center

    sample = _require_dataarray(sample, name="sample", ndim=2)
    transmission = _normalize_scalar_transmission(transmission, name="transmission")

    dark_image = sample * 0.0
    if dark is not None:
        dark_image = _require_dataarray(dark, name="dark", ndim=2)
        _require_same_dataarray_shape(sample, dark_image, name="dark")

    if empty_beam is not None:
        empty_beam = _require_dataarray(empty_beam, name="empty_beam", ndim=2)
        _require_same_dataarray_shape(sample, empty_beam, name="empty_beam")
    if empty_beam_transmission is not None:
        pass

    result = (sample - dark_image) / transmission

    if empty_cell is None:
        return result

    if empty_cell_transmission is None:
        raise ValueError("empty_cell_transmission is required when empty_cell is provided")
    empty_cell_transmission = _normalize_scalar_transmission(
        empty_cell_transmission,
        name="empty_cell_transmission",
    )

    empty_cell = _require_dataarray(empty_cell, name="empty_cell", ndim=2)
    _require_same_dataarray_shape(sample, empty_cell, name="empty_cell")
    return result - (empty_cell - dark_image) / empty_cell_transmission
