from __future__ import annotations

from typing import Optional

import h5py

from ._hdf import as_float_scalar, as_str


MM_TO_M = 1e-3


def mm_to_m(value) -> float:
    return as_float_scalar(value) * MM_TO_M


def mm_or_m_to_m(value) -> float:
    if value is None:
        return float("nan")
    v = as_float_scalar(value)
    if abs(v) <= 5.0:
        return v
    return v * MM_TO_M


def length_dataset_to_m(ds: Optional[h5py.Dataset]) -> Optional[float]:
    if ds is None:
        return None
    units = ds.attrs.get("units")
    units_s = as_str(units).strip().lower() if units is not None else ""
    value = ds[()]
    if units_s in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        return as_float_scalar(value) * MM_TO_M
    if units_s in {"m", "meter", "meters", "metre", "metres"}:
        return as_float_scalar(value)
    return mm_or_m_to_m(value)
