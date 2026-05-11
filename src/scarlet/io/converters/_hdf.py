from __future__ import annotations

from typing import Optional

import h5py
import numpy as np


def as_str(x) -> str:
    """Normalize HDF5 scalar-like values to a Python string."""
    if isinstance(x, (bytes, bytearray)):
        return x.decode(errors="replace")
    if isinstance(x, np.ndarray) and x.size == 1:
        return as_str(x.reshape(()).item())
    return str(x)


def as_float_scalar(x) -> float:
    """Return a scalar float from Python or NumPy scalar-like values."""
    if x is None:
        return float("nan")
    if isinstance(x, np.ndarray) and x.size == 1:
        return float(x.reshape(()))
    return float(x)


def safe_get(fin: h5py.File, path: str):
    """Read a dataset value if the path exists, otherwise return None."""
    return fin[path][()] if path in fin else None


def safe_get_dataset(fin: h5py.File, path: str) -> Optional[h5py.Dataset]:
    """Return the dataset object at a path when it exists and is a dataset."""
    if path not in fin:
        return None
    obj = fin[path]
    return obj if isinstance(obj, h5py.Dataset) else None


def ensure_group(g: h5py.Group, name: str, nx_class: Optional[str] = None) -> h5py.Group:
    """Create a child group and optionally assign its NX_class attribute."""
    gg = g.create_group(name)
    if nx_class:
        gg.attrs["NX_class"] = np.bytes_(nx_class)
    return gg


def write_dataset(
    g: h5py.Group,
    name: str,
    data,
    *,
    as_string: bool = False,
    units: Optional[str] = None,
) -> h5py.Dataset:
    """Create a dataset and optionally encode it as a string and/or add units."""
    if as_string:
        data = np.bytes_(str(data))
    ds = g.create_dataset(name, data=data)
    if units is not None:
        ds.attrs["units"] = np.bytes_(units)
    return ds


def pick_entry(fin: h5py.File, preferred: Optional[str] = None) -> str:
    """Pick the most appropriate NXentry path for a raw NeXus input file."""
    if preferred and preferred in fin:
        return preferred
    for cand in ("/raw_data", "/entry0", "/entry", "/entry1"):
        if cand in fin:
            return cand
    for k in fin.keys():
        p = f"/{k}"
        if isinstance(fin[p], h5py.Group):
            nx = fin[p].attrs.get("NX_class", None)
            if nx is not None and as_str(nx) == "NXentry":
                return p
    raise ValueError("No NXentry found in file.")
