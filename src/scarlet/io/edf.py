from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


_EDF_DTYPES = {
    "SignedByte": np.dtype("i1"),
    "UnsignedByte": np.dtype("u1"),
    "SignedShort": np.dtype("i2"),
    "UnsignedShort": np.dtype("u2"),
    "SignedInteger": np.dtype("i4"),
    "UnsignedInteger": np.dtype("u4"),
    "SignedLong": np.dtype("i8"),
    "UnsignedLong": np.dtype("u8"),
    "FloatValue": np.dtype("f4"),
    "Float": np.dtype("f4"),
    "DoubleValue": np.dtype("f8"),
    "Double": np.dtype("f8"),
}


def _parse_edf_header(header_text: str) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in header_text.splitlines():
        line = line.strip()
        if not line or line in {"{", "}"}:
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        header[key.strip()] = value.strip().rstrip(";").strip()
    return header


def read_edf_image(path: Union[str, Path]) -> np.ndarray:
    """
    Read a simple EDF image into a NumPy array.

    This reader intentionally supports the subset of EDF used by SCARLET mask
    files and does not depend on third-party EDF packages.
    """
    path = Path(path)
    raw = path.read_bytes()
    end = raw.find(b"}")
    if end < 0:
        raise ValueError(f"Invalid EDF header in {path}: missing closing brace")

    preliminary_header = raw[: end + 1].decode("ascii", errors="replace")
    header = _parse_edf_header(preliminary_header)
    header_size = int(header.get("EDF_HeaderSize", end + 1))
    if header_size <= 0:
        raise ValueError(f"Invalid EDF_HeaderSize in {path}: {header_size}")

    header_text = raw[:header_size].decode("ascii", errors="replace")
    header = _parse_edf_header(header_text)

    dim_1 = int(header["Dim_1"])
    dim_2 = int(header["Dim_2"])
    data_type = header["DataType"]
    if data_type not in _EDF_DTYPES:
        raise ValueError(f"Unsupported EDF DataType in {path}: {data_type}")

    dtype = _EDF_DTYPES[data_type]
    byte_order = header.get("ByteOrder", "")
    if dtype.itemsize > 1:
        if byte_order == "LowByteFirst":
            dtype = dtype.newbyteorder("<")
        elif byte_order == "HighByteFirst":
            dtype = dtype.newbyteorder(">")

    binary_size = int(header.get("EDF_BinarySize", dim_1 * dim_2 * dtype.itemsize))
    payload = raw[header_size : header_size + binary_size]
    expected_size = dim_1 * dim_2 * dtype.itemsize
    if len(payload) < expected_size:
        raise ValueError(
            f"EDF payload too short in {path}: expected at least {expected_size} bytes, got {len(payload)}"
        )

    return np.frombuffer(payload[:expected_size], dtype=dtype).reshape(dim_2, dim_1)


def read_edf_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Read an EDF mask and normalize it to uint8 with 1=masked, 0=valid.
    """
    return (read_edf_image(path) != 0).astype(np.uint8)


__all__ = ["read_edf_image", "read_edf_mask"]
