from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal


ApparatusName = Literal["d11", "sam", "sansllb"]


def _normalize_apparatus_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "")


def list_apparatus() -> list[str]:
    return ["d11", "sam", "sansllb"]


def get_converter(apparatus: str) -> Callable[..., Any]:
    """
    Return a converter function by apparatus name.

    Known apparatus names:
    - "d11"
    - "sam"
    - "sansllb" (aliases: "sans-llb", "sans_llb")
    """
    key = _normalize_apparatus_name(apparatus)
    if key == "d11":
        from .d11 import convert_d11_to_scarlet_nxsas_raw

        return convert_d11_to_scarlet_nxsas_raw
    if key == "sam":
        from .sam import convert_sam_to_scarlet_nxsas_raw

        return convert_sam_to_scarlet_nxsas_raw
    if key == "sansllb":
        from .sansllb import convert_sansllb_to_scarlet_nxsas_raw

        return convert_sansllb_to_scarlet_nxsas_raw

    known = ", ".join(list_apparatus())
    raise ValueError(f"Unknown apparatus {apparatus!r}. Known: {known}")


def convert_to_scarlet_nxsas_raw(
    apparatus: str,
    input_path: str | Path,
    output_path: str | Path,
    **kwargs: Any,
):
    """
    Dispatch conversion to a SCARLET NXsas_raw file based on `apparatus`.

    Parameters
    ----------
    apparatus:
        Apparatus name (e.g. "sam", "sansllb").
    input_path, output_path:
        Input and output file paths.
    kwargs:
        Passed through to the underlying converter.
    """
    conv = get_converter(apparatus)
    return conv(input_path, output_path, **kwargs)


def _nx_class_text(value: Any) -> str:
    """Return one HDF5 NX_class attribute as text without importing converter internals."""
    import numpy as np

    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(()).item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="replace")
    return str(value)


def _find_nxattenuator(handle: Any) -> Any | None:
    """Find the first NXattenuator group in a raw or converted NeXus file."""
    import h5py

    preferred_paths = (
        "/raw_data/instrument/attenuator",
        "/entry/instrument/attenuator",
        "/entry0/instrument/attenuator",
        "/entry0/SANS-LLB/attenuator",
    )
    for path in preferred_paths:
        if path in handle and isinstance(handle[path], h5py.Group):
            return handle[path]

    found: list[Any] = []

    def visit(_name: str, obj: Any) -> None:
        if found or not isinstance(obj, h5py.Group):
            return
        if _nx_class_text(obj.attrs.get("NX_class", "")) == "NXattenuator":
            found.append(obj)

    handle.visititems(visit)
    return found[0] if found else None


def _scalar_text(value: Any) -> str:
    import numpy as np

    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(()).item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="replace").strip()
    return str(value).strip()


def _sansllb_attenuator_signature(group: Any) -> tuple[str | None, float | None]:
    """Return the normalized selection and transmission represented by an NXattenuator."""
    import numpy as np

    selection = _scalar_text(group["selection"][()]) if "selection" in group else None
    candidates = (
        ("attenuator_transmission", False),
        ("attenuation_transmission", False),
        ("transmission", False),
        ("attenuation_factor", True),
    )
    for name, invert in candidates:
        if name not in group:
            continue
        try:
            raw_value = np.asarray(group[name][()])
            if raw_value.size != 1:
                return selection, None
            value = float(raw_value.reshape(()))
        except (TypeError, ValueError):
            return selection, None
        if not np.isfinite(value) or value <= 0.0:
            return selection, None
        transmission = 1.0 / value if invert else value
        if transmission > 1.0:
            return selection, None
        return selection, transmission
    return selection, None


def is_conversion_cache_compatible(
    apparatus: str,
    input_path: str | Path,
    output_path: str | Path,
) -> bool:
    """Check metadata that must survive conversion before reusing a cached file."""
    output_path = Path(output_path)
    if not output_path.is_file():
        return False
    if _normalize_apparatus_name(apparatus) != "sansllb":
        return True

    import h5py
    import numpy as np

    try:
        with h5py.File(input_path, "r") as source, h5py.File(output_path, "r") as converted:
            source_attenuator = _find_nxattenuator(source)
            if source_attenuator is None:
                return True

            expected_selection, expected_transmission = _sansllb_attenuator_signature(source_attenuator)
            if expected_selection is None:
                return True

            converted_attenuator = _find_nxattenuator(converted)
            if converted_attenuator is None:
                return False
            actual_selection, actual_transmission = _sansllb_attenuator_signature(converted_attenuator)
            if actual_selection != expected_selection:
                return False
            if expected_transmission is None:
                return True
            if actual_transmission is None:
                return False
            return bool(np.isclose(actual_transmission, expected_transmission, rtol=1e-12, atol=0.0))
    except OSError:
        return False


__all__ = [
    "ApparatusName",
    "convert_to_scarlet_nxsas_raw",
    "get_converter",
    "is_conversion_cache_compatible",
    "list_apparatus",
]
