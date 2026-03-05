from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal


ApparatusName = Literal["sam", "sansllb"]


def _normalize_apparatus_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "")


def list_apparatus() -> list[str]:
    return ["sam", "sansllb"]


def get_converter(apparatus: str) -> Callable[..., Any]:
    """
    Return a converter function by apparatus name.

    Known apparatus names:
    - "sam"
    - "sansllb" (aliases: "sans-llb", "sans_llb")
    """
    key = _normalize_apparatus_name(apparatus)
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


__all__ = ["ApparatusName", "convert_to_scarlet_nxsas_raw", "get_converter", "list_apparatus"]

