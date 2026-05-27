from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Optional, Tuple, Union, List, Literal, Mapping

import h5py
import numpy as np
import math

from scarlet.io.edf import read_edf_mask

ApertureType = Literal["slit", "pinhole"]


@dataclass(frozen=True)
class Aperture:
    type: ApertureType
    x_gap: Optional[float] = None  # m (slit)
    y_gap: Optional[float] = None  # m (slit)
    diameter: Optional[float] = None  # m (pinhole)

    @staticmethod
    def from_group(g: h5py.Group, *, issues: List[str], path: str) -> "Aperture":
        nx = g.attrs.get("NX_class", None)
        nx = nx.decode() if isinstance(nx, (bytes, bytearray)) else str(nx) if nx is not None else ""

        def get_float(name: str) -> Optional[float]:
            if name not in g:
                return None
            try:
                return float(g[name][()])
            except Exception:
                issues.append(f"{path}/{name}: cannot read as float")
                return None

        if nx == "NXslit":
            x = get_float("x_gap")
            y = get_float("y_gap")
            if x is None or y is None:
                issues.append(f"{path}: NXslit requires x_gap and y_gap")
            return Aperture(type="slit", x_gap=x, y_gap=y)

        if nx == "NXpinhole":
            d = get_float("diameter")
            if d is None:
                issues.append(f"{path}: NXpinhole requires diameter")
            return Aperture(type="pinhole", diameter=d)

        # Accept NXaperture as a fallback (map circular->pinhole, rectangular->slit)
        if nx == "NXaperture":
            shape = g["shape"][()].decode() if "shape" in g and isinstance(g["shape"][()], (bytes, bytearray)) else None
            d = get_float("diameter")
            x = get_float("x_gap")
            y = get_float("y_gap")
            if d is not None:
                return Aperture(type="pinhole", diameter=d)
            if x is not None and y is not None:
                return Aperture(type="slit", x_gap=x, y_gap=y)
            issues.append(f"{path}: NXaperture could not be mapped (need diameter or x_gap/y_gap)")
            return Aperture(type="pinhole")  # placeholder

        issues.append(f"{path}: unsupported NX_class={nx!r} (expected NXslit or NXpinhole)")
        return Aperture(type="pinhole")  # placeholder


@dataclass(frozen=True)
class Collimation:
    aperture1: Aperture
    aperture2: Aperture
    collimation_distance: float  # m, distance between apertures
    last_aperture_to_sample_distance: float  # m, distance from aperture2 to sample


@dataclass(frozen=True)
class Configuration:
    wavelength: float  # Å
    sample_detector_distance: Union[float, List[float]]  # m
    collimation: Optional[Collimation] = None
    config_id: Optional[str] = None
    notes: Optional[str] = None


def _write_dataset(parent: h5py.Group, name: str, value) -> None:
    if isinstance(value, str):
        parent.create_dataset(name, data=np.bytes_(value))
    else:
        parent.create_dataset(name, data=value)


def _replace_dataset(parent: h5py.Group, name: str, value) -> None:
    if name in parent:
        del parent[name]
    _write_dataset(parent, name, value)


def _require_number(name: str, value: Optional[float]) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    out = float(value)
    if math.isnan(out):
        raise ValueError(f"{name} must not be NaN")
    return out


def _scalar_distance(value: Union[float, List[float]], *, output_kind: str) -> float:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"sample_detector_distance must be a scalar for {output_kind} output")
        value = value[0]
    return _require_number("sample_detector_distance", float(value))


def _write_aperture(parent: h5py.Group, name: str, aperture: "Aperture") -> None:
    ap = parent.create_group(name)
    if aperture.type == "slit":
        ap.attrs["NX_class"] = np.bytes_("NXslit")
        _write_dataset(ap, "x_gap", _require_number(f"{name}.x_gap", aperture.x_gap))
        _write_dataset(ap, "y_gap", _require_number(f"{name}.y_gap", aperture.y_gap))
        return
    if aperture.type == "pinhole":
        ap.attrs["NX_class"] = np.bytes_("NXpinhole")
        _write_dataset(ap, "diameter", _require_number(f"{name}.diameter", aperture.diameter))
        return
    raise ValueError(f"Unsupported aperture type: {aperture.type!r}")


def _select_entry(source: h5py.File) -> h5py.Group:
    for path in ("/raw_data", "/entry", "/entry0", "/entry1"):
        if path in source and isinstance(source[path], h5py.Group):
            return source[path]
    raise ValueError("Reference file must contain /raw_data, /entry, /entry0, or /entry1")


def _copy_reference(parent: h5py.Group, name: str, source_path: Union[str, Path]) -> Path:
    def _rewrite_internal_soft_links(
        copied_group: h5py.Group,
        *,
        copied_entry_path: str,
        source_entry_path: str,
    ) -> None:
        for key in list(copied_group.keys()):
            child_path = f"{copied_group.name}/{key}"
            link = copied_group.file.get(child_path, getlink=True)
            if isinstance(link, h5py.SoftLink) and isinstance(link.path, str):
                if link.path == source_entry_path or link.path.startswith(f"{source_entry_path}/"):
                    suffix = link.path[len(source_entry_path) :]
                    del copied_group[key]
                    copied_group[key] = h5py.SoftLink(f"{copied_entry_path}{suffix}")
                    continue

            child = copied_group[key]
            if isinstance(child, h5py.Group):
                _rewrite_internal_soft_links(
                    child,
                    copied_entry_path=copied_entry_path,
                    source_entry_path=source_entry_path,
                )

    source_path = Path(source_path).resolve()
    ref_group = parent.create_group(name)
    with h5py.File(source_path, "r") as source_file:
        entry_group = _select_entry(source_file)
        source_file.copy(entry_group, ref_group, name="entry")
        _rewrite_internal_soft_links(
            ref_group["entry"],
            copied_entry_path=ref_group["entry"].name,
            source_entry_path=entry_group.name,
        )
    return source_path


def _file_created_utc(path: Path) -> str:
    stat = path.stat()
    timestamp = getattr(stat, "st_birthtime", stat.st_ctime)
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def _write_configuration_group(entry: h5py.Group, configuration: "Configuration", *, output_kind: str) -> None:
    if configuration.config_id is None:
        raise ValueError(f"configuration.config_id is required for {output_kind} output")
    if configuration.collimation is None:
        raise ValueError(f"configuration.collimation is required for {output_kind} output")

    _write_dataset(entry, "config_id", configuration.config_id)

    cfg = entry.create_group("configuration")
    cfg.attrs["NX_class"] = np.bytes_("NXcollection")
    _write_dataset(cfg, "wavelength", _require_number("wavelength", configuration.wavelength))
    _write_dataset(
        cfg,
        "sample_detector_distance",
        _scalar_distance(configuration.sample_detector_distance, output_kind=output_kind),
    )
    if configuration.notes is not None:
        _write_dataset(cfg, "notes", configuration.notes)

    col = cfg.create_group("collimation")
    col.attrs["NX_class"] = np.bytes_("NXcollection")
    _write_dataset(
        col,
        "collimation_distance",
        _require_number("collimation_distance", configuration.collimation.collimation_distance),
    )
    _write_dataset(
        col,
        "last_aperture_to_sample_distance",
        _require_number(
            "last_aperture_to_sample_distance",
            configuration.collimation.last_aperture_to_sample_distance,
        ),
    )
    _write_aperture(col, "aperture1", configuration.collimation.aperture1)
    _write_aperture(col, "aperture2", configuration.collimation.aperture2)


def _write_mask_group(
    entry: h5py.Group,
    *,
    masks: Optional[Mapping[int, np.ndarray]] = None,
    beamstop_masks: Optional[Mapping[int, np.ndarray]] = None,
) -> None:
    combined_masks: dict[int, np.ndarray] = {}
    for source in (masks or {}, beamstop_masks or {}):
        for detector, mask in source.items():
            detector_index = int(detector)
            array = np.asarray(mask)
            if detector_index in combined_masks:
                if combined_masks[detector_index].shape != array.shape:
                    raise ValueError(
                        f"Mask shape mismatch for detector{detector_index}: "
                        f"{combined_masks[detector_index].shape} vs {array.shape}"
                    )
                combined_masks[detector_index] = np.maximum(combined_masks[detector_index], array)
            else:
                combined_masks[detector_index] = array
    if not combined_masks:
        return

    mask_group = entry.create_group("mask")
    mask_group.attrs["NX_class"] = np.bytes_("NXcollection")
    for detector, mask in sorted(combined_masks.items()):
        _write_dataset(mask_group, f"mask_detector{detector}", mask)


def _read_beam_centers_from_file(path: Path) -> dict[int, tuple[float, float]]:
    beam_centers: dict[int, tuple[float, float]] = {}
    with h5py.File(path, "r") as f:
        entry_path = _entry_path_from_file(f, path)
        instrument_path = f"{entry_path}/instrument"
        if instrument_path not in f:
            return beam_centers

        instrument = f[instrument_path]
        for name in instrument.keys():
            match = re.fullmatch(r"detector(\d+)", name)
            if match is None:
                continue
            detector_path = f"{instrument_path}/{name}"
            beam_center_x_path = f"{detector_path}/beam_center_x"
            beam_center_y_path = f"{detector_path}/beam_center_y"
            if beam_center_x_path not in f or beam_center_y_path not in f:
                continue
            try:
                beam_centers[int(match.group(1))] = (
                    float(f[beam_center_x_path][()]),
                    float(f[beam_center_y_path][()]),
                )
            except Exception:
                continue
    return beam_centers


def _write_beam_center_group(
    entry: h5py.Group,
    *,
    beam_centers: Optional[Mapping[int, tuple[float, float]]] = None,
) -> None:
    if not beam_centers:
        return

    beam_center_group = entry.create_group("beam_center")
    beam_center_group.attrs["NX_class"] = np.bytes_("NXcollection")
    for detector_index, (x, y) in sorted(beam_centers.items()):
        detector_group = beam_center_group.create_group(f"detector{int(detector_index)}")
        detector_group.attrs["NX_class"] = np.bytes_("NXcollection")
        _write_dataset(detector_group, "beam_center_x", float(x))
        _write_dataset(detector_group, "beam_center_y", float(y))


def _write_transmission_roi_group(
    entry: h5py.Group,
    *,
    transmission_roi_detector: Union[int, str],
    transmission_roi: tuple[int, int, int, int],
    transmission_roi_method: Optional[str] = None,
    transmission_roi_notes: Optional[str] = None,
) -> None:
    x0, x1, y0, y1 = transmission_roi
    transmission_roi_group = entry.create_group("transmission_roi")
    transmission_roi_group.attrs["NX_class"] = np.bytes_("NXcollection")
    _write_dataset(transmission_roi_group, "detector", transmission_roi_detector)
    _write_dataset(transmission_roi_group, "roi_type", "rectangle")
    _write_dataset(transmission_roi_group, "x0", int(x0))
    _write_dataset(transmission_roi_group, "x1", int(x1))
    _write_dataset(transmission_roi_group, "y0", int(y0))
    _write_dataset(transmission_roi_group, "y1", int(y1))
    if transmission_roi_method is not None:
        _write_dataset(transmission_roi_group, "method", transmission_roi_method)
    if transmission_roi_notes is not None:
        _write_dataset(transmission_roi_group, "notes", transmission_roi_notes)


def _write_transmission_setup_group(entry: h5py.Group, *, attenuation_factor: Optional[float]) -> None:
    if attenuation_factor is None:
        return
    transmission_setup = entry.create_group("transmission_setup")
    transmission_setup.attrs["NX_class"] = np.bytes_("NXcollection")
    attenuator = transmission_setup.create_group("attenuator")
    attenuator.attrs["NX_class"] = np.bytes_("NXattenuator")
    _write_dataset(
        attenuator,
        "attenuation_factor",
        _require_number("attenuation_factor", attenuation_factor),
    )


def _write_meta_group(
    entry: h5py.Group,
    *,
    source_paths: Mapping[str, Path],
    created_utc: str,
    mask_convention: str,
    scarlet_version: Optional[str] = None,
) -> None:
    meta = entry.create_group("meta")
    meta.attrs["NX_class"] = np.bytes_("NXcollection")
    _write_dataset(meta, "created_utc", created_utc)
    _write_dataset(meta, "mask_convention", mask_convention)
    for name, source_path in source_paths.items():
        _write_dataset(meta, f"{name}_source_file", str(source_path))
    if scarlet_version is not None:
        _write_dataset(meta, "scarlet_version", scarlet_version)


def _reference_detector_shape(entry: h5py.Group, detector_index: int) -> Optional[tuple[int, ...]]:
    references = entry.get("references")
    if not isinstance(references, h5py.Group):
        return None

    dataset_name = f"instrument/detector{int(detector_index)}/data"
    for reference in references.values():
        if not isinstance(reference, h5py.Group):
            continue
        copied_entry = reference.get("entry")
        if not isinstance(copied_entry, h5py.Group):
            continue
        dataset = copied_entry.get(dataset_name)
        if isinstance(dataset, h5py.Dataset):
            return tuple(int(v) for v in dataset.shape)
    return None


def _normalize_mask_array(mask: Union[np.ndarray, str, Path], *, label: str) -> np.ndarray:
    if isinstance(mask, (str, Path)):
        path = Path(mask)
        if path.suffix.lower() != ".edf":
            raise ValueError(f"{label} file must be an EDF mask, got {path}")
        array = read_edf_mask(path)
    else:
        array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array, got shape {array.shape}")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{label} must contain only 0/1 values")
    return array.astype(np.uint8, copy=False)


from .reference import (
    insert_beam_centers_in_refs_file,
    insert_masks_in_refs_file,
    write_refs_norm_file,
    write_refs_sub_file,
)


def configuration_from_nexus(
    file_path: Union[str, Path],
    *,
    entry_path: str = "/raw_data",
    detector_index: int = 0,
) -> Tuple[Configuration, List[str]]:
    """
    Build a Configuration object from a NeXus/HDF5 file.

    Priority:
      1) If <entry_path>/configuration exists, parse it (SCARLET_refs_sub style).
      2) Else try to infer from <entry_path>/instrument (SCARLET NXsas_raw style).

    Returns:
      (Configuration, issues)
    """
    file_path = Path(file_path)
    issues: List[str] = []

    def _as_str(x) -> Optional[str]:
        if x is None:
            return None
        if isinstance(x, (bytes, bytearray)):
            return x.decode(errors="replace")
        return str(x)

    def _read_scalar(f: h5py.File, p: str) -> Optional[float]:
        if p not in f:
            return None
        try:
            return float(f[p][()])
        except Exception:
            issues.append(f"{p}: cannot read as float")
            return None

    def _read_text(f: h5py.File, p: str) -> Optional[str]:
        if p not in f:
            return None
        try:
            return _as_str(f[p][()])
        except Exception:
            issues.append(f"{p}: cannot read as string")
            return None

    with h5py.File(file_path, "r") as f:
        if entry_path not in f:
            # common fallback for older files using /entry, /entry0 or /entry1
            for cand in ("/entry", "/entry0", "/entry1"):
                if cand in f:
                    entry_path = cand
                    break
            else:
                raise ValueError(f"No entry group found at {entry_path}")

        # ---------------------
        # Case 1: explicit /configuration
        # ---------------------
        cfg_path = f"{entry_path}/configuration"
        if cfg_path in f and isinstance(f[cfg_path], h5py.Group):
            wl = _read_scalar(f, f"{cfg_path}/wavelength")
            dsd = _read_scalar(f, f"{cfg_path}/sample_detector_distance")
            notes = _read_text(f, f"{cfg_path}/notes")
            config_id = _read_text(f, f"{entry_path}/config_id")

            if wl is None:
                issues.append(f"{cfg_path}/wavelength missing")
                wl = float("nan")
            if dsd is None:
                issues.append(f"{cfg_path}/sample_detector_distance missing")
                dsd = float("nan")

            # Collimation block
            col_path = f"{cfg_path}/collimation"
            collimation_obj: Optional[Collimation] = None
            if col_path in f and isinstance(f[col_path], h5py.Group):
                cd = _read_scalar(f, f"{col_path}/collimation_distance")
                lad = _read_scalar(f, f"{col_path}/last_aperture_to_sample_distance")

                ap1_path = f"{col_path}/aperture1"
                ap2_path = f"{col_path}/aperture2"

                if ap1_path in f and ap2_path in f and cd is not None and lad is not None:
                    ap1 = Aperture.from_group(f[ap1_path], issues=issues, path=ap1_path)
                    ap2 = Aperture.from_group(f[ap2_path], issues=issues, path=ap2_path)
                    collimation_obj = Collimation(
                        aperture1=ap1,
                        aperture2=ap2,
                        collimation_distance=float(cd),
                        last_aperture_to_sample_distance=float(lad),
                    )
                else:
                    issues.append(f"{col_path}: incomplete collimation (need distances + aperture1 + aperture2)")

            return Configuration(
                wavelength=float(wl),
                sample_detector_distance=float(dsd),
                collimation=collimation_obj,
                config_id=config_id,
                notes=notes,
            ), issues

        # ---------------------
        # Case 2: infer from instrument
        # ---------------------
        inst = f"{entry_path}/instrument"

        # wavelength: prefer monochromator/wavelength
        wl = _read_scalar(f, f"{inst}/monochromator/wavelength")
        if wl is None:
            wl = _read_scalar(f, f"{inst}/source/incident_wavelength")
        if wl is None:
            issues.append("Could not infer wavelength (monochromator/wavelength or source/incident_wavelength missing)")
            wl = float("nan")

        # sample-detector distance: prefer detector{idx}/transformations/translation[2], else detector/distance
        det = f"{inst}/detector{detector_index}"
        dsd = None
        tr_path = f"{det}/transformations/translation"
        if tr_path in f:
            try:
                t = np.array(f[tr_path][()], dtype=float).reshape(-1)
                if t.size >= 3:
                    dsd = float(t[2])
            except Exception:
                issues.append(f"{tr_path}: cannot read translation vector")
        if dsd is None:
            dsd = _read_scalar(f, f"{det}/distance")
        if dsd is None:
            issues.append(f"Could not infer sample_detector_distance from {det}")
            dsd = float("nan")

        # Attempt to infer collimation from /instrument/collimation elements:
        collimation_obj: Optional[Collimation] = None
        col = f"{inst}/collimation"
        if col in f and isinstance(f[col], h5py.Group) and f"{col}/elements" in f:
            elems = f[f"{col}/elements"]
            # collect aperture-like elements with a z position
            candidates = []
            for name, g in elems.items():
                if not isinstance(g, h5py.Group):
                    continue
                nx = g.attrs.get("NX_class", None)
                nx = nx.decode() if isinstance(nx, (bytes, bytearray)) else str(nx) if nx is not None else ""
                if nx not in ("NXslit", "NXpinhole", "NXaperture"):
                    continue
                z = None
                tp = f"{col}/elements/{name}/transformations/translation"
                if tp in f:
                    try:
                        t = np.array(f[tp][()], dtype=float).reshape(-1)
                        if t.size >= 3:
                            z = float(t[2])
                    except Exception:
                        pass
                if z is not None:
                    candidates.append((z, name, g))

            # take two closest to sample on upstream side (z < 0), i.e., largest z values below 0
            upstream = [(z, name, g) for (z, name, g) in candidates if z < 0]
            upstream.sort(key=lambda x: x[0])  # increasing z
            if len(upstream) >= 2:
                z1, n1, g1 = upstream[-2]
                z2, n2, g2 = upstream[-1]  # closest to sample
                cd = float(z2 - z1)
                lad = float(-z2)  # sample at z=0
                ap1 = Aperture.from_group(g1, issues=issues, path=f"{col}/elements/{n1}")
                ap2 = Aperture.from_group(g2, issues=issues, path=f"{col}/elements/{n2}")
                collimation_obj = Collimation(
                    aperture1=ap1, aperture2=ap2,
                    collimation_distance=cd,
                    last_aperture_to_sample_distance=lad
                )
            else:
                issues.append("Could not infer 2 upstream apertures from instrument/collimation (need z positions)")

        return Configuration(
            wavelength=float(wl),
            sample_detector_distance=float(dsd),
            collimation=collimation_obj,
            config_id=None,
            notes=None,
        ), issues
    

@dataclass(frozen=True)
class ConfigTolerance:
    """Tolerances for comparing configurations."""
    distance_m: float = 0.01     # 1 cm
    wavelength_a: float = 0.1    # 0.1 Å
    aperture_m: float = 0.001    # 1 mm (slit gaps or pinhole diameter)


def _is_nan(x: float) -> bool:
    try:
        return math.isnan(x)
    except Exception:
        return False


def _close(a: Optional[float], b: Optional[float], tol: float) -> Tuple[bool, Optional[float]]:
    """Return (is_close, abs_diff). None means 'missing'."""
    if a is None or b is None:
        return False, None
    aa = float(a)
    bb = float(b)
    if _is_nan(aa) or _is_nan(bb):
        return False, None
    return (abs(aa - bb) <= tol), abs(aa - bb)


def _fmt_diff(name: str, a: Optional[float], b: Optional[float], diff: Optional[float], tol: float, unit: str) -> str:
    """Format a difference message for a parameter."""
    if a is None or b is None or diff is None:
        return f"{name}: missing value(s) (a={a}, b={b})"
    return f"{name}: a={a:.6g}{unit}, b={b:.6g}{unit}, |Δ|={diff:.3g}{unit} > tol={tol:.3g}{unit}"


def compare_configurations(
    a: "Configuration",
    b: "Configuration",
    *,
    tol: ConfigTolerance = ConfigTolerance(),
    require_collimation: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Compare two Configuration objects with tolerances.

    Parameters
    ----------
    tol:
      distance_m: tolerance for distances in meters (default 1 cm)
      wavelength_a: tolerance for wavelength in Å (default 0.1 Å)
      aperture_m: tolerance for slit gaps or pinhole diameter in meters (default 1 mm)
    require_collimation:
      If True, collimation must be present in both configs and match within tolerances.
      If False, collimation is compared only if present in both (otherwise ignored).

    Returns
    -------
    same, diffs
      same: True if configs match within tolerances
      diffs: list of human-readable differences
    """
    diffs: List[str] = []

    ok, d = _close(a.wavelength, b.wavelength, tol.wavelength_a)
    if not ok:
        diffs.append(_fmt_diff("wavelength", a.wavelength, b.wavelength, d, tol.wavelength_a, "Å"))

    ok, d = _close(a.sample_detector_distance, b.sample_detector_distance, tol.distance_m)
    if not ok:
        diffs.append(_fmt_diff("sample_detector_distance", a.sample_detector_distance, b.sample_detector_distance, d, tol.distance_m, "m"))

    # Collimation
    if a.collimation is None or b.collimation is None:
        if require_collimation:
            diffs.append(f"collimation: missing in {'a' if a.collimation is None else ''}{' and ' if a.collimation is None and b.collimation is None else ''}{'b' if b.collimation is None else ''}")
        # else: ignore
        return (len(diffs) == 0), diffs

    # distances
    ok, d = _close(a.collimation.collimation_distance, b.collimation.collimation_distance, tol.distance_m)
    if not ok:
        diffs.append(_fmt_diff("collimation_distance", a.collimation.collimation_distance, b.collimation.collimation_distance, d, tol.distance_m, "m"))

    ok, d = _close(a.collimation.last_aperture_to_sample_distance, b.collimation.last_aperture_to_sample_distance, tol.distance_m)
    if not ok:
        diffs.append(_fmt_diff("last_aperture_to_sample_distance", a.collimation.last_aperture_to_sample_distance, b.collimation.last_aperture_to_sample_distance, d, tol.distance_m, "m"))

    # apertures
    def cmp_aperture(label: str, ap_a: "Aperture", ap_b: "Aperture") -> None:
        if ap_a.type != ap_b.type:
            diffs.append(f"{label}: type mismatch (a={ap_a.type}, b={ap_b.type})")
            return

        if ap_a.type == "slit":
            ok, d = _close(ap_a.x_gap, ap_b.x_gap, tol.aperture_m)
            if not ok:
                diffs.append(_fmt_diff(f"{label}.x_gap", ap_a.x_gap, ap_b.x_gap, d, tol.aperture_m, "m"))
            ok, d = _close(ap_a.y_gap, ap_b.y_gap, tol.aperture_m)
            if not ok:
                diffs.append(_fmt_diff(f"{label}.y_gap", ap_a.y_gap, ap_b.y_gap, d, tol.aperture_m, "m"))

        elif ap_a.type == "pinhole":
            ok, d = _close(ap_a.diameter, ap_b.diameter, tol.aperture_m)
            if not ok:
                diffs.append(_fmt_diff(f"{label}.diameter", ap_a.diameter, ap_b.diameter, d, tol.aperture_m, "m"))

    cmp_aperture("aperture1", a.collimation.aperture1, b.collimation.aperture2 if False else b.collimation.aperture1)
    cmp_aperture("aperture2", a.collimation.aperture2, b.collimation.aperture2)

    return (len(diffs) == 0), diffs

def _entry_path_from_file(f: h5py.File, path: Path) -> str:
    entry_path = "/raw_data"
    if entry_path in f:
        return entry_path
    for candidate in ("/entry", "/entry0", "/entry1"):
        if candidate in f:
            return candidate
    raise ValueError(f"No entry group found in {path}")


def _count_time_from_file(path: Path) -> float:
    with h5py.File(path, "r") as f:
        entry_path = _entry_path_from_file(f, path)

        for dataset_path in (
            f"{entry_path}/control/count_time",
            f"{entry_path}/instrument/monitor0/count_time",
            f"{entry_path}/instrument/monitor1/count_time",
            f"{entry_path}/instrument/monitor2/count_time",
        ):
            if dataset_path not in f:
                continue
            try:
                return float(f[dataset_path][()])
            except Exception:
                continue
    return float("-inf")


def _transmission_roi_from_file(
    path: Path,
    *,
    transmission_roi_detector: int,
    transmission_roi_half_size: int,
) -> tuple[int, int, int, int]:
    with h5py.File(path, "r") as f:
        entry_path = _entry_path_from_file(f, path)
        dataset_path = f"{entry_path}/instrument/detector{transmission_roi_detector}/data"
        if dataset_path not in f:
            raise ValueError(f"Missing detector dataset for ROI estimation: {dataset_path}")
        data = np.asarray(f[dataset_path][()], dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Detector dataset must be 2D for ROI estimation: {path}")
    ny, nx = data.shape
    peak_y, peak_x = np.unravel_index(int(np.nanargmax(data)), data.shape)
    peak_value = float(data[peak_y, peak_x])
    background = float(np.nanmedian(data))

    if not np.isfinite(peak_value):
        peak_value = 0.0
    if not np.isfinite(background):
        background = 0.0

    threshold = background + 0.1 * max(0.0, peak_value - background)
    bright_mask = np.isfinite(data) & (data >= threshold)

    if bright_mask[peak_y, peak_x]:
        component_mask = np.zeros_like(bright_mask, dtype=bool)
        stack = [(int(peak_y), int(peak_x))]
        component_mask[peak_y, peak_x] = True
        while stack:
            y, x = stack.pop()
            for yy in range(max(0, y - 1), min(ny, y + 2)):
                for xx in range(max(0, x - 1), min(nx, x + 2)):
                    if component_mask[yy, xx] or not bright_mask[yy, xx]:
                        continue
                    component_mask[yy, xx] = True
                    stack.append((yy, xx))
        ys, xs = np.nonzero(component_mask)
    else:
        ys = np.array([peak_y])
        xs = np.array([peak_x])

    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    pad_x = max(int(transmission_roi_half_size), int(math.ceil(0.05 * width)))
    pad_y = max(int(transmission_roi_half_size), int(math.ceil(0.05 * height)))
    return (
        max(0, x0 - pad_x),
        min(nx - 1, x1 + pad_x),
        max(0, y0 - pad_y),
        min(ny - 1, y1 + pad_y),
    )
