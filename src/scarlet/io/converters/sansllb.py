from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from ._deadtime import correct_detector_data_for_deadtime
from ._report import ConvertReport
from ._hdf import (
    as_float_scalar as _as_float_scalar,
    as_str as _as_str,
    ensure_group as _ensure_group,
    pick_entry as _pick_entry,
    safe_get as _safe_get,
    safe_get_dataset as _safe_get_dataset,
    write_dataset as _write_dataset,
)
from ._units import MM_TO_M, length_dataset_to_m as _length_dataset_to_m, mm_or_m_to_m as _mm_or_m_to_m


NM_TO_ANGSTROM = 10.0
_VALID_MONITOR_MODES = {"monitor", "timer"}


def _sansllb_detector_view_name(detector_name: str) -> Optional[str]:
    """Map raw SANS-LLB detector groups to their NXdata view holding posx/posy axes."""
    explicit = {
        "left_detector": "left_data",
        "bottom_detector": "bottom_data",
    }
    return explicit.get(detector_name)


def _beam_center_axis_from_positions(
    data_group: h5py.Group,
    *,
    axis: str,
) -> Optional[float]:
    """Infer the beam-center pixel coordinate from the raw pos{axis} axis where pos=0 at the direct beam."""
    if axis not in {"x", "y"}:
        raise ValueError(f"Unsupported axis {axis!r}")
    coord_name = axis
    pos_name = f"pos{axis}"
    if coord_name not in data_group or pos_name not in data_group:
        return None

    coord = np.asarray(data_group[coord_name][()], dtype=np.float64).reshape(-1)
    pos = np.asarray(data_group[pos_name][()], dtype=np.float64).reshape(-1)
    if coord.size != pos.size or coord.size < 2:
        return None

    finite = np.isfinite(coord) & np.isfinite(pos)
    if int(np.count_nonzero(finite)) < 2:
        return None
    coord = coord[finite]
    pos = pos[finite]

    slope, intercept = np.polyfit(coord, pos, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept) or abs(slope) <= 1e-12:
        return None
    return float(-intercept / slope)


def _beam_center_from_sansllb_positions(
    fin: h5py.File,
    *,
    entry: str,
    detector_name: str,
) -> tuple[Optional[float], Optional[float]]:
    """Infer beam_center_x/y from entry-level NXdata views such as left_data and bottom_data."""
    view_name = _sansllb_detector_view_name(detector_name)
    if view_name is None:
        return None, None
    group_path = f"{entry}/{view_name}"
    if group_path not in fin or not isinstance(fin[group_path], h5py.Group):
        return None, None
    data_group = fin[group_path]
    return (
        _beam_center_axis_from_positions(data_group, axis="x"),
        _beam_center_axis_from_positions(data_group, axis="y"),
    )


def _wavelength_dataset_to_angstrom(ds: Optional[h5py.Dataset]) -> Optional[float]:
    """Convert a raw SANS-LLB wavelength dataset to angstrom when units are known."""
    if ds is None:
        return None
    units = ds.attrs.get("units")
    units_s = _as_str(units).strip().lower() if units is not None else ""
    value = _as_float_scalar(ds[()])
    if units_s in {"nm", "nanometer", "nanometers", "nanometre", "nanometres"}:
        return value * NM_TO_ANGSTROM
    if units_s in {"a", "å", "angstrom", "angstroms"}:
        return value
    # Fallback: historical SANS-LLB exports store the selector wavelength in nm.
    if value <= 1.0:
        return value * NM_TO_ANGSTROM
    return value


def _wavelength_error_from_spread(wavelength: float, spread: float) -> float:
    """
    SANS_LLB provides incident_wavelength_spread which is often delta_lambda/lambda (~0.1).
    Some exports store it as a percentage (e.g. 11.6 for 11.6%).
    """
    if spread is None or np.isnan(spread):
        return float("nan")
    s = float(spread)
    if s < 1.0:
        return float(wavelength) * s
    if s <= 100.0:
        return float(wavelength) * (s / 100.0)
    return s


def _count_time_to_scalar(value) -> float:
    """Collapse scalar-like or vector count-time values to one exposure duration."""
    return _monitor_value_to_scalar(value)


def _monitor_value_to_scalar(value) -> float:
    """Collapse scalar-like or vector monitor values to one accumulated value."""
    if value is None:
        return float("nan")
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=np.float64)
        if arr.size == 0:
            return float("nan")
        if arr.size == 1:
            return float(arr.reshape(()))
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        return float(np.sum(finite))
    return _as_float_scalar(value)


def _normalize_monitor_mode(value, *, fallback: str = "monitor") -> str:
    """Normalize monitor mode to a schema-valid value."""
    mode = _as_str(value).strip().lower() if value is not None else ""
    if mode in _VALID_MONITOR_MODES:
        return mode
    fallback_mode = _as_str(fallback).strip().lower()
    if fallback_mode in _VALID_MONITOR_MODES:
        return fallback_mode
    return "monitor"


def _select_monitor_group(fin: h5py.File, candidates: List[str]) -> Optional[str]:
    """Return the first existing NXmonitor group among the given absolute paths."""
    for cand in candidates:
        if cand in fin and isinstance(fin[cand], h5py.Group):
            return cand
    return None


def _monitor_index(name: str) -> Optional[int]:
    """Extract the numeric suffix from ``monitorN`` names used by SANS-LLB."""
    if not name.startswith("monitor"):
        return None
    tail = name[len("monitor") :]
    if not tail:
        return None
    return int(tail) if tail.isdigit() else None


def _collect_monitor_sources(fin: h5py.File, entry: str) -> list[tuple[int, str]]:
    """Collect all raw SANS-LLB monitor groups and map them to output monitor indices."""
    entry_group = fin[entry]
    sources: list[tuple[int, str]] = []
    used_indices: set[int] = set()

    for key, obj in entry_group.items():
        if not isinstance(obj, h5py.Group):
            continue
        if _as_str(obj.attrs.get("NX_class")) != "NXmonitor":
            continue
        idx = _monitor_index(key)
        if idx is None:
            continue
        sources.append((idx, f"{entry}/{key}"))
        used_indices.add(idx)

    if f"{entry}/monitor" in fin and isinstance(fin[f"{entry}/monitor"], h5py.Group):
        fallback_idx = 0
        while fallback_idx in used_indices:
            fallback_idx += 1
        sources.append((fallback_idx, f"{entry}/monitor"))

    return sorted(sources, key=lambda item: item[0])


def _write_control_monitor(
    fin: h5py.File,
    entry: str,
    control_in: Optional[str],
    entry_out: h5py.Group,
    warnings: List[str],
) -> None:
    """Write the required /entry/control monitor from monitor2 or the input control group."""
    if control_in is not None:
        control_src = fin[control_in]
        control_out = _ensure_group(entry_out, "control", "NXmonitor")
        _write_dataset(control_out, "mode", "monitor", as_string=True)
        if "integral" in control_src:
            preset = _monitor_value_to_scalar(control_src["integral"][()])
            _write_dataset(control_out, "preset", preset)
            _write_dataset(control_out, "integral", preset)
        else:
            warnings.append(f"{control_in}: missing integral; writing NaN preset/integral in /entry/control")
            _write_dataset(control_out, "preset", float("nan"))
            _write_dataset(control_out, "integral", float("nan"))

        if f"{entry}/control/count_time" in fin:
            _write_dataset(control_out, "count_time", _count_time_to_scalar(fin[f"{entry}/control/count_time"][()]))
        return

    warnings.append("Missing /monitor2 and /control in input; writing NaN /entry/control.")
    control_out = _ensure_group(entry_out, "control", "NXmonitor")
    _write_dataset(control_out, "mode", "monitor", as_string=True)
    _write_dataset(control_out, "preset", float("nan"))
    _write_dataset(control_out, "integral", float("nan"))


def _write_instrument_monitors(
    fin: h5py.File,
    entry: str,
    monitor_sources: list[tuple[int, str]],
    inst_out: h5py.Group,
    warnings: List[str],
    notes: List[str],
) -> None:
    """Write all available SANS-LLB monitor groups under /entry/instrument/monitorN."""
    if not monitor_sources:
        notes.append("No monitor groups found in input; instrument monitors omitted.")
        return

    control_mode = (
        _normalize_monitor_mode(fin[f"{entry}/control/mode"][0])
        if f"{entry}/control/mode" in fin
        else "monitor"
    )
    control_preset = _monitor_value_to_scalar(fin[f"{entry}/control/preset"][()]) if f"{entry}/control/preset" in fin else float("nan")
    control_count_time = _count_time_to_scalar(fin[f"{entry}/control/count_time"][()]) if f"{entry}/control/count_time" in fin else None

    for monitor_idx, monitor_path in monitor_sources:
        mon_in = fin[monitor_path]
        mon_out = _ensure_group(inst_out, f"monitor{monitor_idx}", "NXmonitor")

        mode = _normalize_monitor_mode(mon_in["mode"][0], fallback=control_mode) if "mode" in mon_in else control_mode
        _write_dataset(mon_out, "mode", mode, as_string=True)

        preset = _monitor_value_to_scalar(mon_in["preset"][()]) if "preset" in mon_in else control_preset
        _write_dataset(mon_out, "preset", preset)

        if "integral" in mon_in:
            _write_dataset(mon_out, "integral", _monitor_value_to_scalar(mon_in["integral"][()]))

        if "data" in mon_in:
            _write_dataset(mon_out, "data", mon_in["data"][()])

        count_time = _count_time_to_scalar(mon_in["count_time"][()]) if "count_time" in mon_in else control_count_time
        if count_time is not None:
            _write_dataset(mon_out, "count_time", count_time)


def _sansllb_guide_state_from_selection(selection) -> str:
    """Map raw SANS-LLB guide selection strings to SCARLET NXguide/state."""
    if selection is None:
        return "in"
    s = _as_str(selection).strip().lower()
    if s == "ft":
        return "out"
    if s == "ng":
        return "in"
    return "in"


def _copy_aperture_snapshot(coll_out: h5py.Group, name: str, src: h5py.Group) -> None:
    """Copy the aperture fields needed by SCARLET from a detailed collimation element."""
    dst = _ensure_group(coll_out, name, _as_str(src.attrs.get("NX_class")))
    for field in ("x_gap", "y_gap", "diameter", "shape"):
        if field in src:
            units = src[field].attrs.get("units")
            units_s = _as_str(units) if units is not None else None
            _write_dataset(dst, field, src[field][()], units=units_s)
    if "transformations" in src and isinstance(src["transformations"], h5py.Group):
        src_tr = src["transformations"]
        dst_tr = _ensure_group(dst, "transformations", _as_str(src_tr.attrs.get("NX_class")))
        if "translation" in src_tr:
            units = src_tr["translation"].attrs.get("units")
            units_s = _as_str(units) if units is not None else None
            _write_dataset(dst_tr, "translation", src_tr["translation"][()], units=units_s)


def _derive_aperture1_snapshot(coll_out: h5py.Group, element_names: List[str], warnings: List[str]) -> None:
    """Derive the SCARLET aperture1 snapshot from the ordered collimation elements."""
    elements = coll_out["elements"]
    aperture_classes = {"NXslit", "NXpinhole", "NXaperture"}
    aperture_names = [name for name in element_names if _as_str(elements[name].attrs.get("NX_class")) in aperture_classes]

    if not aperture_names:
        warnings.append("No aperture-like elements available to derive aperture1/aperture2.")
        return

    aperture1_name: Optional[str] = None

    first_in_guide_idx: Optional[int] = None
    for idx, name in enumerate(element_names):
        element = elements[name]
        if _as_str(element.attrs.get("NX_class")) != "NXguide":
            continue
        if "state" in element and _as_str(element["state"][()]) == "in":
            first_in_guide_idx = idx
            break

    if first_in_guide_idx is not None:
        for idx in range(first_in_guide_idx - 1, -1, -1):
            candidate = element_names[idx]
            if candidate in aperture_names:
                aperture1_name = candidate
                break
        if aperture1_name is None:
            for idx in range(first_in_guide_idx + 1, len(element_names)):
                candidate = element_names[idx]
                if candidate in aperture_names:
                    aperture1_name = candidate
                    warnings.append(
                        "Could not find an upstream aperture before the first in-guide; using the nearest downstream aperture."
                    )
                    break

    if aperture1_name is None:
        aperture1_name = aperture_names[0]
        warnings.append("Could not derive aperture1 from guide states; using the first aperture in element_order.")

    _copy_aperture_snapshot(coll_out, "aperture1", elements[aperture1_name])


def _write_aperture2_from_sample_mask(
    fin: h5py.File,
    inst_in: str,
    coll_out: h5py.Group,
    warnings: List[str],
) -> None:
    """Map the raw SANS-LLB sample_mask group to the SCARLET aperture2 snapshot."""
    sample_mask_path = f"{inst_in}/sample_mask"
    if sample_mask_path not in fin or not isinstance(fin[sample_mask_path], h5py.Group):
        aperture2 = _ensure_group(coll_out, "aperture2", "NXslit")
        _write_dataset(aperture2, "x_gap", 0.01, units="m")
        _write_dataset(aperture2, "y_gap", 0.01, units="m")
        warnings.append("Missing sample_mask in SANS-LLB raw data; using default aperture2 = NXslit 10 mm x 10 mm.")
        return

    sample_mask = fin[sample_mask_path]
    raw_shape = _safe_get(sample_mask, "shape")
    shape = _as_str(raw_shape).strip().lower() if raw_shape is not None else ""
    size_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{sample_mask_path}/size"))
    size_y_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{sample_mask_path}/size_y"))

    if shape == "circle":
        if size_m is None:
            warnings.append(f"{sample_mask_path}: circle sample_mask missing size; using default aperture2.")
        else:
            aperture2 = _ensure_group(coll_out, "aperture2", "NXpinhole")
            _write_dataset(aperture2, "diameter", size_m, units="m")
            return

    if size_m is not None and size_y_m is not None:
        aperture2 = _ensure_group(coll_out, "aperture2", "NXslit")
        _write_dataset(aperture2, "x_gap", size_m, units="m")
        _write_dataset(aperture2, "y_gap", size_y_m, units="m")
        return

    aperture2 = _ensure_group(coll_out, "aperture2", "NXslit")
    _write_dataset(aperture2, "x_gap", 0.01, units="m")
    _write_dataset(aperture2, "y_gap", 0.01, units="m")
    warnings.append(
        f"{sample_mask_path}: incomplete sample_mask metadata (shape={shape or 'unknown'}); "
        "using default aperture2 = NXslit 10 mm x 10 mm."
    )


def _write_collimation(
    fin: h5py.File,
    inst_in: str,
    inst_out: h5py.Group,
    *,
    ap_shape,
    ap_xgap_m: Optional[float],
    ap_ygap_m: Optional[float],
    col_dist_m: Optional[float],
    col_len_m: Optional[float],
    warnings: List[str],
) -> None:
    """Write the full SANS-LLB collimation chain and derived aperture snapshots."""
    coll_out = _ensure_group(inst_out, "collimation", None)
    elements_out = _ensure_group(coll_out, "elements", None)

    element_order: List[bytes] = []
    collimation_distance_m: Optional[float] = None
    last_aperture_to_sample_distance_m: Optional[float] = None

    collimator_path = f"{inst_in}/collimator"
    if collimator_path in fin and isinstance(fin[collimator_path], h5py.Group):
        col_g = fin[collimator_path]

        def idx_for(prefix: str, name: str) -> Optional[int]:
            """Return the numeric suffix for names like ``slit0`` or ``guide2``."""
            if not name.startswith(prefix):
                return None
            tail = name[len(prefix) :]
            return int(tail) if tail.isdigit() else None

        slit_idxs = sorted(i for i in (idx_for("slit", k) for k in col_g.keys()) if i is not None)
        guide_idxs = sorted(i for i in (idx_for("guide", k) for k in col_g.keys()) if i is not None)

        max_guide = guide_idxs[-1] if guide_idxs else None
        if max_guide is not None:
            slit_idxs = [i for i in slit_idxs if i <= max_guide + 1]

        col_total_len_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{collimator_path}/length"))
        if col_total_len_m is None:
            col_total_len_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{collimator_path}/geometry/size"))
        total_L = float(col_total_len_m) if col_total_len_m is not None else 1.0
        collimation_distance_m = float(total_L)

        end_dist_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{collimator_path}/distance"))
        if end_dist_m is None:
            for i in slit_idxs:
                d_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{collimator_path}/slit{i}/distance"))
                if d_m is not None:
                    end_dist_m = d_m
        end_d = float(end_dist_m) if end_dist_m is not None else 0.1
        last_aperture_to_sample_distance_m = float(end_d)

        ordered: List[tuple[str, int]] = []
        if max_guide is not None:
            for i in range(max_guide + 1):
                if i in slit_idxs and f"slit{i}" in col_g:
                    ordered.append(("slit", i))
                if i in guide_idxs and f"guide{i}" in col_g:
                    ordered.append(("guide", i))
            last_slit = max_guide + 1
            if last_slit in slit_idxs and f"slit{last_slit}" in col_g:
                ordered.append(("slit", last_slit))
        else:
            max_i = max(slit_idxs[-1] if slit_idxs else -1, guide_idxs[-1] if guide_idxs else -1)
            for i in range(max_i + 1):
                if i in slit_idxs and f"slit{i}" in col_g:
                    ordered.append(("slit", i))
                if i in guide_idxs and f"guide{i}" in col_g:
                    ordered.append(("guide", i))

        if ordered:
            step = (total_L / (len(ordered) - 1)) if len(ordered) > 1 else 0.0
            start_d = end_d + total_L

            for k, (kind, i) in enumerate(ordered):
                name = f"{kind}{i}"
                src = col_g[name]

                if kind == "slit":
                    el = _ensure_group(elements_out, name, "NXslit")
                    x_gap_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{collimator_path}/{name}/x_gap"))
                    y_gap_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{collimator_path}/{name}/y_gap"))
                    if x_gap_m is None:
                        warnings.append(f"{collimator_path}/{name}: missing x_gap; writing NaN")
                        _write_dataset(el, "x_gap", float("nan"), units="m")
                    else:
                        _write_dataset(el, "x_gap", x_gap_m, units="m")

                    if y_gap_m is None:
                        warnings.append(f"{collimator_path}/{name}: missing y_gap; writing NaN")
                        _write_dataset(el, "y_gap", float("nan"), units="m")
                    else:
                        _write_dataset(el, "y_gap", y_gap_m, units="m")
                else:
                    el = _ensure_group(elements_out, name, "NXguide")
                    sel = src["selection"][()] if "selection" in src else None
                    _write_dataset(el, "state", _sansllb_guide_state_from_selection(sel), as_string=True)
                    if "m_value" in src:
                        _write_dataset(el, "m_value", _as_float_scalar(src["m_value"][()]))

                d_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{collimator_path}/{name}/distance"))
                dist_m = float(d_m) if d_m is not None else (start_d - k * step)
                tr = _ensure_group(el, "transformations", "NXtransformations")
                _write_dataset(tr, "translation", np.array([0.0, 0.0, -dist_m], dtype=float), units="m")
                element_order.append(name.encode())
        else:
            warnings.append("No slit*/guide* entries found under input collimator; using heuristic collimation.")

    if not element_order:
        ap_el = _ensure_group(elements_out, "aperture", "NXaperture")
        if ap_shape is not None:
            _write_dataset(ap_el, "shape", ap_shape)
        if ap_xgap_m is not None:
            _write_dataset(ap_el, "x_gap", ap_xgap_m, units="m")
        if ap_ygap_m is not None:
            _write_dataset(ap_el, "y_gap", ap_ygap_m, units="m")

        ap_tr = _ensure_group(ap_el, "transformations", "NXtransformations")
        _write_dataset(ap_tr, "translation", np.array([0.0, 0.0, -0.1], dtype=float), units="m")
        element_order.append(b"aperture")
        last_aperture_to_sample_distance_m = 0.1

        guide_el = _ensure_group(elements_out, "collimator", "NXguide")
        _write_dataset(guide_el, "state", "in", as_string=True)
        if col_len_m is not None:
            _write_dataset(guide_el, "length", col_len_m, units="m")
        if col_dist_m is not None:
            _write_dataset(guide_el, "distance", col_dist_m, units="m")

        guide_tr = _ensure_group(guide_el, "transformations", "NXtransformations")
        L = col_len_m if col_len_m is not None else 1.0
        _write_dataset(guide_tr, "translation", np.array([0.0, 0.0, -max(L, 0.1)], dtype=float), units="m")
        element_order.append(b"collimator")
        collimation_distance_m = float(max(L, 0.0))

    if collimation_distance_m is None:
        collimation_distance_m = float(max(col_len_m, 0.0)) if col_len_m is not None else 1.0
    if last_aperture_to_sample_distance_m is None:
        last_aperture_to_sample_distance_m = 0.1

    _derive_aperture1_snapshot(
        coll_out,
        [name.decode() if isinstance(name, (bytes, bytearray)) else str(name) for name in element_order],
        warnings,
    )
    _write_aperture2_from_sample_mask(fin, inst_in, coll_out, warnings)
    _write_dataset(coll_out, "collimation_distance", float(collimation_distance_m), units="m")
    _write_dataset(coll_out, "last_aperture_to_sample_distance", float(last_aperture_to_sample_distance_m), units="m")
    _write_dataset(coll_out, "element_order", np.array(element_order, dtype="S"))

def convert_sansllb_to_scarlet_nxsas_raw(
    input_path: str | Path,
    output_path: str | Path,
    *,
    entry_in: Optional[str] = None,
    overwrite: bool = False,
) -> ConvertReport:
    """
    Convert a SANS_LLB NeXus file to SCARLET NXsas_raw (monochromatic profile).

    Output will follow SCARLET convention:
      /entry (NXentry)
        definition="NXsas_raw"
        /sample
        /instrument
          /geometry
          /monochromator
          /collimation (SCARLET detailed)
          /detector0..N
          /monitor0..N (optional)
        /control (NXmonitor, required by SCARLET schema)
        /data0..N (NXdata softlinks)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    notes: List[str] = []
    warnings: List[str] = []

    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            raise FileExistsError(f"Output file exists: {output_path}")

    with h5py.File(input_path, "r") as fin:
        entry = _pick_entry(fin, preferred=entry_in)
        notes.append(f"Using input entry: {entry}")

        # --- Input shortcuts ---
        sample_in = f"{entry}/sample"

        # SANS_LLB instrument group name varies between exports (/instrument vs /SANS-LLB)
        inst_in = f"{entry}/instrument"
        if inst_in not in fin:
            for k, obj in fin[entry].items():
                if isinstance(obj, h5py.Group) and _as_str(obj.attrs.get("NX_class")) == "NXinstrument":
                    inst_in = f"{entry}/{k}"
                    break
        if inst_in not in fin:
            raise ValueError("No NXinstrument group found under input entry.")

        # SANS-LLB uses monitor2 as the acquisition preset source.
        control_in = _select_monitor_group(fin, [f"{entry}/monitor2", f"{entry}/control"])
        monitor_sources = _collect_monitor_sources(fin, entry)
        acquisition_time = None
        if f"{entry}/control/count_time" in fin:
            acquisition_time = _count_time_to_scalar(fin[f"{entry}/control/count_time"][()])
        elif control_in is not None and f"{control_in}/count_time" in fin:
            acquisition_time = _count_time_to_scalar(fin[f"{control_in}/count_time"][()])

        # --- Read key values ---
        wavelength_ds = _safe_get_dataset(fin, f"{inst_in}/source/incident_wavelength")
        spread = _safe_get(fin, f"{inst_in}/source/incident_wavelength_spread")
        if wavelength_ds is None:
            wavelength_ds = _safe_get_dataset(fin, f"{inst_in}/velocity_selector/wavelength")
            spread = _safe_get(fin, f"{inst_in}/velocity_selector/wavelength_spread")
        if wavelength_ds is None:
            warnings.append("Missing incident_wavelength; monochromator/wavelength will be NaN.")
            wavelength = float("nan")
        else:
            wavelength = _wavelength_dataset_to_angstrom(wavelength_ds)

        wavelength_error = _wavelength_error_from_spread(wavelength, _as_float_scalar(spread))

        # Collimation / aperture / collimator
        ap_shape = _safe_get(fin, f"{inst_in}/aperture/shape")
        ap_xgap_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/aperture/x_gap"))
        ap_ygap_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/aperture/y_gap"))
        col_dist_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/collimator/distance"))
        col_len_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/collimator/length"))
        if ap_xgap_m is None:
            ap_xgap_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/collimator/slit0/x_gap"))
        if ap_ygap_m is None:
            ap_ygap_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/collimator/slit0/y_gap"))

        # Detector list
        det_names: List[str] = []
        inst_g = fin[inst_in]
        for k, obj in inst_g.items():
            if k.startswith("detector") and isinstance(obj, h5py.Group):
                det_names.append(k)
        if det_names:
            det_names = sorted(det_names, key=lambda s: int(s.replace("detector", "")))
        else:
            for k, obj in inst_g.items():
                if k.endswith("_detector") and isinstance(obj, h5py.Group) and _as_str(obj.attrs.get("NX_class")) == "NXdetector":
                    det_names.append(k)
            preferred = {
                "central_detector": 0,
                "left_detector": 1,
                "bottom_detector": 2,
                "right_detector": 3,
                "top_detector": 4,
            }
            det_names = sorted(det_names, key=lambda s: (preferred.get(s, 99), s))
        if not det_names:
            raise ValueError(f"No detectors found under {inst_in}.")

        # --- Write output ---
        with h5py.File(output_path, "w") as fout:
            # /raw_data
            entry_out = _ensure_group(fout, "raw_data", "NXentry")
            _write_dataset(entry_out, "definition", "NXsas_raw", as_string=True)
            _write_dataset(entry_out, "schema_version", "1.3", as_string=True)

            # /entry/sample
            sample_out = _ensure_group(entry_out, "sample", "NXsample")
            if sample_in in fin:
                # copy a few common fields
                for key in ("name", "thickness", "transmission"):
                    p = f"{sample_in}/{key}"
                    if p in fin:
                        _write_dataset(sample_out, key, fin[p][()])
            else:
                warnings.append("Missing NXsample group in input; creating empty /entry/sample.")
                _write_dataset(sample_out, "name", "unknown", as_string=True)

            # /entry/instrument
            inst_out = _ensure_group(entry_out, "instrument", "NXinstrument")

            _write_control_monitor(fin, entry, control_in, entry_out, warnings)

            # geometry (SCARLET reserved)
            geom_out = _ensure_group(inst_out, "geometry", None)
            _write_dataset(geom_out, "origin_definition", "sample center", as_string=True)
            _write_dataset(geom_out, "axis_convention", "+z downstream, +x beam-right, +y up", as_string=True)

            # source (optional copy)
            if f"{inst_in}/source" in fin:
                src_out = _ensure_group(inst_out, "source", "NXsource")
                src_in = fin[f"{inst_in}/source"]
                for key in ("description", "type", "probe", "shape"):
                    if key in src_in:
                        _write_dataset(src_out, key, src_in[key][()])
                # Optionally store beam size
                for key in ("beam_size_x", "beam_size_y"):
                    beam_size_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/source/{key}"))
                    if beam_size_m is not None:
                        _write_dataset(src_out, key, beam_size_m, units="m")

            # monochromator (required by SCARLET mono profile)
            mono_out = _ensure_group(inst_out, "monochromator", "NXmonochromator")
            _write_dataset(mono_out, "wavelength", wavelength)  # Angstrom
            if not np.isnan(wavelength_error):
                _write_dataset(mono_out, "wavelength_error", wavelength_error)

            _write_collimation(
                fin,
                inst_in,
                inst_out,
                ap_shape=ap_shape,
                ap_xgap_m=ap_xgap_m,
                ap_ygap_m=ap_ygap_m,
                col_dist_m=col_dist_m,
                col_len_m=col_len_m,
                warnings=warnings,
            )

            # detectors + NXdata views
            for i, det_name in enumerate(det_names):
                det_in = fin[f"{inst_in}/{det_name}"]

                det_out = _ensure_group(inst_out, f"detector{i}", "NXdetector")

                # required by SCARLET
                data = det_in["data"][()]
                if "dead_time" in det_in:
                    dead_time = det_in["dead_time"][()]
                elif "deadtime" in det_in:
                    dead_time = det_in["deadtime"][()]
                else:
                    dead_time = float("nan")
                if dead_time is None:
                    dead_time = float("nan")
                corrected_data, deadtime_corrected = correct_detector_data_for_deadtime(
                    data,
                    acquisition_time=acquisition_time,
                    dead_time=_as_float_scalar(dead_time),
                    detector_name=det_name,
                    warnings=warnings,
                )
                _write_dataset(det_out, "data", corrected_data)

                xpix_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/{det_name}/x_pixel_size"))
                ypix_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/{det_name}/y_pixel_size"))
                _write_dataset(det_out, "x_pixel_size", xpix_m if xpix_m is not None else float("nan"), units="m")
                _write_dataset(det_out, "y_pixel_size", ypix_m if ypix_m is not None else float("nan"), units="m")

                derived_beam_center_x, derived_beam_center_y = _beam_center_from_sansllb_positions(
                    fin,
                    entry=entry,
                    detector_name=det_name,
                )

                if derived_beam_center_x is not None:
                    _write_dataset(det_out, "beam_center_x", derived_beam_center_x)
                elif "beam_center_x" in det_in:
                    _write_dataset(det_out, "beam_center_x", _as_float_scalar(det_in["beam_center_x"][()]))
                else:
                    bx = float("nan")
                    if hasattr(corrected_data, "shape") and len(getattr(corrected_data, "shape", ())) >= 2:
                        bx = (float(corrected_data.shape[-1]) - 1.0) / 2.0
                    warnings.append(f"{det_name}: missing beam_center_x; writing {bx}")
                    _write_dataset(det_out, "beam_center_x", bx)

                if derived_beam_center_y is not None:
                    _write_dataset(det_out, "beam_center_y", derived_beam_center_y)
                elif "beam_center_y" in det_in:
                    _write_dataset(det_out, "beam_center_y", _as_float_scalar(det_in["beam_center_y"][()]))
                else:
                    by = float("nan")
                    if hasattr(corrected_data, "shape") and len(getattr(corrected_data, "shape", ())) >= 2:
                        by = (float(corrected_data.shape[-2]) - 1.0) / 2.0
                    warnings.append(f"{det_name}: missing beam_center_y; writing {by}")
                    _write_dataset(det_out, "beam_center_y", by)

                _write_dataset(det_out, "dead_time", _as_float_scalar(dead_time))
                _write_dataset(det_out, "deadtime_corrected", deadtime_corrected)

                # optional fields (copy if present)
                for opt in (
                    "data_errors",
                    "description",
                    "distance",
                    "pixel_mask",
                    "pixel_mask_applied",
                    "countrate_correction_applied",
                ):
                    if opt in det_in:
                        val = det_in[opt][()]
                        # convert distance to meters
                        if opt == "distance":
                            val_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/{det_name}/{opt}"))
                            _write_dataset(det_out, opt, val_m if val_m is not None else float("nan"), units="m")
                            continue
                        _write_dataset(det_out, opt, val)

                # local_name (optional SCARLET)
                _write_dataset(det_out, "local_name", det_name, as_string=True)

                # transformations (required by SCARLET)
                tr_out = _ensure_group(det_out, "transformations", "NXtransformations")
                if "x_position" in det_in:
                    dx_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/{det_name}/x_position"))
                elif "x_offset" in det_in:
                    dx_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/{det_name}/x_offset"))
                else:
                    dx_m = 0.0

                if "y_position" in det_in:
                    dy_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/{det_name}/y_position"))
                elif "y_offset" in det_in:
                    dy_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/{det_name}/y_offset"))
                else:
                    dy_m = 0.0

                if "distance" in det_in:
                    dz_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/{det_name}/distance"))
                elif "z_offset" in det_in:
                    dz_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/{det_name}/z_offset"))
                else:
                    dz_m = 0.0

                translation = np.array([float(dx_m), float(dy_m), float(dz_m)], dtype=float)
                _write_dataset(tr_out, "translation", translation, units="m")

                # NXdata view
                data_out = _ensure_group(entry_out, f"data{i}", "NXdata")
                data_out.attrs["signal"] = np.bytes_("counts")

                # softlink counts
                data_out["counts"] = h5py.SoftLink(f"/raw_data/instrument/detector{i}/data")

                # optional errors link
                if "data_errors" in det_out:
                    data_out["counts_errors"] = h5py.SoftLink(f"/raw_data/instrument/detector{i}/data_errors")

            _write_instrument_monitors(fin, entry, monitor_sources, inst_out, warnings, notes)

    return ConvertReport(
        input_file=input_path,
        output_file=output_path,
        entry_in=entry,
        notes=notes,
        warnings=warnings,
    )
