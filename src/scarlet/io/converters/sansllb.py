from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


MM_TO_M = 1e-3


@dataclass
class ConvertReport:
    input_file: Path
    output_file: Path
    entry_in: str
    notes: List[str]
    warnings: List[str]


def _as_str(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode(errors="replace")
    if isinstance(x, np.ndarray) and x.dtype.kind in {"S", "O"} and x.size == 1:
        return _as_str(x[0])
    return str(x)


def _pick_entry(fin: h5py.File, preferred: Optional[str] = None) -> str:
    """
    Pick an input NXentry.
    For SANS_LLB files we usually want /entry0 (raw), not /entry1 (processed).
    """
    if preferred and preferred in fin:
        return preferred

    # Common patterns: /entry0, /entry1, /entry
    for cand in ("/entry0", "/entry", "/entry1"):
        if cand in fin:
            return cand

    # Fallback: first NXentry-like group
    for k in fin.keys():
        p = f"/{k}"
        if isinstance(fin[p], h5py.Group):
            nx = fin[p].attrs.get("NX_class", None)
            if nx is not None and _as_str(nx) == "NXentry":
                return p

    raise ValueError("No NXentry found in file.")


def _safe_get(fin: h5py.File, path: str):
    return fin[path][()] if path in fin else None


def _ensure_group(g: h5py.Group, name: str, nx_class: Optional[str] = None) -> h5py.Group:
    gg = g.create_group(name)
    if nx_class:
        gg.attrs["NX_class"] = np.bytes_(nx_class)
    return gg


def _write_dataset(g: h5py.Group, name: str, data, *, as_string: bool = False) -> h5py.Dataset:
    if as_string:
        data = np.bytes_(str(data))
    return g.create_dataset(name, data=data)

def _as_float_scalar(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, np.ndarray) and x.size == 1:
        return float(x.reshape(()))
    return float(x)


def _mm_or_m_to_m(value: float) -> float:
    """
    Heuristic: SANS_LLB values in files are often in mm (pixel_size=5.0).
    We'll treat values > 5 as mm and convert to meters.
    """
    if value is None:
        return float("nan")
    v = _as_float_scalar(value)
    # If it's already in meters (e.g. 0.005), keep it.
    if abs(v) <= 5.0:
        return v
    return v * MM_TO_M


def _wavelength_error_from_spread(wavelength: float, spread: float) -> float:
    """
    SANS_LLB provides incident_wavelength_spread which is often delta_lambda/lambda (~0.1).
    If spread < 1, interpret as relative; else absolute.
    """
    if spread is None or np.isnan(spread):
        return float("nan")
    s = float(spread)
    if s < 1.0:
        return float(wavelength) * s
    return s


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
        /control (optional)
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

        # monitors vary; accept a few common locations
        monitor_in: Optional[str] = None
        for cand in (f"{entry}/monitor", f"{entry}/monitor0", f"{entry}/control"):
            if cand in fin:
                monitor_in = cand
                break

        # --- Read key values ---
        wavelength = _safe_get(fin, f"{inst_in}/source/incident_wavelength")
        spread = _safe_get(fin, f"{inst_in}/source/incident_wavelength_spread")
        if wavelength is None:
            wavelength = _safe_get(fin, f"{inst_in}/velocity_selector/wavelength")
            spread = _safe_get(fin, f"{inst_in}/velocity_selector/wavelength_spread")
        if wavelength is None:
            warnings.append("Missing incident_wavelength; monochromator/wavelength will be NaN.")
            wavelength = float("nan")
        else:
            wavelength = _as_float_scalar(wavelength)

        wavelength_error = _wavelength_error_from_spread(wavelength, _as_float_scalar(spread))

        # Collimation / aperture / collimator
        ap_shape = _safe_get(fin, f"{inst_in}/aperture/shape")
        ap_xgap = _safe_get(fin, f"{inst_in}/aperture/x_gap")
        ap_ygap = _safe_get(fin, f"{inst_in}/aperture/y_gap")
        col_dist = _safe_get(fin, f"{inst_in}/collimator/distance")
        col_len = _safe_get(fin, f"{inst_in}/collimator/length")
        if ap_xgap is None:
            ap_xgap = _safe_get(fin, f"{inst_in}/collimator/slit0/x_gap")
        if ap_ygap is None:
            ap_ygap = _safe_get(fin, f"{inst_in}/collimator/slit0/y_gap")

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
            # /entry
            entry_out = _ensure_group(fout, "entry", "NXentry")
            _write_dataset(entry_out, "definition", "NXsas_raw", as_string=True)
            _write_dataset(entry_out, "schema_version", "1.0", as_string=True)

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
                    if key in src_in:
                        _write_dataset(src_out, key, _mm_or_m_to_m(src_in[key][()]))

            # monochromator (required by SCARLET mono profile)
            mono_out = _ensure_group(inst_out, "monochromator", "NXmonochromator")
            _write_dataset(mono_out, "wavelength", wavelength)  # Angstrom
            if not np.isnan(wavelength_error):
                _write_dataset(mono_out, "wavelength_error", wavelength_error)

            # collimation (SCARLET detailed)
            coll_out = _ensure_group(inst_out, "collimation", None)
            elements_out = _ensure_group(coll_out, "elements", None)

            element_order: List[bytes] = []

            collimator_path = f"{inst_in}/collimator"
            if collimator_path in fin and isinstance(fin[collimator_path], h5py.Group):
                col_g = fin[collimator_path]

                def _idx(prefix: str, name: str) -> Optional[int]:
                    if not name.startswith(prefix):
                        return None
                    tail = name[len(prefix) :]
                    return int(tail) if tail.isdigit() else None

                slit_idxs = sorted(
                    i for i in (_idx("slit", k) for k in col_g.keys()) if i is not None  # type: ignore[arg-type]
                )
                guide_idxs_all = sorted(
                    i for i in (_idx("guide", k) for k in col_g.keys()) if i is not None  # type: ignore[arg-type]
                )

                # SANS-LLB exports may include non-guide elements under guide* (e.g. selection="ft").
                # Keep only neutron guide segments (selection="ng") when selection is available.
                guide_idxs: List[int] = []
                for i in guide_idxs_all:
                    gname = f"guide{i}"
                    gg = col_g[gname]
                    if "selection" in gg:
                        sel = _as_str(gg["selection"][()])
                        if sel != "ng":
                            continue
                    guide_idxs.append(i)

                max_guide = guide_idxs[-1] if guide_idxs else None

                # If we have guides, we expect a slit upstream of each guide and one final slit downstream:
                # slit0, guide0, slit1, guide1, ..., slitN (with N = max_guide + 1).
                if max_guide is not None:
                    slit_idxs = [i for i in slit_idxs if i <= max_guide + 1]

                # Collimation length (used to distribute elements if individual distances are missing)
                col_total_len = _safe_get(fin, f"{collimator_path}/length")
                if col_total_len is None:
                    col_total_len = _safe_get(fin, f"{collimator_path}/geometry/size")
                total_L = float(_mm_or_m_to_m(col_total_len)) if col_total_len is not None else 1.0

                # Distance from sample to the downstream-most collimation element.
                # Prefer an explicit /collimator/distance; else use the maximum element distance if present.
                end_dist = _safe_get(fin, f"{collimator_path}/distance")
                if end_dist is None:
                    for i in slit_idxs:
                        d = _safe_get(fin, f"{collimator_path}/slit{i}/distance")
                        if d is not None:
                            end_dist = d
                end_d = float(_mm_or_m_to_m(end_dist)) if end_dist is not None else 0.1

                # Build physical order: slit0, guide0, slit1, guide1, ...
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
                            x_gap = _safe_get(fin, f"{collimator_path}/{name}/x_gap")
                            y_gap = _safe_get(fin, f"{collimator_path}/{name}/y_gap")
                            if x_gap is not None:
                                _write_dataset(el, "x_gap", _mm_or_m_to_m(x_gap))
                            if y_gap is not None:
                                _write_dataset(el, "y_gap", _mm_or_m_to_m(y_gap))
                        else:
                            el = _ensure_group(elements_out, name, "NXguide")
                            if "m_value" in src:
                                _write_dataset(el, "m_value", _as_float_scalar(src["m_value"][()]))
                            if "selection" in src:
                                _write_dataset(el, "selection", src["selection"][()])

                        d = _safe_get(fin, f"{collimator_path}/{name}/distance")
                        dist_m = float(_mm_or_m_to_m(d)) if d is not None else (start_d - k * step)

                        tr = _ensure_group(el, "transformations", "NXtransformations")
                        _write_dataset(tr, "translation", np.array([0.0, 0.0, -dist_m], dtype=float))
                        element_order.append(name.encode())
                else:
                    warnings.append("No slit*/guide* entries found under input collimator; using heuristic collimation.")

            if not element_order:
                # Heuristic fallback (kept for legacy/partial files)
                ap_el = _ensure_group(elements_out, "aperture", "NXaperture")
                if ap_shape is not None:
                    _write_dataset(ap_el, "shape", ap_shape)
                if ap_xgap is not None:
                    _write_dataset(ap_el, "x_gap", _mm_or_m_to_m(ap_xgap))
                if ap_ygap is not None:
                    _write_dataset(ap_el, "y_gap", _mm_or_m_to_m(ap_ygap))

                ap_tr = _ensure_group(ap_el, "transformations", "NXtransformations")
                _write_dataset(ap_tr, "translation", np.array([0.0, 0.0, -0.1], dtype=float))
                element_order.append(b"aperture")

                guide_el = _ensure_group(elements_out, "collimator", "NXguide")
                if col_len is not None:
                    _write_dataset(guide_el, "length", _mm_or_m_to_m(col_len))
                if col_dist is not None:
                    _write_dataset(guide_el, "distance", _mm_or_m_to_m(col_dist))

                guide_tr = _ensure_group(guide_el, "transformations", "NXtransformations")
                L = _mm_or_m_to_m(col_len) if col_len is not None else 1.0
                _write_dataset(guide_tr, "translation", np.array([0.0, 0.0, -max(L, 0.1)], dtype=float))
                element_order.append(b"collimator")

            _write_dataset(coll_out, "element_order", np.array(element_order, dtype="S"))

            # detectors + NXdata views
            for i, det_name in enumerate(det_names):
                det_in = fin[f"{inst_in}/{det_name}"]

                det_out = _ensure_group(inst_out, f"detector{i}", "NXdetector")

                # required by SCARLET
                data = det_in["data"][()]
                _write_dataset(det_out, "data", data)

                xpix = det_in["x_pixel_size"][()] if "x_pixel_size" in det_in else float("nan")
                ypix = det_in["y_pixel_size"][()] if "y_pixel_size" in det_in else float("nan")
                _write_dataset(det_out, "x_pixel_size", _mm_or_m_to_m(xpix))
                _write_dataset(det_out, "y_pixel_size", _mm_or_m_to_m(ypix))

                if "beam_center_x" in det_in:
                    _write_dataset(det_out, "beam_center_x", det_in["beam_center_x"][()])
                else:
                    bx = float("nan")
                    if hasattr(data, "shape") and len(getattr(data, "shape", ())) >= 2:
                        bx = (float(data.shape[-1]) - 1.0) / 2.0
                    warnings.append(f"{det_name}: missing beam_center_x; writing {bx}")
                    _write_dataset(det_out, "beam_center_x", bx)

                if "beam_center_y" in det_in:
                    _write_dataset(det_out, "beam_center_y", det_in["beam_center_y"][()])
                else:
                    by = float("nan")
                    if hasattr(data, "shape") and len(getattr(data, "shape", ())) >= 2:
                        by = (float(data.shape[-2]) - 1.0) / 2.0
                    warnings.append(f"{det_name}: missing beam_center_y; writing {by}")
                    _write_dataset(det_out, "beam_center_y", by)

                if "dead_time" in det_in:
                    dead_time = det_in["dead_time"][()]
                elif "deadtime" in det_in:
                    dead_time = det_in["deadtime"][()]
                else:
                    dead_time = float("nan")
                if dead_time is None:
                    dead_time = float("nan")
                _write_dataset(det_out, "dead_time", _as_float_scalar(dead_time))

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
                            val = _mm_or_m_to_m(val)
                        _write_dataset(det_out, opt, val)

                # local_name (optional SCARLET)
                _write_dataset(det_out, "local_name", det_name, as_string=True)

                # transformations (required by SCARLET)
                tr_out = _ensure_group(det_out, "transformations", "NXtransformations")
                if "x_position" in det_in:
                    dx = det_in["x_position"][()]
                elif "x_offset" in det_in:
                    dx = det_in["x_offset"][()]
                else:
                    dx = 0.0

                if "y_position" in det_in:
                    dy = det_in["y_position"][()]
                elif "y_offset" in det_in:
                    dy = det_in["y_offset"][()]
                else:
                    dy = 0.0

                if "distance" in det_in:
                    dz = det_in["distance"][()]
                elif "z_offset" in det_in:
                    dz = det_in["z_offset"][()]
                else:
                    dz = 0.0

                translation = np.array([_mm_or_m_to_m(dx), _mm_or_m_to_m(dy), _mm_or_m_to_m(dz)], dtype=float)
                _write_dataset(tr_out, "translation", translation)

                # NXdata view (SCARLET expects /entry/dataN with counts -> link to detector/data)
                data_out = _ensure_group(entry_out, f"data{i}", "NXdata")
                data_out.attrs["signal"] = np.bytes_("counts")

                # softlink counts
                data_out["counts"] = h5py.SoftLink(f"/entry/instrument/detector{i}/data")

                # optional errors link
                if "data_errors" in det_out:
                    data_out["counts_errors"] = h5py.SoftLink(f"/entry/instrument/detector{i}/data_errors")

            # monitors (optional)
            if monitor_in and monitor_in in fin:
                mon_in = fin[monitor_in]
                mon_out = _ensure_group(inst_out, "monitor0", "NXmonitor")

                # NXmonitor recommended fields; your schema requires mode+preset+integral|data
                mode = _as_str(mon_in["mode"][0]) if "mode" in mon_in else "monitor"
                _write_dataset(mon_out, "mode", mode, as_string=True)

                # preset not present in SANS_LLB => write NaN but keep schema satisfied
                _write_dataset(mon_out, "preset", float("nan"))

                if "integral" in mon_in:
                    _write_dataset(mon_out, "integral", _as_float_scalar(mon_in["integral"][()]))

                if "count_time" in mon_in:
                    _write_dataset(mon_out, "count_time", float(mon_in["count_time"][()]))

            else:
                notes.append("No /monitor group found in input; monitors omitted.")

    return ConvertReport(
        input_file=input_path,
        output_file=output_path,
        entry_in=entry,
        notes=notes,
        warnings=warnings,
    )
