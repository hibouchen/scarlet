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
    write_dataset as _write_dataset,
)
from ._units import MM_TO_M, mm_to_m as _mm_to_m


def _sam_instrument_path(fin: h5py.File, entry: str) -> str:
    eg = fin[entry]
    for k, obj in eg.items():
        if isinstance(obj, h5py.Group) and _as_str(obj.attrs.get("NX_class")) == "NXinstrument":
            return f"{entry}/{k}"
    raise ValueError("No NXinstrument group found under entry.")


def _sam_read_detector_counts(fin: h5py.File, entry: str, warnings: List[str]) -> np.ndarray:
    """
    SAM files come in (at least) two variants:
    - scan files:  /entry0/data_scan/detector_data/data        shape (n, ny, nx)
    - normal run:  /entry0/data1/detector_data                 shape (ny, nx, 1)
    Return a 2D array (ny, nx). If a scan stack is found, sum over the first axis.
    """
    scan_path = f"{entry}/data_scan/detector_data/data"
    if scan_path in fin:
        data = fin[scan_path][()]
        a = np.asarray(data)
        if a.ndim == 3:
            # (n, ny, nx) -> sum
            warnings.append(f"{scan_path}: summing {a.shape[0]} frames into one detector image")
            return np.sum(a, axis=0)
        if a.ndim == 2:
            return a
        raise ValueError(f"Unsupported detector data shape at {scan_path}: {a.shape}")

    # common NXdata layout: /entry0/data1/detector_data
    for cand in (f"{entry}/data1/detector_data", f"{entry}/data0/detector_data"):
        if cand in fin:
            a = np.asarray(fin[cand][()])
            if a.ndim == 3 and a.shape[-1] == 1:
                return a[..., 0]
            if a.ndim == 2:
                return a
            raise ValueError(f"Unsupported detector data shape at {cand}: {a.shape}")

    # fallback: find a dataset named 'detector_data' under /entry*/data*
    eg = fin[entry]
    for k, obj in eg.items():
        if not (isinstance(obj, h5py.Group) and k.lower().startswith("data")):
            continue
        if "detector_data" in obj and isinstance(obj["detector_data"], h5py.Dataset):
            a = np.asarray(obj["detector_data"][()])
            if a.ndim == 3 and a.shape[-1] == 1:
                return a[..., 0]
            if a.ndim == 2:
                return a
            raise ValueError(f"Unsupported detector data shape at {entry}/{k}/detector_data: {a.shape}")

    raise ValueError("Could not locate SAM detector data.")


def _sam_monitor_mode(value) -> str:
    mode = _as_str(value).strip().lower()
    if mode == "time":
        return "timer"
    if mode in {"monitor", "timer"}:
        return mode
    return "timer"


def _sam_monitor_integral(fin: h5py.File, entry: str) -> float:
    integral = _safe_get(fin, f"{entry}/monitor/integral")
    if integral is not None:
        return _as_float_scalar(integral)

    data = _safe_get(fin, f"{entry}/monitor/data")
    if data is not None:
        a = np.asarray(data, dtype=float)
        if a.size == 1:
            return float(a.reshape(()))
        return float(np.sum(a))

    monsum = _safe_get(fin, f"{entry}/monitor/monsum")
    if monsum is not None:
        return _as_float_scalar(monsum)

    return float("nan")


def _sam_detector_dead_time(fin: h5py.File, inst_in: str) -> float:
    for dataset_name in ("dead_time", "deadtime"):
        value = _safe_get(fin, f"{inst_in}/detector/{dataset_name}")
        if value is not None:
            return _as_float_scalar(value)
    return float("nan")


def _sam_guide_state(value) -> str:
    state = _as_str(value).strip().lower()
    if state == "in":
        return "in"
    if state == "out":
        return "out"
    return "in"


def _copy_aperture_snapshot(coll_out: h5py.Group, name: str, src: h5py.Group) -> None:
    dst = _ensure_group(coll_out, name, _as_str(src.attrs.get("NX_class")))
    for field in ("x_gap", "y_gap", "diameter"):
        if field in src:
            _write_dataset(dst, field, src[field][()])
    if "transformations" in src and isinstance(src["transformations"], h5py.Group):
        src_tr = src["transformations"]
        dst_tr = _ensure_group(dst, "transformations", _as_str(src_tr.attrs.get("NX_class")))
        if "translation" in src_tr:
            _write_dataset(dst_tr, "translation", src_tr["translation"][()])


def _derive_aperture_snapshots(coll_out: h5py.Group, element_names: List[str], warnings: List[str]) -> None:
    elements = coll_out["elements"]
    aperture_classes = {"NXslit", "NXpinhole", "NXaperture"}
    aperture_names = [name for name in element_names if _as_str(elements[name].attrs.get("NX_class")) in aperture_classes]

    if not aperture_names:
        warnings.append("No aperture-like elements available to derive aperture1/aperture2.")
        return

    aperture2_name = aperture_names[-1]
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
    _copy_aperture_snapshot(coll_out, "aperture2", elements[aperture2_name])


def convert_sam_to_scarlet_nxsas_raw(
    input_path: str | Path,
    output_path: str | Path,
    *,
    entry_in: Optional[str] = None,
    overwrite: bool = False,
) -> ConvertReport:
    """
    Convert a SAM NeXus file to SCARLET NXsas_raw (monochromatic profile).
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

        inst_in = _sam_instrument_path(fin, entry)

        # monochromator / selector wavelength
        wl = _safe_get(fin, f"{inst_in}/Selector/wavelength")
        wavelength = _as_float_scalar(wl) if wl is not None else float("nan")
        if np.isnan(wavelength):
            warnings.append("Missing selector wavelength; monochromator/wavelength will be NaN.")

        # sample-detector distance (assumed meters in SAM files)
        det_dist = _safe_get(fin, f"{inst_in}/Distance/S2_Sample")
        det_z_m = _as_float_scalar(det_dist) if det_dist is not None else float("nan")

        # detector pixel sizes (SAM stores these in mm)
        px_x = _safe_get(fin, f"{inst_in}/detector/pixel_size_x")
        px_y = _safe_get(fin, f"{inst_in}/detector/pixel_size_y")
        x_pixel_size_m = _mm_to_m(px_x) if px_x is not None else float("nan")
        y_pixel_size_m = _mm_to_m(px_y) if px_y is not None else float("nan")

        # counts
        counts = _sam_read_detector_counts(fin, entry, warnings)
        detector_dead_time = _sam_detector_dead_time(fin, inst_in)

        # Collimation distance in SAM is provided by /collimation/position.
        # Keep sourceDistance only as a fallback for older variants.
        col_L = _safe_get(fin, f"{inst_in}/collimation/position")
        if col_L is None:
            col_L = _safe_get(fin, f"{inst_in}/collimation/sourceDistance")
        col_L_m = _as_float_scalar(col_L) if col_L is not None else 1.0

        # virtual slits (widths in mm)
        vslit = f"{inst_in}/VirtualSlitAxis"

        def vslit_mm(name: str) -> Optional[float]:
            v = _safe_get(fin, f"{vslit}/{name}")
            return None if v is None else float(_as_float_scalar(v))

        def slit_widths_mm(i: int) -> tuple[Optional[float], Optional[float]]:
            xw = vslit_mm(f"s{i}w_actual_width")
            if xw is None:
                xw = vslit_mm(f"s{i}w_wanted_width")
            yw = vslit_mm(f"s{i}h_actual_width")
            if yw is None:
                yw = vslit_mm(f"s{i}h_wanted_width")
            return xw, yw

        collimation_in = f"{inst_in}/collimation"

        # --- Write output ---
        with h5py.File(output_path, "w") as fout:
            entry_out = _ensure_group(fout, "raw_data", "NXentry")
            _write_dataset(entry_out, "definition", "NXsas_raw", as_string=True)
            _write_dataset(entry_out, "schema_version", "1.3", as_string=True)

            # sample
            sample_out = _ensure_group(entry_out, "sample", "NXsample")
            title = _safe_get(fin, f"{entry}/title")
            if title is not None:
                _write_dataset(sample_out, "name", _as_str(title), as_string=True)
            else:
                _write_dataset(sample_out, "name", "unknown", as_string=True)

            # control monitor required by the v1.3 schema
            control_out = _ensure_group(entry_out, "control", "NXmonitor")
            control_mode_src = _safe_get(fin, f"{entry}/monitor/mode")
            if control_mode_src is None:
                control_mode_src = _safe_get(fin, f"{entry}/modestring")
            _write_dataset(control_out, "mode", _sam_monitor_mode(control_mode_src), as_string=True)

            control_preset = _safe_get(fin, f"{entry}/monitor/preset")
            if control_preset is None:
                control_preset = _safe_get(fin, f"{entry}/preset")
            if control_preset is None:
                warnings.append(f"{entry}: missing preset information; writing NaN in /entry/control/preset.")
                _write_dataset(control_out, "preset", float("nan"))
            else:
                _write_dataset(control_out, "preset", _as_float_scalar(control_preset))

            control_integral = _sam_monitor_integral(fin, entry)
            if np.isnan(control_integral):
                warnings.append(
                    f"{entry}/monitor: missing integral, data, and monsum; writing NaN in /entry/control/integral."
                )
            _write_dataset(control_out, "integral", control_integral)

            control_count_time = _safe_get(fin, f"{entry}/time")
            if control_count_time is None:
                control_count_time = _safe_get(fin, f"{entry}/duration")
            acquisition_time = _as_float_scalar(control_count_time) if control_count_time is not None else None
            if acquisition_time is not None:
                _write_dataset(control_out, "count_time", acquisition_time)

            # instrument
            inst_out = _ensure_group(entry_out, "instrument", "NXinstrument")

            geom_out = _ensure_group(inst_out, "geometry", None)
            _write_dataset(geom_out, "origin_definition", "sample center", as_string=True)
            _write_dataset(geom_out, "axis_convention", "+z downstream, +x beam-right, +y up", as_string=True)

            mono_out = _ensure_group(inst_out, "monochromator", "NXmonochromator")
            _write_dataset(mono_out, "wavelength", wavelength)

            # collimation
            coll_out = _ensure_group(inst_out, "collimation", None)
            elements_out = _ensure_group(coll_out, "elements", None)

            collimation_order: list[tuple[str, int]] = [
                ("slit", 1),
                ("guide", 1),
                ("slit", 2),
                ("guide", 2),
                ("slit", 3),
                ("guide", 3),
                ("slit", 4),
            ]
            end_d = 0.1
            start_d = float(end_d + max(col_L_m, 0.0))
            step = (float(col_L_m) / (len(collimation_order) - 1)) if len(collimation_order) > 1 else 0.0
            notes.append("SAM slit4 position is currently written with an ad hoc downstream distance.")

            element_order: List[bytes] = []
            for k, (kind, idx) in enumerate(collimation_order):
                name = f"{kind}{idx}"
                dist_m = start_d - k * step

                if kind == "slit":
                    el = _ensure_group(elements_out, name, "NXslit")
                    xw_mm, yw_mm = slit_widths_mm(idx)
                    if xw_mm is None:
                        warnings.append(f"{vslit}: missing width for {name}/x_gap; writing NaN.")
                        _write_dataset(el, "x_gap", float("nan"))
                    else:
                        _write_dataset(el, "x_gap", float(xw_mm) * MM_TO_M)
                    if yw_mm is None:
                        warnings.append(f"{vslit}: missing width for {name}/y_gap; writing NaN.")
                        _write_dataset(el, "y_gap", float("nan"))
                    else:
                        _write_dataset(el, "y_gap", float(yw_mm) * MM_TO_M)
                else:
                    el = _ensure_group(elements_out, name, "NXguide")
                    state_value = _safe_get(fin, f"{collimation_in}/col{idx}_state")
                    _write_dataset(el, "state", _sam_guide_state(state_value), as_string=True)

                tr = _ensure_group(el, "transformations", "NXtransformations")
                _write_dataset(tr, "translation", np.array([0.0, 0.0, -float(dist_m)], dtype=float))
                element_order.append(name.encode())

            _derive_aperture_snapshots(
                coll_out,
                [name.decode() if isinstance(name, (bytes, bytearray)) else str(name) for name in element_order],
                warnings,
            )
            _write_dataset(coll_out, "collimation_distance", float(max(col_L_m, 0.0)))
            _write_dataset(coll_out, "last_aperture_to_sample_distance", float(end_d))
            _write_dataset(coll_out, "element_order", np.array(element_order, dtype="S"))

            # detector0
            det_out = _ensure_group(inst_out, "detector0", "NXdetector")
            corrected_counts, deadtime_corrected = correct_detector_data_for_deadtime(
                counts,
                acquisition_time=acquisition_time,
                dead_time=detector_dead_time,
                detector_name="detector0",
                warnings=warnings,
            )
            _write_dataset(det_out, "data", corrected_counts)
            _write_dataset(det_out, "x_pixel_size", x_pixel_size_m)
            _write_dataset(det_out, "y_pixel_size", y_pixel_size_m)

            ny, nx = corrected_counts.shape[-2], corrected_counts.shape[-1]
            _write_dataset(det_out, "beam_center_x", (float(nx) - 1.0) / 2.0)
            _write_dataset(det_out, "beam_center_y", (float(ny) - 1.0) / 2.0)
            _write_dataset(det_out, "dead_time", detector_dead_time)
            _write_dataset(det_out, "deadtime_corrected", deadtime_corrected)

            det_tr = _ensure_group(det_out, "transformations", "NXtransformations")
            _write_dataset(det_tr, "translation", np.array([0.0, 0.0, det_z_m], dtype=float))

            # NXdata view
            data_out = _ensure_group(entry_out, "data0", "NXdata")
            data_out.attrs["signal"] = np.bytes_("counts")
            data_out["counts"] = h5py.SoftLink("/raw_data/instrument/detector0/data")

    return ConvertReport(
        input_file=input_path,
        output_file=output_path,
        entry_in=entry,
        notes=notes,
        warnings=warnings,
    )
