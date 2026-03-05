from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
    if isinstance(x, np.ndarray) and x.size == 1:
        return _as_str(x.reshape(()).item())
    return str(x)


def _as_float_scalar(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, np.ndarray) and x.size == 1:
        return float(x.reshape(()))
    return float(x)


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


def _mm_to_m(value) -> float:
    return _as_float_scalar(value) * MM_TO_M


def _pick_entry(fin: h5py.File, preferred: Optional[str] = None) -> str:
    if preferred and preferred in fin:
        return preferred
    for cand in ("/entry0", "/entry", "/entry1"):
        if cand in fin:
            return cand
    # fallback
    for k in fin.keys():
        p = f"/{k}"
        if isinstance(fin[p], h5py.Group):
            nx = fin[p].attrs.get("NX_class", None)
            if nx is not None and _as_str(nx) == "NXentry":
                return p
    raise ValueError("No NXentry found in file.")


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

        # collimation length (assumed meters; often equals 2.5/6/...)
        col_L = _safe_get(fin, f"{inst_in}/collimation/sourceDistance")
        if col_L is None:
            col_L = _safe_get(fin, f"{inst_in}/collimation/position")
        col_L_m = _as_float_scalar(col_L) if col_L is not None else 1.0

        # virtual slits (widths in mm)
        vslit = f"{inst_in}/VirtualSlitAxis"

        def vslit_mm(name: str) -> Optional[float]:
            v = _safe_get(fin, f"{vslit}/{name}")
            return None if v is None else float(_as_float_scalar(v))

        # S1/S2/S3: use actual widths if available, else wanted widths.
        slit_specs = []
        for i in (1, 2, 3):
            xw = vslit_mm(f"s{i}w_actual_width") or vslit_mm(f"s{i}w_wanted_width")
            yw = vslit_mm(f"s{i}h_actual_width") or vslit_mm(f"s{i}h_wanted_width")
            if xw is None and yw is None:
                continue
            slit_specs.append((f"slit{i}", xw, yw))

        # --- Write output ---
        with h5py.File(output_path, "w") as fout:
            entry_out = _ensure_group(fout, "entry", "NXentry")
            _write_dataset(entry_out, "definition", "NXsas_raw", as_string=True)
            _write_dataset(entry_out, "schema_version", "1.0", as_string=True)

            # sample
            sample_out = _ensure_group(entry_out, "sample", "NXsample")
            title = _safe_get(fin, f"{entry}/title")
            if title is not None:
                _write_dataset(sample_out, "name", _as_str(title), as_string=True)
            else:
                _write_dataset(sample_out, "name", "unknown", as_string=True)

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

            element_order: List[bytes] = []
            if slit_specs:
                end_d = 0.1
                start_d = float(end_d + max(col_L_m, 0.0))
                step = (float(col_L_m) / (len(slit_specs) - 1)) if len(slit_specs) > 1 else 0.0
                for k, (name, xw_mm, yw_mm) in enumerate(slit_specs):
                    el = _ensure_group(elements_out, name, "NXslit")
                    if xw_mm is not None:
                        _write_dataset(el, "x_gap", float(xw_mm) * MM_TO_M)
                    if yw_mm is not None:
                        _write_dataset(el, "y_gap", float(yw_mm) * MM_TO_M)
                    tr = _ensure_group(el, "transformations", "NXtransformations")
                    dist_m = start_d - k * step
                    _write_dataset(tr, "translation", np.array([0.0, 0.0, -float(dist_m)], dtype=float))
                    element_order.append(name.encode())
            else:
                # Minimal placeholder to satisfy schema
                el = _ensure_group(elements_out, "slit1", "NXslit")
                tr = _ensure_group(el, "transformations", "NXtransformations")
                _write_dataset(tr, "translation", np.array([0.0, 0.0, -0.1], dtype=float))
                element_order.append(b"slit1")
                warnings.append("VirtualSlitAxis widths not found; writing placeholder collimation slit1.")

            _write_dataset(coll_out, "element_order", np.array(element_order, dtype="S"))

            # detector0
            det_out = _ensure_group(inst_out, "detector0", "NXdetector")
            _write_dataset(det_out, "data", counts)
            _write_dataset(det_out, "x_pixel_size", x_pixel_size_m)
            _write_dataset(det_out, "y_pixel_size", y_pixel_size_m)

            ny, nx = counts.shape[-2], counts.shape[-1]
            _write_dataset(det_out, "beam_center_x", (float(nx) - 1.0) / 2.0)
            _write_dataset(det_out, "beam_center_y", (float(ny) - 1.0) / 2.0)
            _write_dataset(det_out, "dead_time", float("nan"))

            det_tr = _ensure_group(det_out, "transformations", "NXtransformations")
            _write_dataset(det_tr, "translation", np.array([0.0, 0.0, det_z_m], dtype=float))

            # NXdata view
            data_out = _ensure_group(entry_out, "data0", "NXdata")
            data_out.attrs["signal"] = np.bytes_("counts")
            data_out["counts"] = h5py.SoftLink("/entry/instrument/detector0/data")

    return ConvertReport(
        input_file=input_path,
        output_file=output_path,
        entry_in=entry,
        notes=notes,
        warnings=warnings,
    )

