from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from ._hdf import (
    as_float_scalar as _as_float_scalar,
    as_str as _as_str,
    ensure_group as _ensure_group,
    pick_entry as _pick_entry,
    safe_get as _safe_get,
    safe_get_dataset as _safe_get_dataset,
    write_dataset as _write_dataset,
)
from ._report import ConvertReport
from ._units import MM_TO_M, length_dataset_to_m as _length_dataset_to_m


D11_PIXEL_SIZE_M = 5.0 * MM_TO_M
# TODO: Verify the D11 detector dead-time value from instrument metadata or beamline documentation.
D11_DEAD_TIME_S = 0.0


def _d11_instrument_path(fin: h5py.File, entry: str) -> str:
    preferred = f"{entry}/D11"
    if preferred in fin and isinstance(fin[preferred], h5py.Group):
        return preferred

    eg = fin[entry]
    for key, obj in eg.items():
        if isinstance(obj, h5py.Group) and _as_str(obj.attrs.get("NX_class")) == "NXinstrument":
            return f"{entry}/{key}"
    raise ValueError("No D11 NXinstrument group found under input entry.")


def _read_detector_counts(fin: h5py.File, entry: str, inst_in: str) -> np.ndarray:
    for path in (
        f"{inst_in}/detector/data",
        f"{entry}/data/detector_data",
    ):
        ds = _safe_get_dataset(fin, path)
        if ds is None:
            continue
        data = np.asarray(ds[()])
        if data.ndim == 3 and data.shape[-1] == 1:
            return data[..., 0]
        if data.ndim == 2:
            return data
        raise ValueError(f"Unsupported D11 detector data shape at {path}: {data.shape}")
    raise ValueError("Could not locate D11 detector data.")


def _monitor_mode(value) -> str:
    mode = _as_str(value).strip().lower() if value is not None else ""
    if mode == "time":
        return "timer"
    if mode in {"monitor", "timer"}:
        return mode
    return "monitor"


def _monitor_scalar(fin: h5py.File, path: str) -> float:
    value = _safe_get(fin, path)
    if value is None:
        return float("nan")
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return float(arr.reshape(()))
    finite = arr[np.isfinite(arr)]
    return float(np.sum(finite)) if finite.size else float("nan")


def _sample_name(fin: h5py.File, entry: str) -> str:
    sample_description = _safe_get(fin, f"{entry}/sample_description")
    if sample_description is not None:
        name = _as_str(sample_description).strip()
        if name:
            return name
    return "unknown"


def _wavelength_error(wavelength: float, resolution) -> float:
    if not np.isfinite(wavelength) or resolution is None:
        return float("nan")
    res = _as_float_scalar(resolution)
    if not np.isfinite(res):
        return float("nan")
    if res < 1.0:
        return wavelength * res
    if res <= 100.0:
        return wavelength * (res / 100.0)
    return res


def _beam_center(fin: h5py.File, inst_in: str) -> tuple[Optional[float], Optional[float]]:
    cx = _safe_get(fin, f"{inst_in}/Beam/center_x")
    cy = _safe_get(fin, f"{inst_in}/Beam/center_y")
    return (
        None if cx is None else _as_float_scalar(cx),
        None if cy is None else _as_float_scalar(cy),
    )


def _write_monitor_group(fin: h5py.File, src_path: str, parent: h5py.Group, name: str) -> h5py.Group:
    mon_out = _ensure_group(parent, name, "NXmonitor")
    mode = _safe_get(fin, f"{src_path}/mode")
    _write_dataset(mon_out, "mode", _monitor_mode(mode), as_string=True)
    _write_dataset(mon_out, "preset", _monitor_scalar(fin, f"{src_path}/preset"))

    integral = _monitor_scalar(fin, f"{src_path}/integral")
    if np.isfinite(integral):
        _write_dataset(mon_out, "integral", integral)
    elif f"{src_path}/data" in fin:
        _write_dataset(mon_out, "data", fin[f"{src_path}/data"][()])
    else:
        _write_dataset(mon_out, "integral", float("nan"))
    return mon_out


def _collimation_distance_m(fin: h5py.File, inst_in: str) -> float:
    ds = _safe_get_dataset(fin, f"{inst_in}/collimation/actual_position")
    if ds is None:
        return float("nan")
    units = ds.attrs.get("units")
    units_s = _as_str(units).strip().lower() if units is not None else ""
    if units_s in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        return _as_float_scalar(ds[()]) * MM_TO_M
    if units_s in {"", "m", "meter", "meters", "metre", "metres"}:
        return _as_float_scalar(ds[()])
    value_m = _length_dataset_to_m(ds)
    return float(value_m) if value_m is not None else _as_float_scalar(ds[()])


def _mm_field_to_m(fin: h5py.File, path: str) -> Optional[float]:
    value = _safe_get(fin, path)
    if value is None:
        return None
    return _as_float_scalar(value) * MM_TO_M


def _write_collimation(fin: h5py.File, inst_in: str, inst_out: h5py.Group, warnings: List[str]) -> None:
    coll_out = _ensure_group(inst_out, "collimation", None)
    elements_out = _ensure_group(coll_out, "elements", None)

    collimation_distance_m = _collimation_distance_m(fin, inst_in)
    if not np.isfinite(collimation_distance_m):
        warnings.append(f"{inst_in}/collimation/actual_position missing; writing NaN collimation_distance.")

    # TODO: Check the D11 collimation slit mapping; diaphragm*_position fields may need
    # an instrument lookup table instead of using the guide-exit cross section directly.
    guide_exit = _ensure_group(elements_out, "guide_exit", "NXslit")
    guide_width_m = _mm_field_to_m(fin, f"{inst_in}/collimation/guide_exit_cross_section_width")
    guide_height_m = _mm_field_to_m(fin, f"{inst_in}/collimation/guide_exit_cross_section_height")
    _write_dataset(guide_exit, "x_gap", guide_width_m if guide_width_m is not None else float("nan"), units="m")
    _write_dataset(guide_exit, "y_gap", guide_height_m if guide_height_m is not None else float("nan"), units="m")
    guide_tr = _ensure_group(guide_exit, "transformations", "NXtransformations")
    z = -collimation_distance_m if np.isfinite(collimation_distance_m) else float("nan")
    _write_dataset(guide_tr, "translation", np.array([0.0, 0.0, z], dtype=float), units="m")

    sample_ap_x_m = _mm_field_to_m(fin, f"{inst_in}/Beam/sample_ap_x_or_diam")
    sample_ap_y_m = _mm_field_to_m(fin, f"{inst_in}/Beam/sample_ap_y")
    aperture_class = "NXpinhole" if sample_ap_y_m is not None and sample_ap_y_m == 0.0 else "NXslit"
    sample_ap = _ensure_group(elements_out, "sample_aperture", aperture_class)
    if aperture_class == "NXpinhole":
        _write_dataset(sample_ap, "diameter", sample_ap_x_m if sample_ap_x_m is not None else float("nan"), units="m")
    else:
        _write_dataset(sample_ap, "x_gap", sample_ap_x_m if sample_ap_x_m is not None else float("nan"), units="m")
        _write_dataset(sample_ap, "y_gap", sample_ap_y_m if sample_ap_y_m is not None else float("nan"), units="m")
    sample_ap_tr = _ensure_group(sample_ap, "transformations", "NXtransformations")
    _write_dataset(sample_ap_tr, "translation", np.array([0.0, 0.0, 0.0], dtype=float), units="m")

    aperture1 = _ensure_group(coll_out, "aperture1", "NXslit")
    _write_dataset(aperture1, "x_gap", guide_width_m if guide_width_m is not None else float("nan"), units="m")
    _write_dataset(aperture1, "y_gap", guide_height_m if guide_height_m is not None else float("nan"), units="m")
    aperture2 = _ensure_group(coll_out, "aperture2", aperture_class)
    if aperture_class == "NXpinhole":
        _write_dataset(aperture2, "diameter", sample_ap_x_m if sample_ap_x_m is not None else float("nan"), units="m")
    else:
        _write_dataset(aperture2, "x_gap", sample_ap_x_m if sample_ap_x_m is not None else float("nan"), units="m")
        _write_dataset(aperture2, "y_gap", sample_ap_y_m if sample_ap_y_m is not None else float("nan"), units="m")

    _write_dataset(coll_out, "collimation_distance", collimation_distance_m, units="m")
    _write_dataset(coll_out, "last_aperture_to_sample_distance", 0.0, units="m")
    _write_dataset(coll_out, "element_order", np.array([b"guide_exit", b"sample_aperture"], dtype="S"))


def convert_d11_to_scarlet_nxsas_raw(
    input_path: str | Path,
    output_path: str | Path,
    *,
    entry_in: Optional[str] = None,
    overwrite: bool = False,
) -> ConvertReport:
    """
    Convert an ILL D11 NeXus file to SCARLET NXsas_raw (monochromatic profile).
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
        inst_in = _d11_instrument_path(fin, entry)
        notes.append(f"Using input entry: {entry}")
        notes.append("Using D11 monitor1 as the SCARLET control monitor.")

        monitor1_path = f"{entry}/monitor1"
        if monitor1_path not in fin:
            warnings.append(f"{monitor1_path}: missing; writing NaN control values.")

        counts = _read_detector_counts(fin, entry, inst_in)
        acquisition_time_src = _safe_get(fin, f"{entry}/duration")
        acquisition_time = _as_float_scalar(acquisition_time_src) if acquisition_time_src is not None else None

        wavelength_src = _safe_get(fin, f"{inst_in}/selector/wavelength")
        wavelength = _as_float_scalar(wavelength_src) if wavelength_src is not None else float("nan")
        if not np.isfinite(wavelength):
            warnings.append(f"{inst_in}/selector/wavelength missing; writing NaN wavelength.")
        wavelength_error = _wavelength_error(wavelength, _safe_get(fin, f"{inst_in}/selector/wavelength_res"))

        det_z_m = _length_dataset_to_m(_safe_get_dataset(fin, f"{inst_in}/detector/det_actual"))
        if det_z_m is None:
            det_z_m = float("nan")
            warnings.append(f"{inst_in}/detector/det_actual missing; writing NaN detector translation.")

        beam_center_x, beam_center_y = _beam_center(fin, inst_in)
        ny, nx = counts.shape[-2], counts.shape[-1]
        if beam_center_x is None:
            beam_center_x = (float(nx) - 1.0) / 2.0
            warnings.append(f"{inst_in}/Beam/center_x missing; using detector midpoint.")
        if beam_center_y is None:
            beam_center_y = (float(ny) - 1.0) / 2.0
            warnings.append(f"{inst_in}/Beam/center_y missing; using detector midpoint.")

        with h5py.File(output_path, "w") as fout:
            entry_out = _ensure_group(fout, "raw_data", "NXentry")
            _write_dataset(entry_out, "definition", "NXsas_raw", as_string=True)
            _write_dataset(entry_out, "schema_version", "1.3", as_string=True)

            sample_out = _ensure_group(entry_out, "sample", "NXsample")
            _write_dataset(sample_out, "name", _sample_name(fin, entry), as_string=True)

            if monitor1_path in fin:
                control_out = _write_monitor_group(fin, monitor1_path, entry_out, "control")
            else:
                control_out = _ensure_group(entry_out, "control", "NXmonitor")
                _write_dataset(control_out, "mode", "monitor", as_string=True)
                _write_dataset(control_out, "preset", float("nan"))
                _write_dataset(control_out, "integral", float("nan"))
            if acquisition_time is not None:
                _write_dataset(control_out, "count_time", acquisition_time, units="s")

            inst_out = _ensure_group(entry_out, "instrument", "NXinstrument")
            geom_out = _ensure_group(inst_out, "geometry", None)
            _write_dataset(geom_out, "origin_definition", "sample center", as_string=True)
            _write_dataset(geom_out, "axis_convention", "+z downstream, +x beam-right, +y up", as_string=True)

            mono_out = _ensure_group(inst_out, "monochromator", "NXmonochromator")
            _write_dataset(mono_out, "wavelength", wavelength, units="angstrom")
            if np.isfinite(wavelength_error):
                _write_dataset(mono_out, "wavelength_error", wavelength_error, units="angstrom")
            selector_out = _ensure_group(mono_out, "velocity_selector", "NXvelocity_selector")
            rotation_speed = _safe_get(fin, f"{inst_in}/selector/rotation_speed")
            if rotation_speed is not None:
                _write_dataset(selector_out, "rotation_speed", _as_float_scalar(rotation_speed), units="rpm")

            _write_collimation(fin, inst_in, inst_out, warnings)

            if monitor1_path in fin:
                _write_monitor_group(fin, monitor1_path, inst_out, "monitor1")

            beamstop_in = f"{inst_in}/beamstop"
            if beamstop_in in fin:
                beamstop_out = _ensure_group(inst_out, "beamstop", "NXbeamstop")
                for field in ("actual_beamstop_number", "bx_actual", "by_actual", "height", "width_or_diam"):
                    if f"{beamstop_in}/{field}" in fin:
                        ds = fin[f"{beamstop_in}/{field}"]
                        units = _as_str(ds.attrs.get("units")) if ds.attrs.get("units") is not None else None
                        _write_dataset(beamstop_out, field, ds[()], units=units)

            det_out = _ensure_group(inst_out, "detector0", "NXdetector")
            corrected_counts = np.asarray(counts, dtype=np.float64)
            deadtime_corrected = False
            _write_dataset(det_out, "data", corrected_counts)
            _write_dataset(det_out, "x_pixel_size", D11_PIXEL_SIZE_M, units="m")
            _write_dataset(det_out, "y_pixel_size", D11_PIXEL_SIZE_M, units="m")
            _write_dataset(det_out, "beam_center_x", beam_center_x)
            _write_dataset(det_out, "beam_center_y", beam_center_y)
            _write_dataset(det_out, "dead_time", D11_DEAD_TIME_S, units="s")
            _write_dataset(det_out, "deadtime_corrected", deadtime_corrected)
            _write_dataset(det_out, "local_name", "D11 detector", as_string=True)
            _write_dataset(det_out, "distance", det_z_m, units="m")

            det_tr = _ensure_group(det_out, "transformations", "NXtransformations")
            _write_dataset(det_tr, "translation", np.array([0.0, 0.0, float(det_z_m)], dtype=float), units="m")

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
