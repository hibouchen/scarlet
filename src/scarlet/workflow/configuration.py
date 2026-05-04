from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union, List, Literal, Mapping
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import h5py
import numpy as np
import math

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


def write_refs_sub_file(
    file_path: Union[str, Path],
    configuration: "Configuration",
    *,
    empty_beam_transmission: Union[str, Path],
    dark: Optional[Union[str, Path]] = None,
    empty_beam_scattering: Optional[Union[str, Path]] = None,
    empty_cell_transmission: Optional[Union[str, Path]] = None,
    empty_cell_scattering: Optional[Union[str, Path]] = None,
    transmission_roi_detector: Union[int, str] = 0,
    transmission_roi: tuple[int, int, int, int],
    transmission_roi_notes: Optional[str] = None,
    masks: Optional[Mapping[int, np.ndarray]] = None,
    attenuation_factor: Optional[float] = None,
    created_utc: Optional[str] = None,
    mask_convention: str = "1=masked, 0=valid",
    scarlet_version: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """
    Write a SCARLET_refs_sub file matching the packaged YAML schema.

    Reference source files are passed as keyword arguments and copied into
    ``/entry/references/<name>/entry``.
    """
    file_path = Path(file_path)
    x0, x1, y0, y1 = transmission_roi

    def _write_dataset(parent: h5py.Group, name: str, value) -> None:
        if isinstance(value, str):
            parent.create_dataset(name, data=np.bytes_(value))
        else:
            parent.create_dataset(name, data=value)

    def _require_number(name: str, value: Optional[float]) -> float:
        if value is None:
            raise ValueError(f"{name} is required")
        out = float(value)
        if math.isnan(out):
            raise ValueError(f"{name} must not be NaN")
        return out

    def _scalar_distance(value: Union[float, List[float]]) -> float:
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError("sample_detector_distance must be a scalar for refs_sub output")
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
        for path in ("/entry", "/entry0", "/entry1"):
            if path in source and isinstance(source[path], h5py.Group):
                return source[path]
        raise ValueError("Reference file must contain /entry, /entry0, or /entry1")

    def _copy_reference(parent: h5py.Group, name: str, source_path: Union[str, Path]) -> Path:
        source_path = Path(source_path).resolve()
        ref_group = parent.create_group(name)
        with h5py.File(source_path, "r") as source_file:
            entry_group = _select_entry(source_file)
            source_file.copy(entry_group, ref_group, name="entry")
        return source_path

    def _file_created_utc(path: Path) -> str:
        stat = path.stat()
        timestamp = getattr(stat, "st_birthtime", stat.st_ctime)
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")

    if file_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output file exists: {file_path}")
        file_path.unlink()

    if configuration.config_id is None:
        raise ValueError("configuration.config_id is required for refs_sub output")
    if configuration.collimation is None:
        raise ValueError("configuration.collimation is required for refs_sub output")

    references = {
        "dark": dark,
        "empty_beam_transmission": empty_beam_transmission,
        "empty_beam_scattering": empty_beam_scattering,
        "empty_cell_transmission": empty_cell_transmission,
        "empty_cell_scattering": empty_cell_scattering,
    }
    masks = masks or {}

    with h5py.File(file_path, "w") as f:
        if created_utc is None:
            created_utc = _file_created_utc(file_path)

        entry = f.create_group("entry")
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        _write_dataset(entry, "definition", "SCARLET_refs_sub")
        _write_dataset(entry, "schema_version", "1.0")
        _write_dataset(entry, "config_id", configuration.config_id)

        cfg = entry.create_group("configuration")
        cfg.attrs["NX_class"] = np.bytes_("NXcollection")
        _write_dataset(cfg, "wavelength", _require_number("wavelength", configuration.wavelength))
        _write_dataset(cfg, "sample_detector_distance", _scalar_distance(configuration.sample_detector_distance))
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

        refs = entry.create_group("references")
        refs.attrs["NX_class"] = np.bytes_("NXcollection")
        source_paths: dict[str, Path] = {}
        for name, source_path in references.items():
            if source_path is None:
                continue
            source_paths[name] = _copy_reference(refs, name, source_path)

        if masks:
            mask_group = entry.create_group("mask")
            mask_group.attrs["NX_class"] = np.bytes_("NXcollection")
            for detector, mask in sorted(masks.items()):
                _write_dataset(mask_group, f"mask_detector{int(detector)}", np.asarray(mask))

        transmission_roi = entry.create_group("transmission_roi")
        transmission_roi.attrs["NX_class"] = np.bytes_("NXcollection")
        _write_dataset(transmission_roi, "detector", transmission_roi_detector)
        _write_dataset(transmission_roi, "roi_type", "rectangle")
        _write_dataset(transmission_roi, "x0", int(x0))
        _write_dataset(transmission_roi, "x1", int(x1))
        _write_dataset(transmission_roi, "y0", int(y0))
        _write_dataset(transmission_roi, "y1", int(y1))
        if transmission_roi_notes is not None:
            _write_dataset(transmission_roi, "notes", transmission_roi_notes)

        if attenuation_factor is not None:
            transmission_setup = entry.create_group("transmission_setup")
            transmission_setup.attrs["NX_class"] = np.bytes_("NXcollection")
            attenuator = transmission_setup.create_group("attenuator")
            attenuator.attrs["NX_class"] = np.bytes_("NXattenuator")
            _write_dataset(
                attenuator,
                "attenuation_factor",
                _require_number("attenuation_factor", attenuation_factor),
            )

        meta = entry.create_group("meta")
        meta.attrs["NX_class"] = np.bytes_("NXcollection")
        _write_dataset(meta, "created_utc", created_utc)
        _write_dataset(meta, "mask_convention", mask_convention)
        for name, source_path in source_paths.items():
            _write_dataset(meta, f"{name}_source_file", str(source_path))
        if scarlet_version is not None:
            _write_dataset(meta, "scarlet_version", scarlet_version)

    return file_path


def configuration_from_nexus(
    file_path: Union[str, Path],
    *,
    entry_path: str = "/entry",
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
            # common fallback if the user passes a raw file with entry0
            for cand in ("/entry0", "/entry1"):
                if cand in f:
                    entry_path = cand
                    issues.append(f"entry_path not found, using {cand}")
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


def write_run_configuration_excel(
    file_path: Union[str, Path],
    workflow_context,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Generate a minimal Excel workbook listing each data file, its configuration,
    and the corresponding sample name.
    """
    file_path = Path(file_path)
    if file_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output file exists: {file_path}")
        file_path.unlink()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    def _distance_text(value: Union[float, List[float]]) -> str:
        if isinstance(value, list):
            return ", ".join(f"{float(v):g}" for v in value)
        return f"{float(value):g}"

    def _distance_sort_value(value: Union[float, List[float]]) -> float:
        if isinstance(value, list):
            return min(float(v) for v in value) if value else float("inf")
        return float(value)

    def _aperture_text(aperture: Aperture) -> str:
        if aperture.type == "slit":
            x_gap = "" if aperture.x_gap is None else f"{float(aperture.x_gap):g}"
            y_gap = "" if aperture.y_gap is None else f"{float(aperture.y_gap):g}"
            return f"slit({x_gap} x {y_gap} m)"
        diameter = "" if aperture.diameter is None else f"{float(aperture.diameter):g}"
        return f"pinhole({diameter} m)"

    def _configuration_text(configuration: Optional["Configuration"]) -> str:
        if configuration is None:
            return ""
        parts = [
            f"wavelength={float(configuration.wavelength):g} A",
            f"sample_detector_distance={_distance_text(configuration.sample_detector_distance)} m",
        ]
        if configuration.collimation is not None:
            parts.extend(
                (
                    f"collimation_distance={float(configuration.collimation.collimation_distance):g} m",
                    f"last_aperture_to_sample_distance={float(configuration.collimation.last_aperture_to_sample_distance):g} m",
                    f"aperture1={_aperture_text(configuration.collimation.aperture1)}",
                    f"aperture2={_aperture_text(configuration.collimation.aperture2)}",
                )
            )
        if configuration.notes:
            parts.append(f"notes={configuration.notes}")
        return "; ".join(parts)

    config_aliases: dict[str, str] = {}
    canonical_configurations: dict[str, Configuration] = {}
    for config_id in sorted(workflow_context.configurations):
        configuration = workflow_context.configurations[config_id]
        canonical_id = config_id
        for existing_id, existing_configuration in canonical_configurations.items():
            same, _ = compare_configurations(configuration, existing_configuration)
            if same:
                canonical_id = existing_id
                break
        config_aliases[config_id] = canonical_id
        if canonical_id == config_id:
            canonical_configurations[canonical_id] = configuration

    canonical_labels = {
        canonical_id: f"config_{index}"
        for index, canonical_id in enumerate(
            sorted(
                canonical_configurations,
                key=lambda cid: (
                    _distance_sort_value(canonical_configurations[cid].sample_detector_distance),
                    cid,
                ),
            ),
            start=1,
        )
    }

    rows = [
        ("data_file", "config_id", "configuration", "sample_name", "measurement_type"),
    ]
    for run_key, run_path in sorted(
        workflow_context.runs.items(),
        key=lambda item: (
            config_aliases.get(item[0].config_id, item[0].config_id),
            "" if item[0].sample_id is None else item[0].sample_id,
            item[0].entity,
            item[0].mode,
            str(item[1]),
        ),
    ):
        canonical_config_id = config_aliases.get(run_key.config_id, run_key.config_id)
        config_id = canonical_labels.get(canonical_config_id, canonical_config_id)
        configuration = canonical_configurations.get(
            canonical_config_id,
            workflow_context.configurations.get(run_key.config_id),
        )
        rows.append(
            (
                Path(run_path).name,
                config_id,
                _configuration_text(configuration),
                "" if run_key.sample_id is None else run_key.sample_id,
                run_key.mode,
            )
        )

    def _column_name(index: int) -> str:
        out = []
        while index:
            index, rem = divmod(index - 1, 26)
            out.append(chr(65 + rem))
        return "".join(reversed(out))

    def _cell(ref: str, value: str) -> str:
        return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(value)}</t></is></c>'

    def _sheet_xml() -> str:
        row_chunks = []
        for row_index, row in enumerate(rows, start=1):
            cell_chunks = []
            for column_index, value in enumerate(row, start=1):
                cell_chunks.append(_cell(f"{_column_name(column_index)}{row_index}", value))
            row_chunks.append(f'<row r="{row_index}">{"".join(cell_chunks)}</row>')
        last_cell = f"{_column_name(len(rows[0]))}{len(rows)}"
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="A1:{last_cell}"/>'
            '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
            '<sheetFormatPr defaultRowHeight="15"/>'
            f'<sheetData>{"".join(row_chunks)}</sheetData>'
            '</worksheet>'
        )

    with ZipFile(file_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '</Types>',
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            '</Relationships>',
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="runs" sheetId="1" r:id="rId1"/></sheets>'
            '</workbook>',
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
            '</Relationships>',
        )
        zf.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            '</styleSheet>',
        )
        zf.writestr("xl/worksheets/sheet1.xml", _sheet_xml())

    if hasattr(workflow_context, "add_artifact"):
        workflow_context.add_artifact(file_path.name, file_path, kind="xlsx")

    return file_path
