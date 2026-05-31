from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal, cast
from datetime import datetime, timezone
from collections import OrderedDict
import csv
from html import escape
import json
import re

import numpy as np
import h5py


# -----------------------------
# Small typed helpers
# -----------------------------

Level = Literal["INFO", "WARN", "ERROR"]
Mode = Literal["scattering", "transmission"]

# “entity” describes which physical run it is.
# sample_name may also be used on non-sample entities to preserve manual labels.
Entity = Literal[
    "sample",
    "empty_beam",
    "empty_cell",
    "dark",
    "refs_sub",   # file refs_<config>.nxs
]


@dataclass(frozen=True)
class RunKey:
    """Key used to identify a run in the workflow."""
    config_id: str
    entity: Entity
    mode: Mode
    sample_name: Optional[str] = None  # required for entity=="sample", optional otherwise

    def short(self) -> str:
        s = f"{self.config_id}:{self.entity}:{self.mode}"
        if self.sample_name:
            s += f":{self.sample_name}"
        return s


@dataclass(frozen=True)
class Artifact:
    """A file produced by the workflow (for reporting / reproducibility)."""
    name: str
    path: Path
    kind: str  # e.g. "nexus", "text", "plot", "csv"
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class LogMessage:
    level: Level
    message: str
    where: Optional[str] = None      # step name or component
    when_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Issue:
    level: Literal["WARN", "ERROR"]
    message: str
    where: Optional[str] = None
    key: Optional[str] = None        # optional: a RunKey.short() or dataset path
    when_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TableView:
    """Notebook-friendly tabular view with HTML rendering in Jupyter."""
    columns: Tuple[str, ...]
    rows: List[Dict[str, str]]

    def _repr_html_(self) -> str:
        header = "".join(f"<th>{escape(column)}</th>" for column in self.columns)
        body_rows = []
        for row in self.rows:
            cells = "".join(f"<td>{escape(row.get(column, ''))}</td>" for column in self.columns)
            body_rows.append(f"<tr>{cells}</tr>")
        body = "".join(body_rows)
        return (
            "<table>"
            f"<thead><tr>{header}</tr></thead>"
            f"<tbody>{body}</tbody>"
            "</table>"
        )

    def __repr__(self) -> str:
        lines = [",".join(self.columns)]
        for row in self.rows:
            lines.append(",".join(row.get(column, "") for column in self.columns))
        return "\n".join(lines)


# -----------------------------
# WorkflowContext
# -----------------------------

@dataclass
class WorkflowContext:
    """
    Central state container for SCARLET workflows.

    Principles:
    - Workflow orchestration reads/writes to ctx.
    - Raw physics operations should live in scarlet/reduction/ and be called by steps.
    - ctx must support multi-sample + multi-config + per-config refs_sub bundles.
    """

    # --- identification / paths
    experiment_id: str = "experiment"
    instrument_name: str = "unknown"
    root_dir: Path = field(default_factory=lambda: Path(".").resolve())
    output_dir: Path = field(default_factory=lambda: Path("./out").resolve())

    # --- schemas to validate files (optional, but handy)
    schema_raw: str = "scarlet_nxsas_raw_v1.3_mono.yaml"
    schema_refs_sub: str = "scarlet_refs_sub_v1.0.yaml"
    schema_refs_norm: str = "scarlet_refs_norm_v1.0.yaml"
    schema_masks: str = "scarlet_masks_v1.0.yaml"

    # --- run registry (filled by your experiment YAML loader or by code)
    runs: Dict[RunKey, Path] = field(default_factory=dict)

    # --- derived configurations (filled by reading runs)
    # key: config_id -> configuration object (from workflow/configuration.py)
    configurations: Dict[str, Any] = field(default_factory=dict)

    # --- refs_sub bundles per config
    refs_sub_files: Dict[str, Path] = field(default_factory=dict)  # config_id -> refs_sub .nxs
    refs_norm_files: Dict[str, Path] = field(default_factory=dict)  # config_id -> refs_norm .nxs
    masks_files: Dict[str, Path] = field(default_factory=dict)  # config_id -> masks .nxs
    # --- cached objects
    # store anything; structured cache accessors below are backed by this map
    store: Dict[str, Any] = field(default_factory=dict)

    # --- logging / issues / artefacts
    logs: List[LogMessage] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)

    # --- timings
    timings: Dict[str, float] = field(default_factory=dict)  # step_name -> seconds

    # --- internal HDF5 file cache (avoid reopening many times)
    _h5_cache: "OrderedDict[Path, h5py.File]" = field(default_factory=OrderedDict, init=False, repr=False)
    _h5_cache_size: int = field(default=8, init=False, repr=False)

    def _store_dict(self, key: str) -> Dict[Any, Any]:
        existing = self.store.get(key)
        if existing is None:
            existing = {}
            self.store[key] = existing
        if not isinstance(existing, dict):
            raise TypeError(f"Context store entry '{key}' must be a dict, got {type(existing).__name__}")
        return existing

    @property
    def frames(self) -> Dict[Tuple[RunKey, int], np.ndarray]:
        return cast(Dict[Tuple[RunKey, int], np.ndarray], self._store_dict("frames"))

    @property
    def frame_errors(self) -> Dict[Tuple[RunKey, int], np.ndarray]:
        return cast(Dict[Tuple[RunKey, int], np.ndarray], self._store_dict("frame_errors"))

    @property
    def masks(self) -> Dict[Tuple[str, int], np.ndarray]:
        return cast(Dict[Tuple[str, int], np.ndarray], self._store_dict("masks"))

    @property
    def transmissions(self) -> Dict[Tuple[str, str], float]:
        return cast(Dict[Tuple[str, str], float], self._store_dict("transmissions"))

    # -----------------------------
    # Logging / issues
    # -----------------------------

    def log(self, level: Level, message: str, *, where: Optional[str] = None, **meta: Any) -> None:
        self.logs.append(LogMessage(level=level, message=message, where=where, meta=dict(meta)))

    def info(self, message: str, *, where: Optional[str] = None, **meta: Any) -> None:
        self.log("INFO", message, where=where, **meta)

    def warn(self, message: str, *, where: Optional[str] = None, key: Optional[str] = None, **meta: Any) -> None:
        self.issues.append(Issue(level="WARN", message=message, where=where, key=key, meta=dict(meta)))
        self.log("WARN", message, where=where, key=key, **meta)

    def error(self, message: str, *, where: Optional[str] = None, key: Optional[str] = None, **meta: Any) -> None:
        self.issues.append(Issue(level="ERROR", message=message, where=where, key=key, meta=dict(meta)))
        self.log("ERROR", message, where=where, key=key, **meta)

    def has_errors(self) -> bool:
        return any(i.level == "ERROR" for i in self.issues)

    # -----------------------------
    # Store helpers
    # -----------------------------

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.store.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.store:
            raise KeyError(f"Missing required context key: {key}")
        return self.store[key]

    # -----------------------------
    # Artefacts
    # -----------------------------

    def add_artifact(self, name: str, path: Path, *, kind: str = "file") -> None:
        self.artifacts.append(Artifact(name=name, path=Path(path).resolve(), kind=kind))

    # -----------------------------
    # HDF5 caching
    # -----------------------------

    def open_h5(self, path: Path) -> h5py.File:
        """
        Open an HDF5/NeXus file with a small LRU cache.
        Important: call ctx.close_all_h5() at the end of a pipeline run.
        """
        path = Path(path).resolve()

        if path in self._h5_cache:
            f = self._h5_cache.pop(path)
            self._h5_cache[path] = f
            return f

        # evict oldest if needed
        while len(self._h5_cache) >= self._h5_cache_size:
            old_path, old_file = self._h5_cache.popitem(last=False)
            try:
                old_file.close()
            except Exception:
                pass

        f = h5py.File(path, "r")
        self._h5_cache[path] = f
        return f

    def close_all_h5(self) -> None:
        for _, f in list(self._h5_cache.items()):
            try:
                f.close()
            except Exception:
                pass
        self._h5_cache.clear()

    # -----------------------------
    # Run registry helpers
    # -----------------------------

    def add_run(self, key: RunKey, file_path: Path) -> None:
        self.runs[key] = Path(file_path).resolve()

    def get_run_path(self, key: RunKey) -> Path:
        if key not in self.runs:
            raise KeyError(f"Run not registered: {key.short()}")
        return self.runs[key]

    def iter_runs(self, *, config_id: Optional[str] = None, entity: Optional[Entity] = None) -> Iterable[Tuple[RunKey, Path]]:
        for k, p in self.runs.items():
            if config_id is not None and k.config_id != config_id:
                continue
            if entity is not None and k.entity != entity:
                continue
            yield k, p

    # -----------------------------
    # Frame cache helpers
    # -----------------------------

    def set_frame(self, key: RunKey, detector: int, data: np.ndarray, errors: Optional[np.ndarray] = None) -> None:
        self.frames[(key, detector)] = data
        if errors is not None:
            self.frame_errors[(key, detector)] = errors

    def get_frame(self, key: RunKey, detector: int) -> np.ndarray:
        return self.frames[(key, detector)]

    def get_frame_errors(self, key: RunKey, detector: int) -> Optional[np.ndarray]:
        return self.frame_errors.get((key, detector))

    # -----------------------------
    # Refs_sub helpers
    # -----------------------------

    def set_refs_sub(self, config_id: str, file_path: Path) -> None:
        self.refs_sub_files[config_id] = Path(file_path).resolve()

    def get_refs_sub_path(self, config_id: str) -> Path:
        if config_id not in self.refs_sub_files:
            raise KeyError(f"Missing refs_sub for config_id={config_id}")
        return self.refs_sub_files[config_id]

    def set_refs_norm(self, config_id: str, file_path: Path) -> None:
        self.refs_norm_files[config_id] = Path(file_path).resolve()

    def get_refs_norm_path(self, config_id: str) -> Path:
        if config_id not in self.refs_norm_files:
            raise KeyError(f"Missing refs_norm for config_id={config_id}")
        return self.refs_norm_files[config_id]

    def set_masks_file(self, config_id: str, file_path: Path) -> None:
        self.masks_files[config_id] = Path(file_path).resolve()

    def get_masks_file_path(self, config_id: str) -> Path:
        if config_id not in self.masks_files:
            raise KeyError(f"Missing masks file for config_id={config_id}")
        return self.masks_files[config_id]

    def set_mask(self, config_id: str, detector: int, mask: np.ndarray) -> None:
        self.masks[(config_id, detector)] = mask

    def get_mask(self, config_id: str, detector: int) -> Optional[np.ndarray]:
        return self.masks.get((config_id, detector))

    def set_transmission(self, sample_id: str, config_id: str, value: float) -> None:
        self.transmissions[(sample_id, config_id)] = value

    def get_transmission(self, sample_id: str, config_id: str) -> Optional[float]:
        return self.transmissions.get((sample_id, config_id))

    def update_root_dir(self, root_dir: Path) -> WorkflowContext:
        """Update the raw data root directory and rebase registered run paths."""
        old_root_dir = self.root_dir.resolve()
        new_root_dir = Path(root_dir).resolve()
        self.runs = {
            key: _rebase_path_if_under(path, old_root_dir, new_root_dir)
            for key, path in self.runs.items()
        }
        self.root_dir = new_root_dir
        return self

    def update_output_dir(self, output_dir: Path) -> WorkflowContext:
        """Update the output directory and rebase generated file paths."""
        old_output_dir = self.output_dir.resolve()
        new_output_dir = Path(output_dir).resolve()
        self.refs_sub_files = {
            config_id: _rebase_path_if_under(path, old_output_dir, new_output_dir)
            for config_id, path in self.refs_sub_files.items()
        }
        self.refs_norm_files = {
            config_id: _rebase_path_if_under(path, old_output_dir, new_output_dir)
            for config_id, path in self.refs_norm_files.items()
        }
        self.masks_files = {
            config_id: _rebase_path_if_under(path, old_output_dir, new_output_dir)
            for config_id, path in self.masks_files.items()
        }
        self.artifacts = [
            replace(artifact, path=_rebase_path_if_under(artifact.path, old_output_dir, new_output_dir))
            for artifact in self.artifacts
        ]
        runs_report_csv = self.store.get("runs_report_csv")
        if isinstance(runs_report_csv, Path):
            self.store["runs_report_csv"] = _rebase_path_if_under(runs_report_csv, old_output_dir, new_output_dir)
        self.output_dir = new_output_dir
        return self

    def runs_table(self) -> TableView:
        """Return a notebook-friendly table view of runs using the runs_report.csv columns."""
        return TableView(
            columns=("sample_name", "config_id", "mode", "entity", "file_path"),
            rows=_runs_report_rows(self),
        )

    def configurations_table(self) -> TableView:
        """Return a notebook-friendly table view of configurations and their properties."""
        return TableView(
            columns=(
                "config_id",
                "wavelength",
                "sample_detector_distance",
                "notes",
                "has_collimation",
                "collimation_distance",
                "last_aperture_to_sample_distance",
                "aperture1_type",
                "aperture1_x_gap",
                "aperture1_y_gap",
                "aperture1_diameter",
                "aperture2_type",
                "aperture2_x_gap",
                "aperture2_y_gap",
                "aperture2_diameter",
            ),
            rows=_configurations_rows(self),
        )

    def transmissions_table(self) -> TableView:
        """Return a notebook-friendly table view of sample transmissions."""
        return TableView(
            columns=("sample_name", "config_id", "transmission"),
            rows=_transmissions_rows(self),
        )


def _write_scalar_dataset(parent: h5py.Group, name: str, value: Any) -> None:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        parent.create_dataset(name, data=np.bytes_(value))
    else:
        parent.create_dataset(name, data=value)


def _replace_scalar_dataset(parent: h5py.Group, name: str, value: Any) -> None:
    if name in parent:
        del parent[name]
    _write_scalar_dataset(parent, name, value)


def _read_text_dataset(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    return _read_text_value(value)


def _read_text_value(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(()).item()
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return value.decode(errors="replace")
    return str(value)


def _rebase_path_if_under(path: Path, old_base: Path, new_base: Path) -> Path:
    resolved_path = Path(path).resolve()
    try:
        relative_path = resolved_path.relative_to(old_base)
    except ValueError:
        return resolved_path
    return (new_base / relative_path).resolve()


def _read_sample_name(path: Path) -> str:
    with h5py.File(path, "r") as h5:
        for entry_path in ("/raw_data", "/entry", "/entry0", "/entry1"):
            if entry_path not in h5:
                continue
            for dataset_path in (f"{entry_path}/sample/name", f"{entry_path}/title"):
                if dataset_path in h5:
                    try:
                        return _read_text_dataset(h5[dataset_path]).strip() or path.stem
                    except Exception:
                        continue
    return path.stem


def _classify_entity_from_sample_name(sample_name: str) -> Entity:
    normalized = re.sub(r"[^a-z0-9]+", "", sample_name.strip().lower())
    if "emptybeam" in normalized:
        return "empty_beam"
    if normalized == "emptycell":
        return "empty_cell"
    if normalized in {"cd", "cadmium", "b4c", "dark"}:
        return "dark"
    return "sample"


def _flattened_nxsas_name(input_dir: Path, raw_path: Path) -> str:
    relative = raw_path.relative_to(input_dir)
    stem = relative.with_suffix("").as_posix().replace("/", "__")
    return f"{stem}.nxs"


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _is_hdf5_file(path: Path) -> bool:
    try:
        return h5py.is_hdf5(path)
    except Exception:
        return False


def _path_to_storage_string(path: Path, *, base_dir: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _resolve_stored_path(raw_path: str, *, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return {"__type__": "path", "value": str(value)}
    if isinstance(value, list):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_to_json_compatible(item) for item in value]}
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Only string dict keys are supported")
            out[key] = _to_json_compatible(item)
        return out
    raise TypeError(f"Unsupported value type: {type(value).__name__}")


def _from_json_compatible(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_json_compatible(item) for item in value]
    if isinstance(value, dict):
        value_type = value.get("__type__")
        if value_type == "path":
            return Path(str(value["value"]))
        if value_type == "tuple":
            return tuple(_from_json_compatible(item) for item in value["items"])
        return {key: _from_json_compatible(item) for key, item in value.items()}
    return value


def _serialize_configuration(parent: h5py.Group, configuration: Any) -> None:
    _write_scalar_dataset(parent, "wavelength", float(configuration.wavelength))
    sample_detector_distance = configuration.sample_detector_distance
    if isinstance(sample_detector_distance, list):
        parent.create_dataset("sample_detector_distance", data=np.asarray(sample_detector_distance, dtype=np.float64))
    else:
        _write_scalar_dataset(parent, "sample_detector_distance", float(sample_detector_distance))
    if configuration.config_id is not None:
        _write_scalar_dataset(parent, "config_id", configuration.config_id)
    if configuration.notes is not None:
        _write_scalar_dataset(parent, "notes", configuration.notes)

    if configuration.collimation is None:
        return

    col = parent.create_group("collimation")
    _write_scalar_dataset(col, "collimation_distance", float(configuration.collimation.collimation_distance))
    _write_scalar_dataset(
        col,
        "last_aperture_to_sample_distance",
        float(configuration.collimation.last_aperture_to_sample_distance),
    )
    for aperture_name in ("aperture1", "aperture2"):
        aperture = getattr(configuration.collimation, aperture_name)
        ap = col.create_group(aperture_name)
        _write_scalar_dataset(ap, "type", aperture.type)
        if aperture.x_gap is not None:
            _write_scalar_dataset(ap, "x_gap", float(aperture.x_gap))
        if aperture.y_gap is not None:
            _write_scalar_dataset(ap, "y_gap", float(aperture.y_gap))
        if aperture.diameter is not None:
            _write_scalar_dataset(ap, "diameter", float(aperture.diameter))


def _deserialize_configuration(group: h5py.Group) -> Any:
    from scarlet.workflow.configuration import Aperture, Collimation, Configuration

    sample_detector_distance_ds = group["sample_detector_distance"][()]
    sample_detector_distance_arr = np.asarray(sample_detector_distance_ds)
    if sample_detector_distance_arr.ndim == 0:
        sample_detector_distance: float | list[float] = float(sample_detector_distance_arr.reshape(()))
    else:
        sample_detector_distance = [float(item) for item in sample_detector_distance_arr.tolist()]

    collimation = None
    if "collimation" in group:
        col_group = group["collimation"]

        def read_aperture(ap_group: h5py.Group) -> Aperture:
            return Aperture(
                type=_read_text_dataset(ap_group["type"]),
                x_gap=float(ap_group["x_gap"][()]) if "x_gap" in ap_group else None,
                y_gap=float(ap_group["y_gap"][()]) if "y_gap" in ap_group else None,
                diameter=float(ap_group["diameter"][()]) if "diameter" in ap_group else None,
            )

        collimation = Collimation(
            aperture1=read_aperture(col_group["aperture1"]),
            aperture2=read_aperture(col_group["aperture2"]),
            collimation_distance=float(col_group["collimation_distance"][()]),
            last_aperture_to_sample_distance=float(col_group["last_aperture_to_sample_distance"][()]),
        )

    return Configuration(
        wavelength=float(group["wavelength"][()]),
        sample_detector_distance=sample_detector_distance,
        collimation=collimation,
        config_id=_read_text_dataset(group["config_id"]) if "config_id" in group else None,
        notes=_read_text_dataset(group["notes"]) if "notes" in group else None,
    )


def _runs_report_rows(workflow_context: WorkflowContext) -> List[Dict[str, str]]:
    rows: List[Tuple[int, Dict[str, str]]] = []
    for key, path in workflow_context.runs.items():
        stat = path.stat()
        timestamp_ns = getattr(stat, "st_birthtime_ns", None)
        if timestamp_ns is None:
            timestamp_ns = getattr(stat, "st_ctime_ns", int(stat.st_ctime * 1_000_000_000))
        rows.append(
            (
                timestamp_ns,
                {
                    "sample_name": key.sample_name or "",
                    "config_id": key.config_id,
                    "mode": key.mode,
                    "entity": key.entity,
                    "file_path": str(path),
                },
            )
        )

    rows.sort(
        key=lambda item: (
            item[0],
            item[1]["sample_name"],
            item[1]["config_id"],
            item[1]["mode"],
            item[1]["entity"],
            item[1]["file_path"],
        )
    )
    return [row for _, row in rows]


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def _configurations_rows(workflow_context: WorkflowContext) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for config_id, cfg in sorted(workflow_context.configurations.items()):
        collimation = getattr(cfg, "collimation", None)
        aperture1 = None if collimation is None else collimation.aperture1
        aperture2 = None if collimation is None else collimation.aperture2
        rows.append(
            {
                "config_id": config_id,
                "wavelength": _format_value(getattr(cfg, "wavelength", None)),
                "sample_detector_distance": _format_value(getattr(cfg, "sample_detector_distance", None)),
                "notes": _format_value(getattr(cfg, "notes", None)),
                "has_collimation": "True" if collimation is not None else "False",
                "collimation_distance": _format_value(None if collimation is None else collimation.collimation_distance),
                "last_aperture_to_sample_distance": _format_value(
                    None if collimation is None else collimation.last_aperture_to_sample_distance
                ),
                "aperture1_type": _format_value(None if aperture1 is None else aperture1.type),
                "aperture1_x_gap": _format_value(None if aperture1 is None else aperture1.x_gap),
                "aperture1_y_gap": _format_value(None if aperture1 is None else aperture1.y_gap),
                "aperture1_diameter": _format_value(None if aperture1 is None else aperture1.diameter),
                "aperture2_type": _format_value(None if aperture2 is None else aperture2.type),
                "aperture2_x_gap": _format_value(None if aperture2 is None else aperture2.x_gap),
                "aperture2_y_gap": _format_value(None if aperture2 is None else aperture2.y_gap),
                "aperture2_diameter": _format_value(None if aperture2 is None else aperture2.diameter),
            }
        )
    return rows


def _transmissions_rows(workflow_context: WorkflowContext) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for (sample_name, config_id), value in sorted(
        workflow_context.transmissions.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        rows.append(
            {
                "sample_name": sample_name,
                "config_id": config_id,
                "transmission": _format_value(value),
            }
        )
    return rows


def save_workflow_context(
    workflow_context: WorkflowContext,
    file_path: str | Path,
) -> Path:
    """
    Save the lightweight state of a WorkflowContext to a NeXus/HDF5 file.

    Heavy transient caches such as open HDF5 handles, frames and frame errors
    are intentionally excluded.
    """
    file_path = Path(file_path).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = file_path.parent

    with h5py.File(file_path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        _write_scalar_dataset(entry, "definition", "SCARLET_workflow_context")
        _write_scalar_dataset(entry, "schema_version", "1.0")

        metadata = entry.create_group("metadata")
        metadata.attrs["NX_class"] = np.bytes_("NXcollection")
        _write_scalar_dataset(metadata, "experiment_id", workflow_context.experiment_id)
        _write_scalar_dataset(metadata, "instrument_name", workflow_context.instrument_name)
        _write_scalar_dataset(metadata, "root_dir", _path_to_storage_string(workflow_context.root_dir, base_dir=base_dir))
        _write_scalar_dataset(metadata, "output_dir", _path_to_storage_string(workflow_context.output_dir, base_dir=base_dir))
        _write_scalar_dataset(metadata, "schema_raw", workflow_context.schema_raw)
        _write_scalar_dataset(metadata, "schema_refs_sub", workflow_context.schema_refs_sub)
        _write_scalar_dataset(metadata, "schema_refs_norm", workflow_context.schema_refs_norm)
        _write_scalar_dataset(metadata, "schema_masks", workflow_context.schema_masks)
        _write_scalar_dataset(metadata, "created_utc", datetime.now(timezone.utc).isoformat())

        runs_group = entry.create_group("runs")
        runs_group.attrs["NX_class"] = np.bytes_("NXcollection")
        dt = h5py.string_dtype(encoding="utf-8")
        rows = _runs_report_rows(workflow_context)
        runs_group.create_dataset("sample_name", data=np.asarray([row["sample_name"] for row in rows], dtype=dt))
        runs_group.create_dataset("config_id", data=np.asarray([row["config_id"] for row in rows], dtype=dt))
        runs_group.create_dataset("mode", data=np.asarray([row["mode"] for row in rows], dtype=dt))
        runs_group.create_dataset("entity", data=np.asarray([row["entity"] for row in rows], dtype=dt))
        runs_group.create_dataset(
            "file_path",
            data=np.asarray(
                [_path_to_storage_string(Path(row["file_path"]), base_dir=base_dir) for row in rows],
                dtype=dt,
            ),
        )

        configs_group = entry.create_group("configurations")
        configs_group.attrs["NX_class"] = np.bytes_("NXcollection")
        for config_id, configuration in sorted(workflow_context.configurations.items()):
            cfg_group = configs_group.create_group(config_id)
            cfg_group.attrs["NX_class"] = np.bytes_("NXcollection")
            _serialize_configuration(cfg_group, configuration)

        refs_group = entry.create_group("references")
        refs_group.attrs["NX_class"] = np.bytes_("NXcollection")
        for group_name, mapping in (
            ("refs_sub_files", workflow_context.refs_sub_files),
            ("refs_norm_files", workflow_context.refs_norm_files),
            ("masks_files", workflow_context.masks_files),
        ):
            subgroup = refs_group.create_group(group_name)
            subgroup.attrs["NX_class"] = np.bytes_("NXcollection")
            for config_id, path in sorted(mapping.items()):
                _write_scalar_dataset(subgroup, config_id, _path_to_storage_string(path, base_dir=base_dir))

        masks_group = entry.create_group("masks")
        masks_group.attrs["NX_class"] = np.bytes_("NXcollection")
        for (config_id, detector), mask in sorted(workflow_context.masks.items()):
            cfg_group = masks_group.require_group(config_id)
            cfg_group.attrs["NX_class"] = np.bytes_("NXcollection")
            cfg_group.create_dataset(f"detector{detector}", data=np.asarray(mask, dtype=np.uint8))

        transmissions_group = entry.create_group("transmissions")
        transmissions_group.attrs["NX_class"] = np.bytes_("NXcollection")
        transmissions_group.create_dataset(
            "sample_name",
            data=np.asarray([sample_name for sample_name, _ in workflow_context.transmissions.keys()], dtype=dt),
        )
        transmissions_group.create_dataset(
            "config_id",
            data=np.asarray([config_id for _, config_id in workflow_context.transmissions.keys()], dtype=dt),
        )
        transmissions_group.create_dataset(
            "value",
            data=np.asarray(list(workflow_context.transmissions.values()), dtype=np.float64),
        )

        artifacts_group = entry.create_group("artifacts")
        artifacts_group.attrs["NX_class"] = np.bytes_("NXcollection")
        artifacts_group.create_dataset("name", data=np.asarray([artifact.name for artifact in workflow_context.artifacts], dtype=dt))
        artifacts_group.create_dataset(
            "path",
            data=np.asarray(
                [_path_to_storage_string(artifact.path, base_dir=base_dir) for artifact in workflow_context.artifacts],
                dtype=dt,
            ),
        )
        artifacts_group.create_dataset("kind", data=np.asarray([artifact.kind for artifact in workflow_context.artifacts], dtype=dt))
        artifacts_group.create_dataset(
            "created_utc",
            data=np.asarray([artifact.created_utc for artifact in workflow_context.artifacts], dtype=dt),
        )

        logs_group = entry.create_group("logs")
        logs_group.attrs["NX_class"] = np.bytes_("NXcollection")
        logs_group.create_dataset("level", data=np.asarray([log.level for log in workflow_context.logs], dtype=dt))
        logs_group.create_dataset("message", data=np.asarray([log.message for log in workflow_context.logs], dtype=dt))
        logs_group.create_dataset(
            "where",
            data=np.asarray([(log.where or "") for log in workflow_context.logs], dtype=dt),
        )
        logs_group.create_dataset("when_utc", data=np.asarray([log.when_utc for log in workflow_context.logs], dtype=dt))
        logs_group.create_dataset(
            "meta_json",
            data=np.asarray([json.dumps(_to_json_compatible(log.meta), sort_keys=True) for log in workflow_context.logs], dtype=dt),
        )

        issues_group = entry.create_group("issues")
        issues_group.attrs["NX_class"] = np.bytes_("NXcollection")
        issues_group.create_dataset("level", data=np.asarray([issue.level for issue in workflow_context.issues], dtype=dt))
        issues_group.create_dataset("message", data=np.asarray([issue.message for issue in workflow_context.issues], dtype=dt))
        issues_group.create_dataset(
            "where",
            data=np.asarray([(issue.where or "") for issue in workflow_context.issues], dtype=dt),
        )
        issues_group.create_dataset(
            "key",
            data=np.asarray([(issue.key or "") for issue in workflow_context.issues], dtype=dt),
        )
        issues_group.create_dataset("when_utc", data=np.asarray([issue.when_utc for issue in workflow_context.issues], dtype=dt))
        issues_group.create_dataset(
            "meta_json",
            data=np.asarray([json.dumps(_to_json_compatible(issue.meta), sort_keys=True) for issue in workflow_context.issues], dtype=dt),
        )

        timings_group = entry.create_group("timings")
        timings_group.attrs["NX_class"] = np.bytes_("NXcollection")
        for key, value in sorted(workflow_context.timings.items()):
            _write_scalar_dataset(timings_group, key, float(value))

        store_group = entry.create_group("store")
        store_group.attrs["NX_class"] = np.bytes_("NXcollection")
        skipped_store_keys: list[str] = []
        for key, value in sorted(workflow_context.store.items()):
            if key in {"frames", "frame_errors", "masks", "transmissions"}:
                continue
            try:
                encoded = json.dumps(_to_json_compatible(value), sort_keys=True)
            except TypeError:
                skipped_store_keys.append(key)
                continue
            _write_scalar_dataset(store_group, key, encoded)
        if skipped_store_keys:
            _write_scalar_dataset(store_group, "_skipped_keys", json.dumps(sorted(skipped_store_keys)))

    return file_path


def load_workflow_context(file_path: str | Path) -> WorkflowContext:
    """Load a WorkflowContext previously saved with save_workflow_context()."""
    file_path = Path(file_path).resolve()
    base_dir = file_path.parent

    with h5py.File(file_path, "r") as f:
        entry = f["/entry"]
        definition = _read_text_dataset(entry["definition"])
        if definition != "SCARLET_workflow_context":
            raise ValueError(f"Unsupported workflow context definition: {definition!r}")

        metadata = entry["metadata"]
        workflow_context = WorkflowContext(
            experiment_id=_read_text_dataset(metadata["experiment_id"]),
            instrument_name=_read_text_dataset(metadata["instrument_name"]),
            root_dir=_resolve_stored_path(_read_text_dataset(metadata["root_dir"]), base_dir=base_dir),
            output_dir=_resolve_stored_path(_read_text_dataset(metadata["output_dir"]), base_dir=base_dir),
        )
        workflow_context.schema_raw = _read_text_dataset(metadata["schema_raw"])
        workflow_context.schema_refs_sub = _read_text_dataset(metadata["schema_refs_sub"])
        workflow_context.schema_refs_norm = _read_text_dataset(metadata["schema_refs_norm"])
        workflow_context.schema_masks = _read_text_dataset(metadata["schema_masks"])

        runs_group = entry["runs"]
        sample_names = [_read_text_value(runs_group["sample_name"][i]) for i in range(len(runs_group["sample_name"]))]
        config_ids = [_read_text_value(runs_group["config_id"][i]) for i in range(len(runs_group["config_id"]))]
        modes = [_read_text_value(runs_group["mode"][i]) for i in range(len(runs_group["mode"]))]
        entities = [_read_text_value(runs_group["entity"][i]) for i in range(len(runs_group["entity"]))]
        file_paths = [_read_text_value(runs_group["file_path"][i]) for i in range(len(runs_group["file_path"]))]
        for sample_name, config_id, mode, entity, raw_path in zip(sample_names, config_ids, modes, entities, file_paths):
            workflow_context.add_run(
                RunKey(
                    config_id=config_id,
                    entity=cast(Entity, entity),
                    mode=cast(Mode, mode),
                    sample_name=sample_name or None,
                ),
                _resolve_stored_path(raw_path, base_dir=base_dir),
            )

        if "configurations" in entry:
            for config_id, group in entry["configurations"].items():
                if isinstance(group, h5py.Group):
                    workflow_context.configurations[config_id] = _deserialize_configuration(group)

        refs_group = entry["references"]
        for attribute_name, group_name in (
            ("refs_sub_files", "refs_sub_files"),
            ("refs_norm_files", "refs_norm_files"),
            ("masks_files", "masks_files"),
        ):
            mapping = cast(dict[str, Path], getattr(workflow_context, attribute_name))
            for key, dataset in refs_group[group_name].items():
                if isinstance(dataset, h5py.Dataset):
                    mapping[key] = _resolve_stored_path(_read_text_dataset(dataset), base_dir=base_dir)

        if "masks" in entry:
            for config_id, config_group in entry["masks"].items():
                if not isinstance(config_group, h5py.Group):
                    continue
                for detector_name, dataset in config_group.items():
                    if not isinstance(dataset, h5py.Dataset):
                        continue
                    detector = int(detector_name.removeprefix("detector"))
                    workflow_context.set_mask(config_id, detector, np.asarray(dataset[()], dtype=np.uint8))

        if "transmissions" in entry:
            transmissions_group = entry["transmissions"]
            sample_names = [_read_text_value(transmissions_group["sample_name"][i]) for i in range(len(transmissions_group["sample_name"]))]
            config_ids = [_read_text_value(transmissions_group["config_id"][i]) for i in range(len(transmissions_group["config_id"]))]
            values = np.asarray(transmissions_group["value"][()], dtype=np.float64)
            for sample_name, config_id, value in zip(sample_names, config_ids, values):
                workflow_context.set_transmission(sample_name, config_id, float(value))

        if "artifacts" in entry:
            artifacts_group = entry["artifacts"]
            for i in range(len(artifacts_group["name"])):
                workflow_context.artifacts.append(
                    Artifact(
                        name=_read_text_value(artifacts_group["name"][i]),
                        path=_resolve_stored_path(_read_text_value(artifacts_group["path"][i]), base_dir=base_dir),
                        kind=_read_text_value(artifacts_group["kind"][i]),
                        created_utc=_read_text_value(artifacts_group["created_utc"][i]),
                    )
                )

        if "logs" in entry:
            logs_group = entry["logs"]
            for i in range(len(logs_group["level"])):
                workflow_context.logs.append(
                    LogMessage(
                        level=cast(Level, _read_text_value(logs_group["level"][i])),
                        message=_read_text_value(logs_group["message"][i]),
                        where=(_read_text_value(logs_group["where"][i]) or None),
                        when_utc=_read_text_value(logs_group["when_utc"][i]),
                        meta=_from_json_compatible(json.loads(_read_text_value(logs_group["meta_json"][i]))),
                    )
                )

        if "issues" in entry:
            issues_group = entry["issues"]
            for i in range(len(issues_group["level"])):
                workflow_context.issues.append(
                    Issue(
                        level=cast(Literal["WARN", "ERROR"], _read_text_value(issues_group["level"][i])),
                        message=_read_text_value(issues_group["message"][i]),
                        where=(_read_text_value(issues_group["where"][i]) or None),
                        key=(_read_text_value(issues_group["key"][i]) or None),
                        when_utc=_read_text_value(issues_group["when_utc"][i]),
                        meta=_from_json_compatible(json.loads(_read_text_value(issues_group["meta_json"][i]))),
                    )
                )

        if "timings" in entry:
            for key, dataset in entry["timings"].items():
                if isinstance(dataset, h5py.Dataset):
                    workflow_context.timings[key] = float(dataset[()])

        if "store" in entry:
            for key, dataset in entry["store"].items():
                if not isinstance(dataset, h5py.Dataset) or key == "_skipped_keys":
                    continue
                workflow_context.store[key] = _from_json_compatible(json.loads(_read_text_dataset(dataset)))

    return workflow_context


def initialize_workflow_context_from_raw_directory(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    experiment_id: str = "experiment",
    instrument_name: str | None = None,
    overwrite: bool = False,
) -> WorkflowContext:
    from scarlet.io.converters import convert_to_scarlet_nxsas_raw
    from scarlet.io.mode_inference import guess_measurement_mode_from_nexus_image
    from scarlet.workflow.configuration import compare_configurations, configuration_from_nexus

    input_dir = Path(input_dir).resolve()
    if output_dir is None:
        output_dir = input_dir / "processed"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and not _is_relative_to(path.resolve(), output_dir)
    )
    if not candidate_files:
        raise FileNotFoundError(f"No input files found in {input_dir}")

    ctx = WorkflowContext(
        experiment_id=experiment_id,
        instrument_name=instrument_name,
        root_dir=input_dir,
        output_dir=output_dir,
    )
    ctx.set("converted_data_dir", output_dir)

    raw_files: list[Path] = []
    for path in candidate_files:
        if _is_hdf5_file(path):
            raw_files.append(path)
            continue
        ctx.warn(
            "Skipping non-HDF5 input file",
            where="initialize_workflow_context_from_raw_directory",
            key=str(path),
        )

    if not raw_files:
        raise FileNotFoundError(f"No HDF5 input files found in {input_dir}")

    for raw_path in raw_files:
        converted_path = output_dir / _flattened_nxsas_name(input_dir, raw_path)
        report = convert_to_scarlet_nxsas_raw(
            instrument_name,
            raw_path,
            converted_path,
            overwrite=overwrite,
        )
        ctx.add_artifact(converted_path.name, converted_path, kind="nexus")

        configuration, issues = configuration_from_nexus(converted_path)
        for issue in issues:
            ctx.warn(issue, where="configuration_from_nexus", key=str(converted_path))

        config_id: Optional[str] = None
        for existing_config_id, existing_configuration in ctx.configurations.items():
            same, _ = compare_configurations(configuration, existing_configuration)
            if same:
                config_id = existing_config_id
                break
        if config_id is None:
            config_id = f"config_{len(ctx.configurations) + 1}"
            ctx.configurations[config_id] = replace(configuration, config_id=config_id)

        sample_name = _read_sample_name(converted_path)
        entity = _classify_entity_from_sample_name(sample_name)
        mode_guess = guess_measurement_mode_from_nexus_image(converted_path)
        if mode_guess.mode == "transmission":
            mode: Mode = "transmission"
        elif mode_guess.mode == "scattering":
            mode = "scattering"
        else:
            mode = "transmission" if entity == "empty_beam" else "scattering"
            ctx.warn(
                "Could not confidently infer measurement mode; using heuristic fallback",
                where="initialize_workflow_context_from_raw_directory",
                key=str(converted_path),
                guessed_mode=mode,
                converter_output=str(report.output_file),
            )

        run_key = RunKey(
            config_id=config_id,
            entity=entity,
            mode=mode,
            sample_name=sample_name,
        )
        if run_key in ctx.runs:
            ctx.warn(
                "Duplicate run key detected; overwriting previous file",
                where="initialize_workflow_context_from_raw_directory",
                key=run_key.short(),
                previous_path=str(ctx.runs[run_key]),
                new_path=str(converted_path),
            )
        ctx.add_run(run_key, converted_path)

    return ctx


def write_runs_report_csv(
    workflow_context: WorkflowContext,
    csv_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    if not workflow_context.runs:
        raise ValueError("WorkflowContext has no runs")

    if csv_path is None:
        csv_path = workflow_context.output_dir / "runs_report.csv"

    csv_path = Path(csv_path).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {csv_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_name", "config_id", "mode", "entity", "file_path"))
        writer.writerows(
            (
                row["sample_name"],
                row["config_id"],
                row["mode"],
                row["entity"],
                row["file_path"],
            )
            for row in _runs_report_rows(workflow_context)
        )

    workflow_context.add_artifact(csv_path.name, csv_path, kind="csv")
    workflow_context.set("runs_report_csv", csv_path)
    return csv_path


def update_workflow_context_from_runs_report_csv(
    workflow_context: WorkflowContext,
    csv_path: str | Path | None = None,
) -> WorkflowContext:
    """
    Update a WorkflowContext from an edited runs_report.csv file.

    The CSV may be manually edited to:
    - change ``sample_name``
    - change ``mode``
    - change ``entity``
    - remove rows entirely

    The function rebuilds ``ctx.runs`` from the CSV rows that remain, refreshes
    ``ctx.configurations`` from the referenced files, and invalidates derived
    state that may no longer be valid after the manual edits.
    """
    from scarlet.workflow.configuration import configuration_from_nexus

    if csv_path is None:
        csv_path = workflow_context.get("runs_report_csv")
        if csv_path is None:
            csv_path = workflow_context.output_dir / "runs_report.csv"

    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Runs report CSV not found: {csv_path}")

    def _resolve_report_path(raw_path: str) -> Path:
        path = Path(raw_path.strip())
        if path.is_absolute():
            return path.resolve()
        return (csv_path.parent / path).resolve()

    valid_entities: set[str] = {"sample", "empty_beam", "empty_cell", "dark", "refs_sub"}
    valid_modes: set[str] = {"scattering", "transmission"}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"sample_name", "config_id", "mode", "entity", "file_path"}
        missing = required.difference(fieldnames)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"runs_report.csv missing required columns: {missing_text}")
        rows = list(reader)

    rebuilt_runs: dict[RunKey, Path] = {}
    rebuilt_configurations: dict[str, Any] = {}
    row_order = 0

    for row in rows:
        row_order += 1
        config_id = (row.get("config_id") or "").strip()
        mode = (row.get("mode") or "").strip().lower()
        entity = (row.get("entity") or "").strip().lower()
        sample_name_raw = (row.get("sample_name") or "").strip()
        file_path_raw = (row.get("file_path") or "").strip()

        if not any((config_id, mode, entity, sample_name_raw, file_path_raw)):
            continue
        if not config_id:
            raise ValueError(f"Row {row_order}: config_id must not be empty")
        if mode not in valid_modes:
            expected = ", ".join(sorted(valid_modes))
            raise ValueError(f"Row {row_order}: invalid mode {mode!r}; expected one of {expected}")
        if entity not in valid_entities:
            expected = ", ".join(sorted(valid_entities))
            raise ValueError(f"Row {row_order}: invalid entity {entity!r}; expected one of {expected}")
        if not file_path_raw:
            raise ValueError(f"Row {row_order}: file_path must not be empty")

        file_path = _resolve_report_path(file_path_raw)
        if not file_path.exists():
            raise FileNotFoundError(f"Row {row_order}: data file not found: {file_path}")

        sample_name = sample_name_raw or None
        if entity == "sample" and sample_name is None:
            raise ValueError(f"Row {row_order}: sample entity requires a non-empty sample_name")

        run_key = RunKey(
            config_id=config_id,
            entity=cast(Entity, entity),
            mode=cast(Mode, mode),
            sample_name=sample_name,
        )
        if run_key in rebuilt_runs:
            workflow_context.warn(
                "Duplicate run key detected in runs_report.csv; overwriting previous row",
                where="update_workflow_context_from_runs_report_csv",
                key=run_key.short(),
                previous_path=str(rebuilt_runs[run_key]),
                new_path=str(file_path),
            )
        rebuilt_runs[run_key] = file_path

        if config_id not in rebuilt_configurations:
            configuration, issues = configuration_from_nexus(file_path)
            for issue in issues:
                workflow_context.warn(
                    issue,
                    where="update_workflow_context_from_runs_report_csv",
                    key=str(file_path),
                )
            try:
                configuration = replace(configuration, config_id=config_id)
            except TypeError:
                pass
            rebuilt_configurations[config_id] = configuration

    workflow_context.close_all_h5()
    workflow_context.runs.clear()
    workflow_context.runs.update(rebuilt_runs)

    workflow_context.configurations.clear()
    workflow_context.configurations.update(rebuilt_configurations)

    workflow_context.frames.clear()
    workflow_context.frame_errors.clear()
    workflow_context.transmissions.clear()
    workflow_context.refs_sub_files.clear()
    workflow_context.refs_norm_files.clear()

    valid_config_ids = set(rebuilt_configurations)
    stale_mask_keys = [key for key in workflow_context.masks if key[0] not in valid_config_ids]
    for key in stale_mask_keys:
        del workflow_context.masks[key]

    workflow_context.set("runs_report_csv", csv_path)
    return workflow_context


def _normalize_sample_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _is_water_sample_name(name: str) -> bool:
    sample_norm = _normalize_sample_name(name)
    return sample_norm in {"h2o", "d2o"} or "water" in sample_norm


def generate_reference_files_from_workflow_context(ctx: WorkflowContext) -> WorkflowContext:
    from scarlet.workflow.configuration import (
        _transmission_roi_from_file,
        configuration_from_nexus,
        write_refs_norm_file,
        write_refs_sub_file,
    )

    if not ctx.runs:
        raise ValueError("WorkflowContext has no runs")

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ctx.refs_sub_files.clear()
    ctx.refs_norm_files.clear()

    count_time_cache: dict[Path, float] = {}
    transmission_roi_detector = int(ctx.get("transmission_roi_detector", 0))
    transmission_roi_half_size = int(ctx.get("transmission_roi_half_size", 1))

    def _entry_path_from_file(handle: h5py.File, path: Path) -> str:
        for entry_path in ("/raw_data", "/entry", "/entry0", "/entry1"):
            if entry_path in handle:
                return entry_path
        raise ValueError(f"No entry group found in {path}")

    def _count_time(path: Path) -> float:
        path = Path(path).resolve()
        cached = count_time_cache.get(path)
        if cached is not None:
            return cached

        handle = ctx.open_h5(path)
        entry_path = _entry_path_from_file(handle, path)
        value = float("-inf")
        for dataset_path in (
            f"{entry_path}/control/count_time",
            f"{entry_path}/instrument/monitor0/count_time",
            f"{entry_path}/instrument/monitor1/count_time",
            f"{entry_path}/instrument/monitor2/count_time",
        ):
            if dataset_path not in handle:
                continue
            try:
                value = float(handle[dataset_path][()])
                break
            except Exception:
                continue
        count_time_cache[path] = value
        return value

    all_runs = [(key, path.resolve()) for key, path in ctx.runs.items()]

    def _best_run(
        runs: list[tuple[RunKey, Path]],
        *,
        predicate,
        mode: Optional[Mode] = None,
    ) -> Optional[tuple[RunKey, Path]]:
        candidates = [
            (key, path)
            for key, path in runs
            if predicate(key) and (mode is None or key.mode == mode)
        ]
        return max(candidates, key=lambda item: _count_time(item[1]), default=None)

    config_ids = sorted({*ctx.configurations.keys(), *(key.config_id for key, _ in all_runs)})
    for config_id in config_ids:
        config_runs = [(key, path) for key, path in all_runs if key.config_id == config_id]
        if not config_runs:
            continue

        empty_beam_transmission = _best_run(
            config_runs,
            predicate=lambda key: key.entity == "empty_beam",
            mode="transmission",
        )
        if empty_beam_transmission is None:
            empty_beam_transmission = _best_run(
                config_runs,
                predicate=lambda key: key.entity == "empty_beam",
            )

        empty_beam_scattering = _best_run(
            config_runs,
            predicate=lambda key: key.entity == "empty_beam",
            mode="scattering",
        )
        empty_cell_transmission = _best_run(
            config_runs,
            predicate=lambda key: key.entity == "empty_cell",
            mode="transmission",
        )
        empty_cell_scattering = _best_run(
            config_runs,
            predicate=lambda key: key.entity == "empty_cell",
            mode="scattering",
        )
        dark = _best_run(
            config_runs,
            predicate=lambda key: key.entity == "dark",
        )

        local_water_scattering = _best_run(
            config_runs,
            predicate=lambda key: key.entity == "sample" and _is_water_sample_name(key.sample_name or ""),
            mode="scattering",
        )
        local_water_transmission = _best_run(
            config_runs,
            predicate=lambda key: key.entity == "sample" and _is_water_sample_name(key.sample_name or ""),
            mode="transmission",
        )

        global_water_scattering = _best_run(
            all_runs,
            predicate=lambda key: key.entity == "sample" and _is_water_sample_name(key.sample_name or ""),
            mode="scattering",
        )
        global_water_transmission = _best_run(
            all_runs,
            predicate=lambda key: key.entity == "sample" and _is_water_sample_name(key.sample_name or ""),
            mode="transmission",
        )

        water_scattering = local_water_scattering or global_water_scattering
        water_transmission = local_water_transmission or global_water_transmission
        if water_scattering is None:
            raise ValueError(f"Missing water scattering reference for {config_id}")
        if water_transmission is None:
            raise ValueError(f"Missing water transmission reference for {config_id}")

        water_scattering_source_config_id = None
        if water_scattering[0].config_id != config_id:
            water_scattering_source_config_id = water_scattering[0].config_id
        water_transmission_source_config_id = None
        if water_transmission[0].config_id != config_id:
            water_transmission_source_config_id = water_transmission[0].config_id

        configuration = ctx.configurations.get(config_id)
        if configuration is None:
            configuration_source = (
                empty_beam_transmission
                or water_transmission
                or water_scattering
                or config_runs[0]
            )
            configuration, issues = configuration_from_nexus(configuration_source[1])
            for issue in issues:
                ctx.warn(issue, where="configuration_from_nexus", key=str(configuration_source[1]))
        try:
            configuration = replace(configuration, config_id=config_id)
        except TypeError:
            pass
        ctx.configurations[config_id] = configuration

        masks = {
            detector: mask
            for (mask_config_id, detector), mask in ctx.masks.items()
            if mask_config_id == config_id
        }

        roi_source = empty_beam_transmission or water_transmission
        if roi_source is None:
            raise ValueError(f"Missing transmission-like source file for ROI estimation: {config_id}")

        roi = _transmission_roi_from_file(
            roi_source[1],
            transmission_roi_detector=transmission_roi_detector,
            transmission_roi_half_size=transmission_roi_half_size,
        )

        if empty_beam_transmission is None:
            raise ValueError(f"Missing empty_beam transmission reference for {config_id}")

        refs_sub_path = write_refs_sub_file(
            ctx.output_dir / f"refs_sub_{config_id}.nxs",
            configuration,
            empty_beam_transmission=empty_beam_transmission[1],
            dark=None if dark is None else dark[1],
            empty_beam_scattering=None if empty_beam_scattering is None else empty_beam_scattering[1],
            empty_cell_transmission=None if empty_cell_transmission is None else empty_cell_transmission[1],
            empty_cell_scattering=None if empty_cell_scattering is None else empty_cell_scattering[1],
            transmission_roi_detector=transmission_roi_detector,
            transmission_roi=roi,
            transmission_roi_notes="estimated from empty_beam transmission image",
            masks=masks or None,
            overwrite=True,
        ).resolve()
        ctx.set_refs_sub(config_id, refs_sub_path)
        ctx.add_artifact(refs_sub_path.name, refs_sub_path, kind="nexus")

        refs_norm_path = write_refs_norm_file(
            ctx.output_dir / f"refs_norm_{config_id}.nxs",
            configuration,
            water_scattering=water_scattering[1],
            water_transmission=water_transmission[1],
            water_scattering_source_config_id=water_scattering_source_config_id,
            water_transmission_source_config_id=water_transmission_source_config_id,
            dark=None if dark is None else dark[1],
            empty_beam_transmission=None if empty_beam_transmission is None else empty_beam_transmission[1],
            empty_beam_scattering=None if empty_beam_scattering is None else empty_beam_scattering[1],
            empty_cell_transmission=None if empty_cell_transmission is None else empty_cell_transmission[1],
            empty_cell_scattering=None if empty_cell_scattering is None else empty_cell_scattering[1],
            transmission_roi_detector=transmission_roi_detector,
            transmission_roi=roi,
            transmission_roi_notes=(
                "estimated from empty_beam transmission image"
                if empty_beam_transmission is not None
                else "estimated from water transmission image"
            ),
            masks=masks or None,
            overwrite=True,
        ).resolve()
        ctx.set_refs_norm(config_id, refs_norm_path)
        ctx.add_artifact(refs_norm_path.name, refs_norm_path, kind="nexus")

    return ctx


def update_transmissions_from_workflow_context(
    ctx: WorkflowContext,
    *,
    refs_entry_path: str = "/entry",
) -> WorkflowContext:
    from scarlet.reduction.transmission import compute_transmission

    if not ctx.runs:
        raise ValueError("WorkflowContext has no runs")

    all_runs = [(key, path.resolve()) for key, path in ctx.runs.items()]
    sample_targets = sorted(
        {
            (key.config_id, key.sample_name)
            for key, _ in all_runs
            if key.entity == "sample" and key.sample_name
        }
    )
    if not sample_targets:
        ctx.transmissions.clear()
        return ctx

    count_time_cache: dict[Path, float] = {}

    def _entry_path_from_file(handle: h5py.File, path: Path) -> str:
        for entry_path in ("/raw_data", "/entry", "/entry0", "/entry1"):
            if entry_path in handle:
                return entry_path
        raise ValueError(f"No entry group found in {path}")

    def _count_time(path: Path) -> float:
        path = Path(path).resolve()
        cached = count_time_cache.get(path)
        if cached is not None:
            return cached

        handle = ctx.open_h5(path)
        entry_path = _entry_path_from_file(handle, path)
        value = float("-inf")
        for dataset_path in (
            f"{entry_path}/control/count_time",
            f"{entry_path}/instrument/monitor0/count_time",
            f"{entry_path}/instrument/monitor1/count_time",
            f"{entry_path}/instrument/monitor2/count_time",
        ):
            if dataset_path not in handle:
                continue
            try:
                value = float(handle[dataset_path][()])
                break
            except Exception:
                continue
        count_time_cache[path] = value
        return value

    def _best_sample_transmission_run(config_id: str, sample_name: str) -> Optional[tuple[RunKey, Path]]:
        candidates = [
            (key, path)
            for key, path in all_runs
            if key.config_id == config_id
            and key.entity == "sample"
            and key.mode == "transmission"
            and key.sample_name == sample_name
        ]
        return max(candidates, key=lambda item: _count_time(item[1]), default=None)

    def _configuration_wavelength(config_id: str, refs_sub_path: Path) -> float:
        configuration = ctx.configurations.get(config_id)
        if configuration is not None:
            try:
                return float(getattr(configuration, "wavelength"))
            except Exception:
                pass

        with h5py.File(refs_sub_path, "r") as refs_file:
            dataset_path = f"{refs_entry_path}/configuration/wavelength"
            if dataset_path not in refs_file:
                raise ValueError(f"Missing configuration wavelength in refs_sub: {dataset_path}")
            return float(np.asarray(refs_file[dataset_path][()]).reshape(()))

    def _read_text_dataset(group: h5py.Group, dataset_path: str) -> Optional[str]:
        if dataset_path not in group:
            return None
        raw = group[dataset_path][()]
        if isinstance(raw, (bytes, bytearray, np.bytes_)):
            return raw.decode(errors="replace")
        return str(raw)

    ctx.transmissions.clear()
    direct_values: dict[tuple[str, str], float] = {}
    wavelength_by_config: dict[str, float] = {}

    config_ids = sorted({config_id for config_id, _ in sample_targets})
    for config_id in config_ids:
        refs_sub_path = ctx.refs_sub_files.get(config_id)
        if refs_sub_path is None:
            raise ValueError(f"Missing refs_sub for config_id={config_id}")
        refs_sub_path = refs_sub_path.resolve()
        wavelength_by_config[config_id] = _configuration_wavelength(config_id, refs_sub_path)

        with h5py.File(refs_sub_path, "r") as refs_file:
            definition_path = f"{refs_entry_path}/definition"
            if definition_path not in refs_file:
                raise ValueError(f"Missing refs_sub definition: {definition_path}")
            definition = np.asarray(refs_file[definition_path][()]).reshape(()).item()
            if isinstance(definition, (bytes, bytearray, np.bytes_)):
                definition = definition.decode(errors="replace")
            if str(definition) != "SCARLET_refs_sub":
                raise ValueError(f"Unsupported refs bundle definition: {definition!r}")

            roi_root = f"{refs_entry_path}/transmission_roi"
            roi = (
                int(np.asarray(refs_file[f"{roi_root}/x0"][()]).reshape(())),
                int(np.asarray(refs_file[f"{roi_root}/x1"][()]).reshape(())),
                int(np.asarray(refs_file[f"{roi_root}/y0"][()]).reshape(())),
                int(np.asarray(refs_file[f"{roi_root}/y1"][()]).reshape(())),
            )
            detector_raw = np.asarray(refs_file[f"{roi_root}/detector"][()]).reshape(()).item()
            if isinstance(detector_raw, (bytes, bytearray, np.bytes_)):
                detector_raw = detector_raw.decode(errors="replace")
            if isinstance(detector_raw, str) and detector_raw.lower().startswith("detector"):
                detector_number = int(detector_raw[len("detector") :])
            else:
                detector_number = int(detector_raw)

            empty_beam_source = _read_text_dataset(refs_file, f"{refs_entry_path}/meta/empty_beam_transmission_source_file")
            if empty_beam_source is None:
                raise ValueError(f"refs_sub is missing empty_beam_transmission_source_file for {config_id}")
            empty_beam_transmission_file = Path(empty_beam_source).resolve()

        config_sample_names = sorted(
            {
                sample_name
                for target_config_id, sample_name in sample_targets
                if target_config_id == config_id and sample_name is not None
            }
        )
        for sample_name in config_sample_names:
            transmission_run = _best_sample_transmission_run(config_id, sample_name)
            if transmission_run is None:
                continue
            value = compute_transmission(
                transmission_run[1],
                empty_beam_transmission_file,
                roi,
                detector_number=detector_number,
            )
            ctx.set_transmission(sample_name, config_id, value)
            direct_values[(sample_name, config_id)] = value

    for config_id, sample_name in sample_targets:
        if sample_name is None or (sample_name, config_id) in ctx.transmissions:
            continue
        wavelength = wavelength_by_config.get(config_id)
        if wavelength is None:
            refs_sub_path = ctx.refs_sub_files.get(config_id)
            if refs_sub_path is None:
                continue
            wavelength = _configuration_wavelength(config_id, refs_sub_path)
            wavelength_by_config[config_id] = wavelength

        fallback_value = None
        for (other_sample_name, other_config_id), value in sorted(direct_values.items()):
            if other_sample_name != sample_name or other_config_id == config_id:
                continue
            other_wavelength = wavelength_by_config.get(other_config_id)
            if other_wavelength is None:
                refs_sub_path = ctx.refs_sub_files.get(other_config_id)
                if refs_sub_path is None:
                    continue
                other_wavelength = _configuration_wavelength(other_config_id, refs_sub_path)
                wavelength_by_config[other_config_id] = other_wavelength
            if np.isclose(wavelength, other_wavelength, rtol=0.0, atol=1e-6):
                fallback_value = value
                break
        if fallback_value is not None:
            ctx.set_transmission(sample_name, config_id, fallback_value)
        else:
            ctx.warn(
                "No transmission file or wavelength-matched fallback found",
                where="update_transmissions_from_workflow_context",
                key=f"{config_id}:{sample_name}",
            )

    return ctx


def update_reference_masks_from_workflow_context(
    ctx: WorkflowContext,
    *,
    search_dir: str | Path | None = None,
) -> WorkflowContext:
    from scarlet.workflow.configuration import compare_configurations, configuration_from_nexus, insert_masks_in_refs_file

    if search_dir is None:
        search_dir = ctx.output_dir
    search_dir = Path(search_dir).resolve()
    if not search_dir.exists():
        raise FileNotFoundError(f"Mask search directory not found: {search_dir}")
    if not search_dir.is_dir():
        raise NotADirectoryError(f"Mask search path is not a directory: {search_dir}")

    def _read_text_dataset(group: h5py.Group, dataset_path: str) -> Optional[str]:
        if dataset_path not in group:
            return None
        raw = group[dataset_path][()]
        if isinstance(raw, (bytes, bytearray, np.bytes_)):
            return raw.decode(errors="replace")
        return str(raw)

    def _load_mask_bundle(path: Path) -> Optional[tuple[Optional[str], dict[int, np.ndarray]]]:
        try:
            with h5py.File(path, "r") as f:
                if "/entry" not in f or not isinstance(f["/entry"], h5py.Group):
                    return None
                entry = f["/entry"]
                definition = _read_text_dataset(entry, "definition")
                if definition != "SCARLET_masks":
                    return None
                if "mask" not in entry or not isinstance(entry["mask"], h5py.Group):
                    raise ValueError(f"Missing /entry/mask group in mask bundle: {path}")

                masks: dict[int, np.ndarray] = {}
                for dataset_name, dataset in entry["mask"].items():
                    if not isinstance(dataset, h5py.Dataset):
                        continue
                    match = re.fullmatch(r"mask_detector(\d+)", dataset_name)
                    if match is None:
                        continue
                    masks[int(match.group(1))] = np.asarray(dataset[()], dtype=np.uint8)
                if not masks:
                    raise ValueError(f"No mask_detectorN datasets found in mask bundle: {path}")
                return _read_text_dataset(entry, "config_id"), masks
        except OSError:
            return None

    target_config_ids = {
        *ctx.configurations.keys(),
        *ctx.refs_sub_files.keys(),
        *ctx.refs_norm_files.keys(),
        *(key.config_id for key in ctx.runs),
    }
    selected_mask_files: dict[str, Path] = {}
    selected_masks: dict[str, dict[int, np.ndarray]] = {}

    for path in sorted(search_dir.rglob("*.nxs")):
        loaded = _load_mask_bundle(path)
        if loaded is None:
            continue
        config_id, masks = loaded

        matched_config_id: Optional[str] = None
        if config_id is not None and config_id in target_config_ids:
            matched_config_id = config_id
        else:
            mask_configuration, _issues = configuration_from_nexus(path)
            for candidate_config_id in sorted(target_config_ids):
                candidate_configuration = ctx.configurations.get(candidate_config_id)
                if candidate_configuration is None:
                    continue
                same, _diffs = compare_configurations(mask_configuration, candidate_configuration)
                if same:
                    matched_config_id = candidate_config_id
                    break

        if matched_config_id is None:
            ctx.warn(
                "Could not match mask bundle to a workflow configuration",
                where="update_reference_masks_from_workflow_context",
                key=str(path),
                config_id=config_id,
            )
            continue

        previous_path = selected_mask_files.get(matched_config_id)
        if previous_path is not None:
            previous_mtime_ns = previous_path.stat().st_mtime_ns
            current_mtime_ns = path.stat().st_mtime_ns
            if current_mtime_ns < previous_mtime_ns:
                continue
            ctx.warn(
                "Multiple mask bundles matched the same configuration; keeping the newest file",
                where="update_reference_masks_from_workflow_context",
                key=matched_config_id,
                previous_path=str(previous_path),
                new_path=str(path),
            )

        selected_mask_files[matched_config_id] = path.resolve()
        selected_masks[matched_config_id] = masks

    ctx.masks_files.clear()
    for config_id, mask_path in selected_mask_files.items():
        ctx.set_masks_file(config_id, mask_path)

        stale_mask_keys = [key for key in ctx.masks if key[0] == config_id]
        for key in stale_mask_keys:
            del ctx.masks[key]
        for detector, mask in selected_masks[config_id].items():
            ctx.set_mask(config_id, detector, mask)

        refs_sub_path = ctx.refs_sub_files.get(config_id)
        if refs_sub_path is not None and refs_sub_path.exists():
            insert_masks_in_refs_file(refs_sub_path, mask=mask_path)

        refs_norm_path = ctx.refs_norm_files.get(config_id)
        if refs_norm_path is not None and refs_norm_path.exists():
            insert_masks_in_refs_file(refs_norm_path, mask=mask_path)

    return ctx


def integrate_scattering_from_workflow_context(
    ctx: WorkflowContext,
    *,
    n_bins: int = 200,
    refs_entry_path: str = "/entry",
) -> WorkflowContext:
    from scarlet.reduction import (
        azimuthal_average,
        compute_q_norm_map,
        compute_q_resolution_circular,
        normalize_by_monitor,
        normalize_by_solid_angle,
        subtract_scattering_references,
    )

    if not ctx.runs:
        raise ValueError("WorkflowContext has no runs")

    if int(n_bins) <= 0:
        raise ValueError(f"n_bins must be > 0, got {n_bins!r}")

    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    def _entry_path_from_file(handle: h5py.File, path: Path) -> str:
        for entry_path in ("/raw_data", "/entry", "/entry0", "/entry1"):
            if entry_path in handle:
                return entry_path
        raise ValueError(f"No entry group found in {path}")

    def _read_monitor_value(handle: h5py.File, entry_path: str, *, path: Path) -> float:
        dataset_path = f"{entry_path}/control/integral"
        if dataset_path not in handle:
            raise ValueError(f"Missing monitor integral in {path}: {dataset_path}")
        monitor_value = float(np.asarray(handle[dataset_path][()]).reshape(()))
        if not np.isfinite(monitor_value) or monitor_value <= 0.0:
            raise ValueError(f"Monitor integral must be > 0 in {path}: {dataset_path}")
        return monitor_value

    def _available_detectors(path: Path) -> list[int]:
        path = Path(path).resolve()
        with h5py.File(path, "r") as handle:
            entry_path = _entry_path_from_file(handle, path)
            instrument_path = f"{entry_path}/instrument"
            if instrument_path not in handle or not isinstance(handle[instrument_path], h5py.Group):
                raise ValueError(f"Missing instrument group in {path}: {instrument_path}")
            detector_numbers: list[int] = []
            for name, group in handle[instrument_path].items():
                if not isinstance(group, h5py.Group):
                    continue
                match = re.fullmatch(r"detector(\d+)", name)
                if match is None:
                    continue
                if f"{instrument_path}/{name}/data" in handle:
                    detector_numbers.append(int(match.group(1)))
            if not detector_numbers:
                raise ValueError(f"No detectorN/data datasets found in {path}")
            return sorted(detector_numbers)

    def _read_raw_detector_image_and_error(
        path: Path,
        detector_number: int,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        path = Path(path).resolve()
        with h5py.File(path, "r") as handle:
            entry_path = _entry_path_from_file(handle, path)
            detector_path = f"{entry_path}/instrument/detector{detector_number}"
            data_path = f"{detector_path}/data"
            if data_path not in handle:
                raise ValueError(f"Missing detector data in {path}: {data_path}")
            x_pixel_size_path = f"{detector_path}/x_pixel_size"
            y_pixel_size_path = f"{detector_path}/y_pixel_size"
            if x_pixel_size_path not in handle or y_pixel_size_path not in handle:
                raise ValueError(f"Missing pixel size datasets in {path}: {detector_path}")

            raw_data = np.asarray(handle[data_path][()], dtype=np.float64)
            if raw_data.ndim != 2:
                raise ValueError(f"Detector data must be 2D in {path}, got shape {raw_data.shape}")
            monitor_value = _read_monitor_value(handle, entry_path, path=path)
            normalized = normalize_by_monitor(raw_data, monitor_value)
            error = np.sqrt(np.clip(raw_data, 0.0, None)) / monitor_value
            pixel_size = (
                float(np.asarray(handle[x_pixel_size_path][()]).reshape(())),
                float(np.asarray(handle[y_pixel_size_path][()]).reshape(())),
            )
            return normalized, error, pixel_size

    def _read_reference_detector_image_and_error(
        refs_file: h5py.File,
        reference_name: str,
        detector_number: int,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        reference_root = f"{refs_entry_path}/references/{reference_name}/entry"
        if reference_root not in refs_file or not isinstance(refs_file[reference_root], h5py.Group):
            return None, None
        data_path = f"{reference_root}/instrument/detector{detector_number}/data"
        if data_path not in refs_file:
            return None, None

        raw_data = np.asarray(refs_file[data_path][()], dtype=np.float64)
        if raw_data.ndim != 2:
            raise ValueError(f"Reference detector data must be 2D, got shape {raw_data.shape}")
        monitor_value = _read_monitor_value(refs_file, reference_root, path=Path(reference_name))
        normalized = normalize_by_monitor(raw_data, monitor_value)
        error = np.sqrt(np.clip(raw_data, 0.0, None)) / monitor_value
        return normalized, error

    def _combine_masks(
        base_mask: Optional[np.ndarray],
        invalid_mask: np.ndarray,
    ) -> np.ndarray:
        invalid = np.asarray(invalid_mask, dtype=bool)
        if base_mask is None:
            return invalid.astype(np.uint8)
        base = np.asarray(base_mask, dtype=bool)
        if base.shape != invalid.shape:
            raise ValueError(f"Mask shape mismatch: expected {invalid.shape}, got {base.shape}")
        return np.logical_or(base, invalid).astype(np.uint8)

    def _read_beam_center_and_mask(
        refs_sub_path: Path,
        detector_number: int,
    ) -> tuple[tuple[float, float], Optional[np.ndarray]]:
        with h5py.File(refs_sub_path, "r") as refs_file:
            beam_center_root = f"{refs_entry_path}/beam_center/detector{detector_number}"
            beam_center_x_path = f"{beam_center_root}/beam_center_x"
            beam_center_y_path = f"{beam_center_root}/beam_center_y"
            if beam_center_x_path not in refs_file or beam_center_y_path not in refs_file:
                raise ValueError(f"Missing beam center in refs_sub: {beam_center_root}")
            beam_center = (
                float(np.asarray(refs_file[beam_center_x_path][()]).reshape(())),
                float(np.asarray(refs_file[beam_center_y_path][()]).reshape(())),
            )
            mask_path = f"{refs_entry_path}/mask/mask_detector{detector_number}"
            mask = None
            if mask_path in refs_file:
                mask = np.asarray(refs_file[mask_path][()], dtype=np.uint8)
            return beam_center, mask

    def _scalar_detector_distance(configuration: Any, detector_number: int) -> float:
        value = getattr(configuration, "sample_detector_distance", None)
        if isinstance(value, list):
            if len(value) == 1:
                value = value[0]
            elif detector_number < len(value):
                value = value[detector_number]
            else:
                raise ValueError(f"Missing detector distance for detector{detector_number}")
        distance = float(value)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError(f"Invalid sample_detector_distance: {value!r}")
        return distance

    def _empty_cell_transmission_from_context(config_id: str) -> Optional[float]:
        for key, _ in ctx.iter_runs(config_id=config_id, entity="empty_cell"):
            if key.sample_name is None:
                continue
            value = ctx.get_transmission(key.sample_name, config_id)
            if value is not None:
                return float(value)
        return None

    def _empty_cell_transmission_from_refs(refs_file: h5py.File) -> Optional[float]:
        dataset_path = f"{refs_entry_path}/references/empty_cell_transmission/entry/sample/transmission"
        if dataset_path not in refs_file:
            return None
        value = float(np.asarray(refs_file[dataset_path][()]).reshape(()))
        return value if np.isfinite(value) and value > 0.0 else None

    def _sample_transmission(sample_name: str, config_id: str) -> float:
        value = ctx.get_transmission(sample_name, config_id)
        if value is None:
            raise ValueError(f"Missing transmission in workflow context for sample={sample_name!r}, config_id={config_id}")
        return float(value)

    scattering_runs = [
        (key, path.resolve())
        for key, path in ctx.runs.items()
        if key.entity == "sample" and key.mode == "scattering" and key.sample_name
    ]
    if not scattering_runs:
        raise ValueError("WorkflowContext has no sample scattering runs")

    for key, path in scattering_runs:
        sample_name = key.sample_name or "sample"
        config_id = key.config_id
        refs_sub_path = ctx.refs_sub_files.get(config_id)
        if refs_sub_path is None:
            raise ValueError(f"Missing refs_sub for config_id={config_id}")
        refs_sub_path = refs_sub_path.resolve()
        refs_norm_path = ctx.refs_norm_files.get(config_id)
        if refs_norm_path is None:
            raise ValueError(f"Missing refs_norm for config_id={config_id}")
        refs_norm_path = refs_norm_path.resolve()

        configuration = ctx.configurations.get(config_id)
        if configuration is None:
            raise ValueError(f"Missing configuration for config_id={config_id}")
        wavelength = float(getattr(configuration, "wavelength"))
        sample_t = _sample_transmission(sample_name, config_id)
        for detector_number in _available_detectors(path):
            detector_distance = _scalar_detector_distance(configuration, detector_number)
            sample_image, sample_error, pixel_size = _read_raw_detector_image_and_error(path, detector_number)
            beam_center, mask = _read_beam_center_and_mask(refs_sub_path, detector_number)

            with h5py.File(refs_sub_path, "r") as refs_file:
                dark_image, dark_error = _read_reference_detector_image_and_error(refs_file, "dark", detector_number)
                empty_cell_image, empty_cell_error = _read_reference_detector_image_and_error(
                    refs_file,
                    "empty_cell_scattering",
                    detector_number,
                )
                empty_beam_image, empty_beam_error = _read_reference_detector_image_and_error(
                    refs_file,
                    "empty_beam_scattering",
                    detector_number,
                )

                empty_cell_t = None
                if empty_cell_image is not None:
                    empty_cell_t = _empty_cell_transmission_from_context(config_id)
                    if empty_cell_t is None:
                        empty_cell_t = _empty_cell_transmission_from_refs(refs_file)
                    if empty_cell_t is None:
                        raise ValueError(f"Missing empty-cell transmission for config_id={config_id}")

            corrected = subtract_scattering_references(
                sample_image,
                sample_t,
                dark=dark_image,
                empty_cell=empty_cell_image,
                empty_cell_transmission=empty_cell_t,
                empty_beam=empty_beam_image,
                empty_beam_transmission=1.0 if empty_beam_image is not None else None,
                distance=detector_distance,
                beam_center=beam_center,
            )

            corrected_variance = np.square(sample_error) / (sample_t * sample_t)
            if dark_error is not None and empty_cell_image is None:
                corrected_variance += np.square(dark_error) / (sample_t * sample_t)
            elif dark_error is not None and empty_cell_image is not None and empty_cell_t is not None:
                corrected_variance += np.square(dark_error) * np.square((1.0 / empty_cell_t) - (1.0 / sample_t))
            if empty_cell_error is not None and empty_cell_t is not None:
                corrected_variance += np.square(empty_cell_error) / (empty_cell_t * empty_cell_t)
            corrected_error = np.sqrt(corrected_variance)

            solid_angle_corrected = normalize_by_solid_angle(
                corrected,
                detector_distance=detector_distance,
                beam_center=beam_center,
                pixel_size=pixel_size,
            )
            solid_angle_correction = normalize_by_solid_angle(
                np.ones_like(corrected, dtype=np.float64),
                detector_distance=detector_distance,
                beam_center=beam_center,
                pixel_size=pixel_size,
            )
            solid_angle_error = corrected_error * solid_angle_correction

            with h5py.File(refs_norm_path, "r") as refs_norm_file:
                water_corrected_image, _ = _read_reference_detector_image_and_error(
                    refs_norm_file,
                    "water_corrected",
                    detector_number,
                )
            if water_corrected_image is None:
                raise ValueError(
                    f"Missing water_corrected in refs_norm for config_id={config_id}, detector{detector_number}"
                )
            if water_corrected_image.shape != solid_angle_corrected.shape:
                raise ValueError(
                    f"water_corrected shape mismatch for detector{detector_number}: "
                    f"expected {solid_angle_corrected.shape}, got {water_corrected_image.shape}"
                )
            invalid_water_mask = ~np.isfinite(water_corrected_image) | (water_corrected_image <= 0.0)
            solid_angle_corrected = solid_angle_corrected / water_corrected_image
            solid_angle_error = solid_angle_error / np.abs(water_corrected_image)
            mask = _combine_masks(mask, invalid_water_mask)

            q_map = compute_q_norm_map(
                corrected,
                beam_center=beam_center,
                detector_distance=detector_distance,
                pixel_size=pixel_size,
                wavelength=wavelength,
            )

            q_error_map = None
            wavelength_spread = ctx.get("wavelength_spread")
            collimation = getattr(configuration, "collimation", None)
            if (
                wavelength_spread is not None
                and collimation is not None
                and getattr(collimation.aperture1, "diameter", None) is not None
                and getattr(collimation.aperture2, "diameter", None) is not None
            ):
                q_error_map = compute_q_resolution_circular(
                    q_map,
                    r1=float(collimation.aperture1.diameter),
                    r2=float(collimation.aperture2.diameter),
                    collimation_distance=float(collimation.collimation_distance),
                    distance=detector_distance,
                    wavelength_spread=float(wavelength_spread),
                    wavelength=wavelength,
                    pixel_size=pixel_size,
                )

            integration = azimuthal_average(
                solid_angle_corrected,
                q_map,
                mask=mask,
                intensity_error=solid_angle_error,
                q_error=q_error_map,
                n_bins=n_bins,
            )

            valid_bins = integration.counts > 0
            q_values = integration.q[valid_bins]
            intensity_values = integration.intensity[valid_bins]
            intensity_error_values = (
                np.full_like(q_values, np.nan)
                if integration.intensity_error is None
                else integration.intensity_error[valid_bins]
            )
            q_error_values = (
                np.full_like(q_values, np.nan)
                if integration.q_error is None
                else integration.q_error[valid_bins]
            )

            safe_sample_name = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_name).strip("_") or "sample"
            output_path = (ctx.output_dir / f"{safe_sample_name}_det{detector_number}_{config_id}.txt").resolve()
            header = "\n".join(
                (
                    f"sample_name: {sample_name}",
                    f"config_id: {config_id}",
                    f"detector: detector{detector_number}",
                    f"transmission: {sample_t:.6g}",
                    "columns: q i i_error q_error",
                )
            )
            np.savetxt(
                output_path,
                np.column_stack((q_values, intensity_values, intensity_error_values, q_error_values)),
                header=header,
                comments="# ",
            )
            ctx.add_artifact(output_path.name, output_path, kind="text")

    return ctx
