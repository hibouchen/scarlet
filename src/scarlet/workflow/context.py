from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal, cast
from datetime import datetime, timezone
from collections import OrderedDict
import csv
import re

import numpy as np
import h5py


# -----------------------------
# Small typed helpers
# -----------------------------

Level = Literal["INFO", "WARN", "ERROR"]
Mode = Literal["scattering", "transmission"]

# “entity” describes which physical run it is.
# sample_name is only meaningful when entity == "sample" (or "background" later).
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
    sample_name: Optional[str] = None  # required only for entity=="sample"

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

    # --- run registry (filled by your experiment YAML loader or by code)
    runs: Dict[RunKey, Path] = field(default_factory=dict)

    # --- derived configurations (filled by reading runs)
    # key: config_id -> configuration object (from workflow/configuration.py)
    configurations: Dict[str, Any] = field(default_factory=dict)

    # --- refs_sub bundles per config
    refs_sub_files: Dict[str, Path] = field(default_factory=dict)  # config_id -> refs_sub .nxs
    refs_norm_files: Dict[str, Path] = field(default_factory=dict)  # config_id -> refs_norm .nxs

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

    def set_mask(self, config_id: str, detector: int, mask: np.ndarray) -> None:
        self.masks[(config_id, detector)] = mask

    def get_mask(self, config_id: str, detector: int) -> Optional[np.ndarray]:
        return self.masks.get((config_id, detector))

    def set_transmission(self, sample_id: str, config_id: str, value: float) -> None:
        self.transmissions[(sample_id, config_id)] = value

    def get_transmission(self, sample_id: str, config_id: str) -> Optional[float]:
        return self.transmissions.get((sample_id, config_id))


def _read_text_dataset(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(()).item()
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return value.decode(errors="replace")
    return str(value)


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

    rows = []
    for key, path in workflow_context.runs.items():
        stat = path.stat()
        timestamp_ns = getattr(stat, "st_birthtime_ns", None)
        if timestamp_ns is None:
            timestamp_ns = getattr(stat, "st_ctime_ns", int(stat.st_ctime * 1_000_000_000))
        rows.append(
            (
                timestamp_ns,
                key.sample_name or "",
                key.config_id,
                key.mode,
                key.entity,
                str(path),
            )
        )

    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4], row[5]))

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_name", "config_id", "mode", "entity", "file_path"))
        writer.writerows(row[1:] for row in rows)

    workflow_context.add_artifact(csv_path.name, csv_path, kind="csv")
    workflow_context.set("runs_report_csv", csv_path)
    return csv_path


def _normalize_sample_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _is_water_sample_name(name: str) -> bool:
    sample_norm = _normalize_sample_name(name)
    return sample_norm in {"h2o", "d2o"} or "water" in sample_norm


def generate_reference_files_from_workflow_context(ctx: WorkflowContext) -> None:
    from scarlet.workflow.configuration import (
        _transmission_roi_from_file,
        configuration_from_nexus,
        write_refs_norm_file,
        write_refs_sub_file,
    )

    if not ctx.runs:
        raise ValueError("WorkflowContext has no runs")

    ctx.output_dir.mkdir(parents=True, exist_ok=True)

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
