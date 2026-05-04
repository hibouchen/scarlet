from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal
from datetime import datetime, timezone
from collections import OrderedDict

import numpy as np
import h5py


# -----------------------------
# Small typed helpers
# -----------------------------

Level = Literal["INFO", "WARN", "ERROR"]
Mode = Literal["scattering", "transmission"]

# “entity” describes which physical run it is.
# sample_id is only meaningful when entity == "sample" (or "background" later).
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
    sample_id: Optional[str] = None  # required only for entity=="sample"

    def short(self) -> str:
        s = f"{self.config_id}:{self.entity}:{self.mode}"
        if self.sample_id:
            s += f":{self.sample_id}"
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
    root_dir: Path = field(default_factory=lambda: Path(".").resolve())
    output_dir: Path = field(default_factory=lambda: Path("./out").resolve())

    # --- schemas to validate files (optional, but handy)
    schema_raw: str = "scarlet_nxsas_raw_v1.3_mono.yaml"
    schema_refs_sub: str = "scarlet_refs_sub_v1.0.yaml"

    # --- run registry (filled by your experiment YAML loader or by code)
    runs: Dict[RunKey, Path] = field(default_factory=dict)

    # --- derived configurations (filled by reading runs)
    # key: config_id -> configuration object (from workflow/configuration.py)
    configurations: Dict[str, Any] = field(default_factory=dict)

    # --- refs_sub bundles per config
    refs_sub_files: Dict[str, Path] = field(default_factory=dict)  # config_id -> refs_sub .nxs

    # --- cached objects
    # store anything; prefer structured maps below when possible
    store: Dict[str, Any] = field(default_factory=dict)

    # --- per-run cached arrays (detector images, masks, etc.)
    # Frames are typically per detector: (RunKey, detector_index) -> np.ndarray
    frames: Dict[Tuple[RunKey, int], np.ndarray] = field(default_factory=dict)
    frame_errors: Dict[Tuple[RunKey, int], np.ndarray] = field(default_factory=dict)

    # Masks per config and detector (coming from refs_sub)
    masks: Dict[Tuple[str, int], np.ndarray] = field(default_factory=dict)  # (config_id, det) -> mask array (0/1)

    # Transmission results per sample and config (computed later)
    transmissions: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (sample_id, config_id) -> Tr

    # --- logging / issues / artefacts
    logs: List[LogMessage] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)

    # --- timings
    timings: Dict[str, float] = field(default_factory=dict)  # step_name -> seconds

    # --- internal HDF5 file cache (avoid reopening many times)
    _h5_cache: "OrderedDict[Path, h5py.File]" = field(default_factory=OrderedDict, init=False, repr=False)
    _h5_cache_size: int = field(default=8, init=False, repr=False)

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

    def set_mask(self, config_id: str, detector: int, mask: np.ndarray) -> None:
        self.masks[(config_id, detector)] = mask

    def get_mask(self, config_id: str, detector: int) -> Optional[np.ndarray]:
        return self.masks.get((config_id, detector))