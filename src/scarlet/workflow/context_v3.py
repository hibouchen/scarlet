from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Iterator, Literal, Optional

import h5py
import numpy as np

from scarlet.reduction.transmission import (
    compute_beam_center,
    compute_transmission,
    compute_transmission_roi,
)


# Log levels used by the workflow during execution.
Level = Literal["INFO", "WARN", "ERROR"]
# Run acquisition modes.
Mode = Literal["scattering", "transmission"]
# Logical run categories handled by the workflow.
Entity = Literal["sample", "empty_beam", "empty_cell", "dark", "water"]
# One detector center is stored as (x, y).
BeamCenter = tuple[float, float]
# One configuration can store one beam center per detector number.
DetectorBeamCenters = Dict[int, BeamCenter]
# Transmission ROI stored as (x0, x1, y0, y1).
TransmissionRoi = tuple[int, int, int, int]
# Reference files stored by acquisition mode.
ReferenceFilesByMode = Dict[Mode, Path]
# Sample transmission values indexed by sample name and configuration id.
TransmissionValues = Dict[tuple[str, str], float]


@dataclass(frozen=True)
class RunKey:
    """Logical identifier for one workflow run."""

    config_id: str
    entity: Entity
    mode: Mode
    sample_name: Optional[str] = None
    transmission: Optional[float] = field(default=None, compare=False)

    def short(self) -> str:
        value = f"{self.config_id}:{self.entity}:{self.mode}"
        if self.sample_name:
            value += f":{self.sample_name}"
        return value


@dataclass(frozen=True)
class Artifact:
    """File created during workflow execution."""

    name: str
    path: Path
    kind: str = "file"
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class LogMessage:
    """Structured log entry."""

    level: Level
    message: str
    where: Optional[str] = None
    when_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Issue:
    """Structured workflow warning or error."""

    level: Literal["WARN", "ERROR"]
    message: str
    where: Optional[str] = None
    key: Optional[str] = None
    when_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowContext:
    """
    Lightweight workflow state container.

    `context_v3` intentionally starts small. Add behavior only when it is
    clearly needed by the next workflow step.
    """

    experiment_id: str = "experiment"
    instrument_name: str = "unknown"
    root_dir: Path = field(default_factory=lambda: Path(".").resolve())
    output_dir: Path = field(default_factory=lambda: Path("./out").resolve())

    # Core workflow registry: one logical run key maps to one file on disk.
    runs: Dict[RunKey, Path] = field(default_factory=dict)
    # Derived metadata can be attached here as the v3 API grows.
    configurations: Dict[str, Any] = field(default_factory=dict)
    # Beam centers indexed by configuration id, then detector number.
    beam_centers: Dict[str, DetectorBeamCenters] = field(default_factory=dict)
    # Transmission ROI indexed by configuration id.
    rois: Dict[str, TransmissionRoi] = field(default_factory=dict)
    # Scattering dark files indexed by configuration id.
    dark: Dict[str, Path] = field(default_factory=dict)
    # Empty-beam files indexed by configuration id, then by acquisition mode.
    empty_beam: Dict[str, ReferenceFilesByMode] = field(default_factory=dict)
    # Empty-cell files indexed by configuration id, then by acquisition mode.
    empty_cell: Dict[str, ReferenceFilesByMode] = field(default_factory=dict)
    # Computed sample transmissions indexed by sample name and configuration id.
    transmissions: TransmissionValues = field(default_factory=dict)
    # Free-form storage for intermediate pipeline state.
    store: Dict[str, Any] = field(default_factory=dict)

    # Execution traceability.
    logs: list[LogMessage] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    def _resolve_path(self, file_path: Path) -> Path:
        """Normalize a file path before storing it in the context."""
        return Path(file_path).resolve()

    def _sync_reference_store(self, key: RunKey, file_path: Path) -> None:
        """Mirror reference runs into the dedicated reference attributes."""
        if key.entity == "dark" and key.mode == "scattering":
            self.set_dark(key.config_id, file_path)
            return
        if key.entity == "empty_beam":
            self.set_empty_beam(key.config_id, key.mode, file_path)
            return
        if key.entity == "empty_cell":
            self.set_empty_cell(key.config_id, key.mode, file_path)

    def add_run(self, key: RunKey, file_path: Path) -> None:
        """Register a run file under its logical key."""
        resolved_path = self._resolve_path(file_path)
        self.runs[key] = resolved_path
        self._sync_reference_store(key, resolved_path)

    def get_run_path(self, key: RunKey) -> Path:
        """Return the registered path for a run key."""
        try:
            return self.runs[key]
        except KeyError as exc:
            raise KeyError(f"Run not registered: {key.short()}") from exc

    def iter_runs(
        self,
        *,
        config_id: Optional[str] = None,
        entity: Optional[Entity] = None,
        mode: Optional[Mode] = None,
        sample_name: Optional[str] = None,
    ) -> Iterator[tuple[RunKey, Path]]:
        """Iterate over runs with optional filters."""
        for key, path in self.runs.items():
            # Keep filtering simple and explicit so new criteria can be added safely.
            if config_id is not None and key.config_id != config_id:
                continue
            if entity is not None and key.entity != entity:
                continue
            if mode is not None and key.mode != mode:
                continue
            if sample_name is not None and key.sample_name != sample_name:
                continue
            yield key, path

    def set_beam_center(self, config_id: str, detector_number: int, center: BeamCenter) -> None:
        """Store one detector beam center for a configuration."""
        self.beam_centers.setdefault(config_id, {})[detector_number] = center

    def get_beam_center(self, config_id: str, detector_number: int) -> BeamCenter:
        """Return one detector beam center for a configuration."""
        try:
            return self.beam_centers[config_id][detector_number]
        except KeyError as exc:
            raise KeyError(
                f"Missing beam center for config_id={config_id}, detector_number={detector_number}"
            ) from exc

    def set_roi(self, config_id: str, roi: TransmissionRoi) -> None:
        """Store the transmission ROI for a configuration."""
        self.rois[config_id] = roi

    def get_roi(self, config_id: str) -> TransmissionRoi:
        """Return the transmission ROI for a configuration."""
        try:
            return self.rois[config_id]
        except KeyError as exc:
            raise KeyError(f"Missing ROI for config_id={config_id}") from exc

    def set_dark(self, config_id: str, file_path: Path) -> None:
        """Store the scattering dark file for a configuration."""
        self.dark[config_id] = self._resolve_path(file_path)

    def get_dark(self, config_id: str) -> Path:
        """Return the scattering dark file for a configuration."""
        try:
            return self.dark[config_id]
        except KeyError as exc:
            raise KeyError(f"Missing dark file for config_id={config_id}") from exc

    def set_empty_beam(self, config_id: str, mode: Mode, file_path: Path) -> None:
        """Store one empty-beam reference file for a configuration and mode."""
        self.empty_beam.setdefault(config_id, {})[mode] = self._resolve_path(file_path)

    def get_empty_beam(self, config_id: str, mode: Mode) -> Path:
        """Return one empty-beam reference file for a configuration and mode."""
        try:
            return self.empty_beam[config_id][mode]
        except KeyError as exc:
            raise KeyError(f"Missing empty_beam file for config_id={config_id}, mode={mode}") from exc

    def set_empty_cell(self, config_id: str, mode: Mode, file_path: Path) -> None:
        """Store one empty-cell reference file for a configuration and mode."""
        self.empty_cell.setdefault(config_id, {})[mode] = self._resolve_path(file_path)

    def get_empty_cell(self, config_id: str, mode: Mode) -> Path:
        """Return one empty-cell reference file for a configuration and mode."""
        try:
            return self.empty_cell[config_id][mode]
        except KeyError as exc:
            raise KeyError(f"Missing empty_cell file for config_id={config_id}, mode={mode}") from exc

    def set_transmission(self, sample_name: str, config_id: str, value: float) -> None:
        """Store one computed sample transmission."""
        self.transmissions[(sample_name, config_id)] = float(value)

    def get_transmission(self, sample_name: str, config_id: str) -> float:
        """Return one computed sample transmission."""
        try:
            return self.transmissions[(sample_name, config_id)]
        except KeyError as exc:
            raise KeyError(
                f"Missing transmission for sample_name={sample_name}, config_id={config_id}"
            ) from exc

    def compute_transmissions(self, *, detector_number: int = 0) -> TransmissionValues:
        """Compute transmissions for all sample transmission runs."""
        for key, path in self.iter_runs(entity="sample", mode="transmission"):
            if key.sample_name is None:
                continue
            value = compute_transmission(
                path,
                self.get_empty_beam(key.config_id, "transmission"),
                self.get_roi(key.config_id),
                detector_number=detector_number,
            )
            self.set_transmission(key.sample_name, key.config_id, value)
        return dict(self.transmissions)

    def get_reference_file(self, ref_name: Entity, mode: Mode, config_id: str) -> Path:
        """Return the path of a reference file for a configuration."""
        if ref_name == "dark":
            if mode != "scattering":
                raise KeyError(f"Missing dark file for config_id={config_id}, mode={mode}")
            return self.get_dark(config_id)
        if ref_name == "empty_beam":
            return self.get_empty_beam(config_id, mode)
        if ref_name == "empty_cell":
            return self.get_empty_cell(config_id, mode)

        # Fallback to the run registry for references not yet mirrored in dedicated stores.
        for key, path in self.iter_runs(config_id=config_id, entity=ref_name, mode=mode):
            return path
        raise KeyError(
            f"Missing reference file for entity={ref_name}, mode={mode}, config_id={config_id}"
        )

    def set(self, key: str, value: Any) -> None:
        """Store arbitrary workflow data."""
        self.store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Read arbitrary workflow data."""
        return self.store.get(key, default)

    def require(self, key: str) -> Any:
        """Read a required stored value."""
        if key not in self.store:
            raise KeyError(f"Missing required context key: {key}")
        return self.store[key]

    def add_artifact(self, name: str, path: Path, *, kind: str = "file") -> None:
        """Register a produced file."""
        self.artifacts.append(Artifact(name=name, path=self._resolve_path(path), kind=kind))

    def log(self, level: Level, message: str, *, where: Optional[str] = None, **meta: Any) -> None:
        """Append a structured log message."""
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
        """Return True when at least one error has been recorded."""
        return any(issue.level == "ERROR" for issue in self.issues)


def iter_reference_runs(
    ctx: WorkflowContext,
    *,
    config_id: Optional[str] = None,
    mode: Optional[Mode] = None,
) -> Iterable[tuple[RunKey, Path]]:
    """Iterate over non-sample runs stored in the workflow context."""
    for key, path in ctx.iter_runs(config_id=config_id, mode=mode):
        if key.entity != "sample":
            yield key, path


def _read_text_value(value: Any) -> str:
    """Convert HDF5 scalar-like text values to Python strings."""
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(()).item()
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return value.decode(errors="replace")
    return str(value)


def _read_text_dataset(dataset: h5py.Dataset) -> str:
    """Read a scalar text dataset and normalize its value to str."""
    return _read_text_value(dataset[()])


def _read_sample_name(path: Path) -> str:
    """Extract the sample name from a raw or converted NeXus file."""
    with h5py.File(path, "r") as handle:
        for entry_path in ("/raw_data", "/entry", "/entry0", "/entry1"):
            if entry_path not in handle:
                continue
            for dataset_path in (f"{entry_path}/sample/name", f"{entry_path}/title"):
                if dataset_path not in handle:
                    continue
                try:
                    return _read_text_dataset(handle[dataset_path]).strip() or path.stem
                except Exception:
                    continue
    return path.stem


def _read_beam_centers_from_file(path: Path) -> dict[int, BeamCenter]:
    """Read detector beam centers stored in a SCARLET-compatible file."""
    beam_centers: dict[int, BeamCenter] = {}
    with h5py.File(path, "r") as handle:
        entry_path = _pick_nexus_entry_path(handle)
        if entry_path is None:
            return beam_centers
        instrument_path = f"{entry_path}/instrument"
        if instrument_path not in handle or not isinstance(handle[instrument_path], h5py.Group):
            return beam_centers

        instrument = handle[instrument_path]
        for name in instrument.keys():
            match = re.fullmatch(r"detector(\d+)", name)
            if match is None:
                continue
            detector_path = f"{instrument_path}/{name}"
            beam_center_x_path = f"{detector_path}/beam_center_x"
            beam_center_y_path = f"{detector_path}/beam_center_y"
            if beam_center_x_path not in handle or beam_center_y_path not in handle:
                continue
            try:
                beam_centers[int(match.group(1))] = (
                    float(np.asarray(handle[beam_center_x_path][()]).reshape(())),
                    float(np.asarray(handle[beam_center_y_path][()]).reshape(())),
                )
            except Exception:
                continue
    return beam_centers


def _classify_entity_from_sample_name(sample_name: str) -> Entity:
    """Infer the workflow entity type from the sample name."""
    normalized = re.sub(r"[^a-z0-9]+", "", sample_name.strip().lower())
    if "emptybeam" in normalized:
        return "empty_beam"
    if normalized == "emptycell":
        return "empty_cell"
    if normalized in {"cd", "cadmium", "b4c", "dark"}:
        return "dark"
    return "sample"


def _flattened_nxsas_name(input_dir: Path, raw_path: Path) -> str:
    """Build a stable flattened output filename from an input file path."""
    relative = raw_path.relative_to(input_dir)
    stem = relative.with_suffix("").as_posix().replace("/", "__")
    return f"{stem}.nxs"


def _is_relative_to(path: Path, other: Path) -> bool:
    """Return True when a path is located under another path."""
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _is_hdf5_file(path: Path) -> bool:
    """Return True when a path points to a readable HDF5 file."""
    try:
        return h5py.is_hdf5(path)
    except Exception:
        return False


_IGNORED_INPUT_DEFINITIONS = {
    "NXsas_raw",
    "SCARLET_masks",
    "SCARLET_refs_sub",
    "SCARLET_refs_norm",
    "SCARLET_workflow_context",
}


def _pick_nexus_entry_path(handle: h5py.File) -> Optional[str]:
    """Return the most likely NXentry-like root group in a NeXus file."""
    for candidate in ("/raw_data", "/entry0", "/entry", "/entry1"):
        if candidate in handle and isinstance(handle[candidate], h5py.Group):
            return candidate
    for key in handle.keys():
        candidate = f"/{key}"
        obj = handle[candidate]
        if not isinstance(obj, h5py.Group):
            continue
        if _read_text_value(obj.attrs.get("NX_class", "")) == "NXentry":
            return candidate
    return None


def _detector_data_dimensionality_issue(entry: h5py.Group, *, entry_path: str) -> Optional[str]:
    """Return a warning message when detector data are not strictly 2D."""
    instrument_group: Optional[h5py.Group] = None
    if "instrument" in entry and isinstance(entry["instrument"], h5py.Group):
        instrument_group = entry["instrument"]
    else:
        for obj in entry.values():
            if not isinstance(obj, h5py.Group):
                continue
            if _read_text_value(obj.attrs.get("NX_class", "")) == "NXinstrument":
                instrument_group = obj
                break

    if instrument_group is None:
        return "Skipping HDF5 input file without NXinstrument under entry"

    found_detector_data = False
    for detector_name, detector_group in instrument_group.items():
        if not isinstance(detector_group, h5py.Group):
            continue
        if _read_text_value(detector_group.attrs.get("NX_class", "")) != "NXdetector":
            continue
        if "data" not in detector_group or not isinstance(detector_group["data"], h5py.Dataset):
            continue
        found_detector_data = True
        data = detector_group["data"]
        if len(data.shape) != 2:
            dataset_path = f"{entry_path}/{instrument_group.name.split('/')[-1]}/{detector_name}/data"
            return (
                "Skipping HDF5 input file with non-2D detector data "
                f"at {dataset_path} (shape={tuple(data.shape)!r})"
            )

    if not found_detector_data:
        return "Skipping HDF5 input file without detector data under NXinstrument"
    return None


def _classify_hdf5_input_candidate(path: Path) -> tuple[bool, Optional[str]]:
    """Tell whether an HDF5 file looks like a raw acquisition usable as input."""
    try:
        with h5py.File(path, "r") as handle:
            entry_path = _pick_nexus_entry_path(handle)
            if entry_path is None:
                return False, "Skipping HDF5 input file without NXentry"

            definition_path = f"{entry_path}/definition"
            if definition_path in handle:
                try:
                    definition = _read_text_dataset(handle[definition_path]).strip()
                except Exception:
                    definition = ""
                if definition in _IGNORED_INPUT_DEFINITIONS:
                    return False, f"Skipping non-raw HDF5 input file with definition {definition!r}"

            entry = handle[entry_path]
            assert isinstance(entry, h5py.Group)
            dimensionality_issue = _detector_data_dimensionality_issue(entry, entry_path=entry_path)
            if dimensionality_issue is not None:
                return False, dimensionality_issue
            return True, None
    except OSError:
        return False, "Skipping unreadable HDF5 input file"


def _next_generated_config_id(existing_config_ids: Iterable[str]) -> str:
    """Generate the next available synthetic configuration identifier."""
    taken = set(existing_config_ids)
    index = 1
    while True:
        candidate = f"config_{index}"
        if candidate not in taken:
            return candidate
        index += 1


def _ingest_raw_directory_into_workflow_context(
    ctx: WorkflowContext,
    *,
    input_dir: Path,
    output_dir: Path,
    instrument_name: str | None,
    overwrite: bool,
    where: str,
) -> WorkflowContext:
    """Convert raw files from a directory and merge them into a workflow context."""
    from scarlet.io.converters import convert_to_scarlet_nxsas_raw
    from scarlet.io.mode_inference import guess_measurement_mode_from_nexus_image
    from scarlet.workflow.configuration import compare_configurations, configuration_from_nexus

    candidate_files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and not _is_relative_to(path.resolve(), output_dir)
    )
    if not candidate_files:
        raise FileNotFoundError(f"No input files found in {input_dir}")

    raw_files: list[Path] = []
    for path in candidate_files:
        if not _is_hdf5_file(path):
            ctx.warn("Skipping non-HDF5 input file", where=where, key=str(path))
            continue
        is_raw_candidate, skip_reason = _classify_hdf5_input_candidate(path)
        if is_raw_candidate:
            raw_files.append(path)
            continue
        ctx.warn(skip_reason or "Skipping unsupported HDF5 input file", where=where, key=str(path))

    if not raw_files:
        raise FileNotFoundError(f"No HDF5 input files found in {input_dir}")

    existing_run_paths = {path.resolve() for path in ctx.runs.values()}
    existing_artifact_paths = {artifact.path.resolve() for artifact in ctx.artifacts}

    for raw_path in raw_files:
        converted_path = (output_dir / _flattened_nxsas_name(input_dir, raw_path)).resolve()
        if converted_path in existing_run_paths and not overwrite:
            continue

        if not converted_path.exists() or overwrite:
            convert_to_scarlet_nxsas_raw(
                instrument_name,
                raw_path,
                converted_path,
                overwrite=overwrite,
            )

        if converted_path not in existing_artifact_paths:
            ctx.add_artifact(converted_path.name, converted_path, kind="nexus")
            existing_artifact_paths.add(converted_path)

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
            config_id = _next_generated_config_id(ctx.configurations.keys())
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
                where=where,
                key=str(converted_path),
                guessed_mode=mode,
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
                where=where,
                key=run_key.short(),
                previous_path=str(ctx.runs[run_key]),
                new_path=str(converted_path),
            )
        ctx.add_run(run_key, converted_path)
        existing_run_paths.add(converted_path)

    _initialize_transmission_geometry(ctx, where=where)
    return ctx


def _initialize_transmission_geometry(ctx: WorkflowContext, *, where: str) -> None:
    """Initialize ROI and detector0 beam center from empty-beam transmission files."""
    for config_id, files_by_mode in sorted(ctx.empty_beam.items()):
        empty_beam_transmission = files_by_mode.get("transmission")
        if empty_beam_transmission is None:
            ctx.warn(
                "Missing empty_beam transmission file; cannot initialize ROI or beam center",
                where=where,
                key=config_id,
            )
            continue
        try:
            ctx.set_roi(config_id, compute_transmission_roi(empty_beam_transmission, detector_number=0))
            for detector_number, center in _read_beam_centers_from_file(empty_beam_transmission).items():
                ctx.set_beam_center(config_id, detector_number, center)
            ctx.set_beam_center(config_id, 0, compute_beam_center(empty_beam_transmission, detector_number=0))
        except Exception as exc:
            ctx.warn(
                "Failed to initialize transmission ROI or detector0 beam center",
                where=where,
                key=config_id,
                error=str(exc),
                empty_beam_transmission=str(empty_beam_transmission),
            )


def initialize_workflow_context_from_raw_directory(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    experiment_id: str = "experiment",
    instrument_name: str | None = None,
    overwrite: bool = False,
) -> WorkflowContext:
    """Create a fresh workflow context by scanning and converting a raw-data directory."""
    input_dir = Path(input_dir).resolve()
    if output_dir is None:
        output_dir = input_dir / "processed"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = WorkflowContext(
        experiment_id=experiment_id,
        instrument_name=instrument_name or "unknown",
        root_dir=input_dir,
        output_dir=output_dir,
    )
    ctx.set("converted_data_dir", output_dir)
    return _ingest_raw_directory_into_workflow_context(
        ctx,
        input_dir=input_dir,
        output_dir=output_dir,
        instrument_name=instrument_name,
        overwrite=overwrite,
        where="initialize_workflow_context_from_raw_directory",
    )
