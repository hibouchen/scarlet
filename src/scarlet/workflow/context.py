from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Iterator, Literal, Mapping, Optional, cast

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
# Empty-cell transmission values indexed by configuration id.
EmptyCellTransmissionValues = Dict[str, float]
# Sample thickness values indexed by sample name and configuration id.
SampleThicknessValues = Dict[tuple[str, str], float]
# Per-detector masks indexed by detector number.
DetectorMasks = Dict[int, np.ndarray]


@dataclass(frozen=True)
class RunKey:
    """Logical identifier for one workflow run."""

    config_id: str
    entity: Entity
    mode: Mode
    sample_name: Optional[str] = None
    duplicate_index: int = field(default=0, repr=False)

    def short(self) -> str:
        value = f"{self.config_id}:{self.entity}:{self.mode}"
        if self.sample_name:
            value += f":{self.sample_name}"
        if self.duplicate_index > 0:
            value += f"#{self.duplicate_index + 1}"
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


@dataclass(frozen=True)
class TableView:
    """Notebook-friendly tabular view with HTML rendering in Jupyter."""

    columns: tuple[str, ...]
    rows: list[dict[str, str]]

    def _repr_html_(self) -> str:
        """Render the table as HTML for rich notebook display."""
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
        """Render the table as a CSV-like plain-text representation."""
        lines = [",".join(self.columns)]
        for row in self.rows:
            lines.append(",".join(row.get(column, "") for column in self.columns))
        return "\n".join(lines)


@dataclass
class WorkflowContext:
    """
    Lightweight workflow state container.

    The workflow context intentionally starts small. Add behavior only when it
    is clearly needed by the next workflow step.
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
    # Water files indexed by configuration id, then by acquisition mode.
    water: Dict[str, ReferenceFilesByMode] = field(default_factory=dict)
    # Computed sample transmissions indexed by sample name and configuration id.
    transmissions: TransmissionValues = field(default_factory=dict)
    # Computed empty-cell transmissions indexed by configuration id.
    empty_cell_transmissions: EmptyCellTransmissionValues = field(default_factory=dict)
    # Sample thicknesses indexed by sample name and configuration id.
    sample_thicknesses: SampleThicknessValues = field(default_factory=dict)
    # Per-configuration detector masks cache with 1=masked, 0=valid.
    masks: Dict[str, DetectorMasks] = field(default_factory=dict)
    # SCARLET_masks bundle path indexed by configuration id.
    mask_files: Dict[str, Path] = field(default_factory=dict)
    # Prepared water flatfield artifacts indexed by configuration id.
    flatfields: Dict[str, Path] = field(default_factory=dict)
    # Optional mapping from target configuration id to source flatfield configuration id.
    flatfield_sources: Dict[str, str] = field(default_factory=dict)
    # Configurations whose flatfield artifact must be rebuilt before reuse.
    stale_flatfields: set[str] = field(default_factory=set)
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
            self.invalidate_flatfield(key.config_id)
            return
        if key.entity == "empty_beam":
            self.set_empty_beam(key.config_id, key.mode, file_path)
            self.invalidate_flatfield(key.config_id)
            return
        if key.entity == "empty_cell":
            self.set_empty_cell(key.config_id, key.mode, file_path)
            self.invalidate_flatfield(key.config_id)
            return
        if key.entity == "water":
            self.set_water(key.config_id, key.mode, file_path)
            self.invalidate_flatfield(key.config_id)

    def _allocate_run_key(self, key: RunKey) -> RunKey:
        """Return a unique run key, preserving duplicate logical runs."""
        candidate = key
        duplicate_index = key.duplicate_index
        while candidate in self.runs:
            duplicate_index += 1
            candidate = replace(key, duplicate_index=duplicate_index)
        return candidate

    def add_run(self, key: RunKey, file_path: Path) -> RunKey:
        """Register a run file under its logical key and preserve duplicates."""
        resolved_path = self._resolve_path(file_path)
        stored_key = self._allocate_run_key(key)
        self.runs[stored_key] = resolved_path
        self._sync_reference_store(stored_key, resolved_path)
        return stored_key

    def get_run_path(self, key: RunKey) -> Optional[Path]:
        """Return the registered path for a run key, or ``None`` when missing."""
        return self.runs.get(key)

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

    def runs_table(self) -> TableView:
        """Return a notebook-friendly table view of workflow runs."""
        return TableView(
            columns=("sample_name", "config_id", "mode", "entity", "thickness", "transmission", "file_path"),
            rows=_runs_rows(self),
        )

    def configurations_table(self) -> TableView:
        """Return a notebook-friendly table view of configuration parameters."""
        return TableView(
            columns=(
                "config_id",
                "wavelength",
                "sample_detector_distance",
                "collimation_distance",
                "last_aperture_to_sample_distance",
                "aperture1",
                "aperture2",
                "notes",
            ),
            rows=_configuration_rows(self),
        )

    def write_runs_table_csv(self, file_path: str | Path, *, overwrite: bool = False) -> Path:
        """Write the current runs table to a CSV file."""
        output_path = Path(file_path).resolve()
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        table = self.runs_table()
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table.columns))
            writer.writeheader()
            writer.writerows(table.rows)

        self.add_artifact(output_path.name, output_path, kind="csv")
        self.set("runs_table_csv", output_path)
        return output_path

    def update_from_runs_table_csv(self, file_path: str | Path | None = None) -> WorkflowContext:
        """Update the workflow context from a CSV previously exported from runs_table()."""
        from scarlet.workflow.configuration import configuration_from_nexus

        if file_path is None:
            file_path = self.get("runs_table_csv")
            if file_path is None:
                file_path = self.output_dir / "runs_table.csv"

        csv_path = Path(file_path).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Runs table CSV not found: {csv_path}")

        existing_by_name: dict[str, Path] = {}
        ambiguous_names: set[str] = set()
        for existing_path in self.runs.values():
            name = existing_path.name
            previous = existing_by_name.get(name)
            if previous is None:
                existing_by_name[name] = existing_path
            elif previous != existing_path:
                ambiguous_names.add(name)

        def _resolve_csv_path(raw_path: str, *, row_order: int) -> Path:
            """Resolve one CSV file path from an absolute path, a relative path, or a known file name."""
            path = Path(raw_path.strip())
            if path.is_absolute():
                resolved = path.resolve()
            else:
                candidate = (csv_path.parent / path).resolve()
                if candidate.exists():
                    resolved = candidate
                elif len(path.parts) == 1 and path.name not in ambiguous_names and path.name in existing_by_name:
                    resolved = existing_by_name[path.name]
                else:
                    raise FileNotFoundError(
                        f"Row {row_order}: data file not found or ambiguous from file_path={raw_path!r}"
                    )
            if not resolved.exists():
                raise FileNotFoundError(f"Row {row_order}: data file not found: {resolved}")
            return resolved

        valid_entities: set[str] = {"sample", "empty_beam", "empty_cell", "dark", "water"}
        valid_modes: set[str] = {"scattering", "transmission"}

        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            required = {"sample_name", "config_id", "mode", "entity", "file_path"}
            missing = required.difference(fieldnames)
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(f"runs_table.csv missing required columns: {missing_text}")
            rows = list(reader)

        rebuilt_runs: dict[RunKey, Path] = {}
        rebuilt_configurations: dict[str, Any] = {}
        rebuilt_transmissions: TransmissionValues = {}
        rebuilt_empty_cell_transmissions: EmptyCellTransmissionValues = {}
        rebuilt_sample_thicknesses: SampleThicknessValues = {}

        row_order = 0
        for row in rows:
            row_order += 1
            config_id = (row.get("config_id") or "").strip()
            mode = (row.get("mode") or "").strip().lower()
            entity = (row.get("entity") or "").strip().lower()
            sample_name_raw = (row.get("sample_name") or "").strip()
            thickness_raw = (row.get("thickness") or "").strip()
            transmission_raw = (row.get("transmission") or "").strip()
            file_path_raw = (row.get("file_path") or "").strip()

            if not any((config_id, mode, entity, sample_name_raw, thickness_raw, transmission_raw, file_path_raw)):
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

            file_path = _resolve_csv_path(file_path_raw, row_order=row_order)
            sample_name = sample_name_raw or None
            thickness = _parse_optional_float(thickness_raw, row_order=row_order)
            transmission = _parse_optional_float(transmission_raw, row_order=row_order)
            if entity == "sample" and sample_name is None:
                raise ValueError(f"Row {row_order}: sample entity requires a non-empty sample_name")

            run_key = RunKey(
                config_id=config_id,
                entity=cast(Entity, entity),
                mode=cast(Mode, mode),
                sample_name=sample_name,
            )
            stored_run_key = _allocate_unique_run_key(rebuilt_runs, run_key)
            if stored_run_key != run_key:
                self.warn(
                    "Duplicate run key detected in runs table CSV; preserving both rows",
                    where="update_from_runs_table_csv",
                    key=stored_run_key.short(),
                    previous_path=str(rebuilt_runs[replace(stored_run_key, duplicate_index=stored_run_key.duplicate_index - 1)]),
                    new_path=str(file_path),
                )
            rebuilt_runs[stored_run_key] = file_path

            if sample_name is not None and thickness is None and entity in {"sample", "water"}:
                thickness = _read_sample_thickness(file_path)
            if sample_name is not None and thickness is not None and entity in {"sample", "water"}:
                rebuilt_sample_thicknesses[(sample_name, config_id)] = thickness

            if sample_name is not None and transmission is not None:
                rebuilt_transmissions[(sample_name, config_id)] = transmission
                if entity == "empty_cell":
                    rebuilt_empty_cell_transmissions[config_id] = transmission

            if config_id not in rebuilt_configurations:
                configuration, issues = configuration_from_nexus(file_path)
                for issue in issues:
                    self.warn(issue, where="configuration_from_nexus", key=str(file_path))
                try:
                    configuration = replace(configuration, config_id=config_id)
                except TypeError:
                    pass
                rebuilt_configurations[config_id] = configuration

        self.runs.clear()
        self.configurations.clear()
        self.dark.clear()
        self.empty_beam.clear()
        self.empty_cell.clear()
        self.water.clear()
        self.transmissions.clear()
        self.empty_cell_transmissions.clear()
        self.sample_thicknesses.clear()
        self.masks.clear()
        self.mask_files.clear()
        self.flatfields.clear()
        self.flatfield_sources.clear()
        self.stale_flatfields.clear()
        self.beam_centers.clear()
        self.rois.clear()

        self.configurations.update(rebuilt_configurations)
        for run_key, path in rebuilt_runs.items():
            self.add_run(run_key, path)
        self.transmissions.update(rebuilt_transmissions)
        self.empty_cell_transmissions.update(rebuilt_empty_cell_transmissions)
        self.sample_thicknesses.update(rebuilt_sample_thicknesses)

        _initialize_transmission_geometry(self, where="update_from_runs_table_csv")
        self.set("runs_table_csv", csv_path)
        return self

    def set_beam_center(self, config_id: str, detector_number: int, center: BeamCenter) -> None:
        """Store one detector beam center for a configuration."""
        self.beam_centers.setdefault(config_id, {})[detector_number] = center

    def get_beam_center(self, config_id: str, detector_number: int) -> Optional[BeamCenter]:
        """Return one detector beam center for a configuration, or ``None`` when missing."""
        return self.beam_centers.get(config_id, {}).get(detector_number)

    def set_roi(self, config_id: str, roi: TransmissionRoi) -> None:
        """Store the transmission ROI for a configuration."""
        self.rois[config_id] = roi

    def get_roi(self, config_id: str) -> Optional[TransmissionRoi]:
        """Return the transmission ROI for a configuration, or ``None`` when missing."""
        return self.rois.get(config_id)

    def set_dark(self, config_id: str, file_path: Path) -> None:
        """Store the scattering dark file for a configuration."""
        self.dark[config_id] = self._resolve_path(file_path)
        self.invalidate_flatfield(config_id)

    def get_dark(self, config_id: str) -> Optional[Path]:
        """Return the scattering dark file for a configuration, or ``None`` when missing."""
        return self.dark.get(config_id)

    def set_empty_beam(self, config_id: str, mode: Mode, file_path: Path) -> None:
        """Store one empty-beam reference file for a configuration and mode."""
        self.empty_beam.setdefault(config_id, {})[mode] = self._resolve_path(file_path)
        self.invalidate_flatfield(config_id)

    def get_empty_beam(self, config_id: str, mode: Mode) -> Optional[Path]:
        """Return one empty-beam reference file for a configuration and mode, or ``None`` when missing."""
        return self.empty_beam.get(config_id, {}).get(mode)

    def set_empty_cell(self, config_id: str, mode: Mode, file_path: Path) -> None:
        """Store one empty-cell reference file for a configuration and mode."""
        self.empty_cell.setdefault(config_id, {})[mode] = self._resolve_path(file_path)
        self.invalidate_flatfield(config_id)

    def get_empty_cell(self, config_id: str, mode: Mode) -> Optional[Path]:
        """Return one empty-cell reference file for a configuration and mode, or ``None`` when missing."""
        return self.empty_cell.get(config_id, {}).get(mode)

    def set_water(self, config_id: str, mode: Mode, file_path: Path) -> None:
        """Store one water reference file for a configuration and mode."""
        self.water.setdefault(config_id, {})[mode] = self._resolve_path(file_path)
        self.invalidate_flatfield(config_id)

    def get_water(self, config_id: str, mode: Mode) -> Optional[Path]:
        """Return one water reference file for a configuration and mode, or ``None`` when missing."""
        return self.water.get(config_id, {}).get(mode)

    def set_transmission(self, sample_name: str, config_id: str, value: float) -> None:
        """Store one computed sample transmission."""
        self.transmissions[(sample_name, config_id)] = float(value)

    def get_transmission(self, sample_name: str, config_id: str) -> Optional[float]:
        """Return one computed sample transmission, or ``None`` when missing."""
        return self.transmissions.get((sample_name, config_id))

    def set_empty_cell_transmission(self, config_id: str, value: float) -> None:
        """Store one computed empty-cell transmission for a configuration."""
        self.empty_cell_transmissions[config_id] = float(value)
        self.invalidate_flatfield(config_id)

    def get_empty_cell_transmission(self, config_id: str) -> Optional[float]:
        """Return one computed empty-cell transmission, or ``None`` when missing."""
        return self.empty_cell_transmissions.get(config_id)

    def set_sample_thickness(self, sample_name: str, config_id: str, value: float) -> None:
        """Store one sample thickness in meters."""
        self.sample_thicknesses[(sample_name, config_id)] = float(value)

    def get_sample_thickness(self, sample_name: str, config_id: str) -> Optional[float]:
        """Return one stored sample thickness in meters, or ``None`` when missing."""
        return self.sample_thicknesses.get((sample_name, config_id))

    def set_mask(self, config_id: str, detector_number: int, mask: np.ndarray) -> None:
        """Store one detector mask for a configuration with 1=masked and 0=valid."""
        mask_array = np.asarray(mask, dtype=np.uint8)
        if mask_array.ndim != 2:
            raise ValueError(f"mask for detector{detector_number} must be 2D, got shape {mask_array.shape}")
        if not np.all((mask_array == 0) | (mask_array == 1)):
            raise ValueError("mask values must use the SCARLET convention 1=masked, 0=valid")
        self.masks.setdefault(config_id, {})[int(detector_number)] = np.array(mask_array, copy=True)
        self.invalidate_flatfield(config_id)

    def set_masks(self, config_id: str, masks: Mapping[int, np.ndarray]) -> None:
        """Store all detector masks for a configuration in the in-memory cache."""
        stored: DetectorMasks = {}
        for detector_number, mask in masks.items():
            mask_array = np.asarray(mask, dtype=np.uint8)
            if mask_array.ndim != 2:
                raise ValueError(f"mask for detector{detector_number} must be 2D, got shape {mask_array.shape}")
            if not np.all((mask_array == 0) | (mask_array == 1)):
                raise ValueError("mask values must use the SCARLET convention 1=masked, 0=valid")
            stored[int(detector_number)] = np.array(mask_array, copy=True)
        self.masks[config_id] = stored
        self.invalidate_flatfield(config_id)

    def set_mask_file(self, config_id: str, file_path: Path) -> None:
        """Register a SCARLET_masks bundle for one configuration."""
        resolved = self._resolve_path(file_path)
        _load_masks_from_bundle(resolved)
        self.mask_files[config_id] = resolved
        self.masks.pop(config_id, None)
        self.invalidate_flatfield(config_id)

    def get_mask_file(self, config_id: str) -> Optional[Path]:
        """Return the SCARLET_masks bundle path for a configuration, or ``None`` when missing."""
        return self.mask_files.get(config_id)

    def attach_mask_bundle(self, file_path: str | Path, *, config_id: str | None = None) -> str:
        """Attach one SCARLET_masks bundle to the workflow."""
        resolved = self._resolve_path(Path(file_path))
        definition, bundle_config_id = _read_bundle_definition_and_config_id(resolved)
        if definition != "SCARLET_masks":
            raise ValueError(f"Unsupported mask bundle definition in {resolved}: {definition!r}")
        attached_config_id = config_id or bundle_config_id
        if attached_config_id is None:
            matched_config_ids = self._find_matching_config_ids_for_mask_bundle(resolved)
            if len(matched_config_ids) == 1:
                attached_config_id = matched_config_ids[0]
            elif len(matched_config_ids) > 1:
                raise ValueError(
                    "Multiple workflow configurations match mask bundle; "
                    f"please provide config_id explicitly: {resolved}"
                )
            else:
                raise ValueError(f"No workflow configuration matched mask bundle: {resolved}")
        if not attached_config_id:
            raise ValueError(f"Missing config_id in mask bundle and no override provided: {resolved}")
        if config_id and bundle_config_id and bundle_config_id != config_id:
            raise ValueError(
                f"Mask bundle config_id mismatch for {resolved}: file has {bundle_config_id!r}, override is {config_id!r}"
            )
        self.set_mask_file(attached_config_id, resolved)
        return attached_config_id

    def attach_mask_bundles_from_output_dir(
        self,
        output_dir: str | Path | None = None,
    ) -> Dict[str, Path]:
        """Discover and attach every SCARLET_masks bundle matching one workflow configuration."""
        search_root = self.output_dir if output_dir is None else self._resolve_path(Path(output_dir))
        selected: Dict[str, Path] = {}
        mtimes: Dict[str, float] = {}
        for path in sorted(search_root.rglob("*")):
            if not path.is_file() or not _is_hdf5_file(path):
                continue
            try:
                definition, bundle_config_id = _read_bundle_definition_and_config_id(path)
            except Exception:
                continue
            if definition != "SCARLET_masks":
                continue
            try:
                matched_config_ids = self._find_matching_config_ids_for_mask_bundle(path)
            except Exception as exc:
                self.warn(
                    "Failed to read mask bundle configuration; skipping file",
                    where="attach_mask_bundles_from_output_dir",
                    key=str(path.resolve()),
                    error=str(exc),
                )
                continue
            if not matched_config_ids:
                self.warn(
                    "No workflow configuration matched mask bundle; skipping file",
                    where="attach_mask_bundles_from_output_dir",
                    key=str(path.resolve()),
                    bundle_config_id=bundle_config_id,
                )
                continue
            resolved = path.resolve()
            mtime = resolved.stat().st_mtime
            for matched_config_id in matched_config_ids:
                previous = selected.get(matched_config_id)
                if previous is None or mtime > mtimes[matched_config_id] or (
                    mtime == mtimes[matched_config_id] and str(resolved) > str(previous)
                ):
                    if previous is not None and previous != resolved:
                        self.warn(
                            "Multiple mask bundles found for one configuration; keeping the most recent file",
                            where="attach_mask_bundles_from_output_dir",
                            key=matched_config_id,
                            previous_path=str(previous),
                            selected_path=str(resolved),
                        )
                    selected[matched_config_id] = resolved
                    mtimes[matched_config_id] = mtime
        for bundle_config_id, path in selected.items():
            self.set_mask_file(bundle_config_id, path)
        return dict(selected)

    def _find_matching_config_ids_for_mask_bundle(self, file_path: Path) -> list[str]:
        """Return workflow configuration ids compatible with one mask bundle snapshot."""
        from scarlet.workflow.configuration import compare_configurations

        bundle_configuration = _read_mask_bundle_configuration(file_path)
        matched_config_ids: list[str] = []
        for existing_config_id, existing_configuration in self.configurations.items():
            same, _ = compare_configurations(bundle_configuration, existing_configuration)
            if same:
                matched_config_ids.append(existing_config_id)
        return matched_config_ids

    def get_mask(self, config_id: str, detector_number: int) -> Optional[np.ndarray]:
        """Return one detector mask for a configuration, loading a SCARLET_masks bundle when configured."""
        if config_id in self.mask_files and config_id not in self.masks:
            self.masks[config_id] = _load_masks_from_bundle(self.mask_files[config_id])
        mask = self.masks.get(config_id, {}).get(int(detector_number))
        if mask is None:
            return None
        return np.array(mask, copy=True)

    def get_masks(self, config_id: str) -> DetectorMasks:
        """Return copies of all detector masks for a configuration."""
        if config_id in self.mask_files and config_id not in self.masks:
            self.masks[config_id] = _load_masks_from_bundle(self.mask_files[config_id])
        return {
            detector_number: np.array(mask, copy=True)
            for detector_number, mask in self.masks.get(config_id, {}).items()
        }

    def set_flatfield(self, config_id: str, file_path: Path) -> None:
        """Store one prepared flatfield artifact for a configuration."""
        self.flatfields[config_id] = self._resolve_path(file_path)
        self.stale_flatfields.discard(config_id)

    def invalidate_flatfield(self, config_id: str) -> None:
        """Mark one configuration flatfield as stale so it is rebuilt before reuse."""
        self.flatfields.pop(config_id, None)
        self.stale_flatfields.add(config_id)

    def get_flatfield(self, config_id: str) -> Optional[Path]:
        """Return one prepared flatfield artifact for a configuration, or ``None`` when missing."""
        effective_config_id = self.resolve_flatfield_config(config_id)
        return self.flatfields.get(effective_config_id)

    def set_flatfield_source(self, config_id: str, source_config_id: str) -> str:
        """Declare that one configuration must reuse the flatfield of another configuration."""
        config_id = str(config_id).strip()
        source_config_id = str(source_config_id).strip()
        if not config_id or not source_config_id:
            raise ValueError("config_id and source_config_id must not be empty")
        if config_id == source_config_id:
            self.flatfield_sources.pop(config_id, None)
            return config_id

        previous = self.flatfield_sources.get(config_id)
        self.flatfield_sources[config_id] = source_config_id
        try:
            resolved = self.resolve_flatfield_config(config_id)
        except Exception:
            if previous is None:
                self.flatfield_sources.pop(config_id, None)
            else:
                self.flatfield_sources[config_id] = previous
            raise

        self.invalidate_flatfield(config_id)
        return resolved

    def get_flatfield_source(self, config_id: str) -> Optional[str]:
        """Return the direct source configuration used for flatfield reuse, if any."""
        return self.flatfield_sources.get(config_id)

    def resolve_flatfield_config(self, config_id: str) -> str:
        """Resolve the effective configuration that provides the flatfield for one target configuration."""
        current = str(config_id).strip()
        if not current:
            raise ValueError("config_id must not be empty")
        seen: list[str] = []
        while True:
            if current in seen:
                cycle = " -> ".join([*seen, current])
                raise ValueError(f"Cycle detected in flatfield source mapping: {cycle}")
            seen.append(current)
            source = self.flatfield_sources.get(current)
            if source is None or source == current:
                return current
            current = source

    def build_water_flatfield(
        self,
        config_id: str,
        *,
        output_path: str | Path | None = None,
        overwrite: bool = False,
        detector_number_for_transmission: int = 0,
    ) -> Path:
        """Compute and persist the water flatfield for one configuration."""
        from .normalization import build_water_flatfield_from_workflow_context

        return build_water_flatfield_from_workflow_context(
            self,
            config_id,
            output_path=output_path,
            overwrite=overwrite,
            detector_number_for_transmission=detector_number_for_transmission,
        )

    def compute_transmissions(self, *, detector_number: int = 0) -> TransmissionValues:
        """Compute transmissions for sample and empty-cell transmission runs."""
        for entity in ("sample", "empty_cell"):
            for key, path in self.iter_runs(entity=cast(Entity, entity), mode="transmission"):
                if key.sample_name is None:
                    continue
                empty_beam_path = self.get_empty_beam(key.config_id, "transmission")
                roi = self.get_roi(key.config_id)
                if empty_beam_path is None or roi is None:
                    self.warn(
                        "Skipping transmission computation because prerequisites are missing",
                        where="compute_transmissions",
                        key=key.short(),
                        missing_empty_beam=empty_beam_path is None,
                        missing_roi=roi is None,
                    )
                    continue
                value = compute_transmission(
                    path,
                    empty_beam_path,
                    roi,
                    detector_number=detector_number,
                )
                self.set_transmission(key.sample_name, key.config_id, value)
                if key.entity == "empty_cell":
                    self.set_empty_cell_transmission(key.config_id, value)
        return dict(self.transmissions)

    def get_reference_file(self, ref_name: Entity, mode: Mode, config_id: str) -> Optional[Path]:
        """Return the path of a reference file for a configuration, or ``None`` when missing."""
        if ref_name == "dark":
            if mode != "scattering":
                return None
            return self.get_dark(config_id)
        if ref_name == "empty_beam":
            return self.get_empty_beam(config_id, mode)
        if ref_name == "empty_cell":
            return self.get_empty_cell(config_id, mode)
        if ref_name == "water":
            return self.get_water(config_id, mode)

        # Fallback to the run registry for references not yet mirrored in dedicated stores.
        for key, path in self.iter_runs(config_id=config_id, entity=ref_name, mode=mode):
            return path
        return None

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


def _format_value(value: Any) -> str:
    """Format scalar values for lightweight tabular display."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def _get_field_value(value: Any, name: str) -> Any:
    """Read one field from a dataclass-like object or mapping."""
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _format_measurement(value: Any, unit: str) -> str:
    """Format one numeric configuration value with its unit."""
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "missing"
    return f"{numeric:.6g} {unit}"


def _format_distance_values(value: Any) -> str:
    """Format one scalar or per-detector distance value."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        parts = []
        for detector_number, item in enumerate(value):
            parts.append(f"detector{detector_number}={_format_measurement(item, 'm')}")
        return "; ".join(parts)
    return _format_measurement(value, "m")


def _format_aperture(aperture: Any) -> str:
    """Format one aperture definition for display."""
    if aperture is None:
        return ""
    aperture_type = _get_field_value(aperture, "type")
    if aperture_type == "slit":
        return (
            "slit "
            f"x={_format_measurement(_get_field_value(aperture, 'x_gap'), 'm')} "
            f"y={_format_measurement(_get_field_value(aperture, 'y_gap'), 'm')}"
        )
    if aperture_type == "pinhole":
        return f"pinhole d={_format_measurement(_get_field_value(aperture, 'diameter'), 'm')}"
    return str(aperture)


def _parse_optional_float(value: str, *, row_order: Optional[int] = None) -> Optional[float]:
    """Parse an optional floating-point string from CSV workflow metadata."""
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError as exc:
        if row_order is None:
            raise ValueError(f"Invalid transmission value: {value!r}") from exc
        raise ValueError(f"Row {row_order}: invalid transmission value {value!r}") from exc


def _allocate_unique_run_key(mapping: Dict[RunKey, Path], key: RunKey) -> RunKey:
    """Return a unique run key for one mapping while preserving duplicates."""
    candidate = key
    duplicate_index = key.duplicate_index
    while candidate in mapping:
        duplicate_index += 1
        candidate = replace(key, duplicate_index=duplicate_index)
    return candidate


def _runs_rows(workflow_context: WorkflowContext) -> list[dict[str, str]]:
    """Build sorted run rows for notebook-friendly display."""
    rows: list[tuple[int, dict[str, str]]] = []
    for key, path in workflow_context.runs.items():
        transmission = None
        thickness = None
        if key.sample_name is not None:
            transmission = workflow_context.transmissions.get((key.sample_name, key.config_id))
            thickness = workflow_context.sample_thicknesses.get((key.sample_name, key.config_id))

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
                    "thickness": _format_value(thickness),
                    "transmission": _format_value(transmission),
                    "file_path": path.name,
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
            item[1]["thickness"],
            item[1]["transmission"],
            item[1]["file_path"],
        )
    )
    return [row for _, row in rows]


def _configuration_rows(workflow_context: WorkflowContext) -> list[dict[str, str]]:
    """Build sorted configuration rows for notebook-friendly display."""
    rows: list[dict[str, str]] = []
    for config_id, configuration in sorted(workflow_context.configurations.items()):
        collimation = _get_field_value(configuration, "collimation")
        rows.append(
            {
                "config_id": config_id,
                "wavelength": _format_measurement(_get_field_value(configuration, "wavelength"), "A"),
                "sample_detector_distance": _format_distance_values(
                    _get_field_value(configuration, "sample_detector_distance")
                ),
                "collimation_distance": _format_measurement(
                    _get_field_value(collimation, "collimation_distance"),
                    "m",
                ),
                "last_aperture_to_sample_distance": _format_measurement(
                    _get_field_value(collimation, "last_aperture_to_sample_distance"),
                    "m",
                ),
                "aperture1": _format_aperture(_get_field_value(collimation, "aperture1")),
                "aperture2": _format_aperture(_get_field_value(collimation, "aperture2")),
                "notes": _format_value(_get_field_value(configuration, "notes")),
            }
        )
    return rows


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


def _read_bundle_definition_and_config_id(file_path: Path) -> tuple[str, Optional[str]]:
    """Read the NeXus definition and optional config_id from a bundle-like file."""
    with h5py.File(file_path, "r") as handle:
        definition = ""
        if "/entry/definition" in handle:
            definition = _read_text_dataset(handle["/entry/definition"]).strip()
        config_id = None
        if "/entry/config_id" in handle:
            config_id = _read_text_dataset(handle["/entry/config_id"]).strip() or None
        return definition, config_id


def _read_mask_bundle_configuration(file_path: Path):
    """Read the configuration snapshot stored in one SCARLET_masks bundle."""
    from scarlet.workflow.configuration import configuration_from_nexus

    configuration, _ = configuration_from_nexus(file_path, entry_path="/entry")
    return configuration


def _load_masks_from_bundle(file_path: Path) -> DetectorMasks:
    """Load detector masks from a SCARLET_masks bundle."""
    masks: DetectorMasks = {}
    with h5py.File(file_path, "r") as handle:
        definition_path = "/entry/definition"
        if definition_path not in handle:
            raise ValueError(f"Missing definition dataset in mask bundle: {file_path}")
        definition = _read_text_dataset(handle[definition_path]).strip()
        if definition != "SCARLET_masks":
            raise ValueError(f"Unsupported mask bundle definition in {file_path}: {definition!r}")

        convention_path = "/entry/meta/mask_convention"
        if convention_path in handle:
            convention = _read_text_dataset(handle[convention_path]).strip()
            if convention != "1=masked, 0=valid":
                raise ValueError(f"Unsupported mask convention in {file_path}: {convention!r}")

        mask_group_path = "/entry/mask"
        if mask_group_path not in handle or not isinstance(handle[mask_group_path], h5py.Group):
            raise ValueError(f"Missing mask group in mask bundle: {file_path}")

        for name, dataset in handle[mask_group_path].items():
            match = re.fullmatch(r"mask_detector(\d+)", name)
            if match is None or not isinstance(dataset, h5py.Dataset):
                continue
            mask = np.asarray(dataset[()], dtype=np.uint8)
            if mask.ndim != 2:
                raise ValueError(
                    f"Mask dataset must be 2D for detector{int(match.group(1))} in {file_path}, got {mask.shape}"
                )
            if not np.all((mask == 0) | (mask == 1)):
                raise ValueError(
                    f"Mask dataset for detector{int(match.group(1))} in {file_path} must contain only 0/1 values"
                )
            masks[int(match.group(1))] = mask
    return masks


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


def _read_sample_thickness(path: Path) -> Optional[float]:
    """Extract the optional sample thickness from a raw or converted NeXus file."""
    with h5py.File(path, "r") as handle:
        for entry_path in ("/raw_data", "/entry", "/entry0", "/entry1"):
            dataset_path = f"{entry_path}/sample/thickness"
            if dataset_path not in handle:
                continue
            try:
                value = float(np.asarray(handle[dataset_path][()]).reshape(()))
            except Exception:
                continue
            if np.isfinite(value):
                return value
    return None


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
    if normalized.startswith("water") or normalized == "h2o":
        return "water"
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
        sample_thickness = _read_sample_thickness(converted_path)
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
        stored_run_key = ctx._allocate_run_key(run_key)
        if stored_run_key != run_key:
            ctx.warn(
                "Duplicate run key detected; preserving both files",
                where=where,
                key=stored_run_key.short(),
                previous_path=str(ctx.runs[run_key]),
                new_path=str(converted_path),
            )
        ctx.add_run(run_key, converted_path)
        if sample_thickness is not None and entity in {"sample", "water"}:
            ctx.set_sample_thickness(sample_name, config_id, sample_thickness)
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
