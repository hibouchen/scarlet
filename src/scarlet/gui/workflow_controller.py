from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
import traceback
from typing import Any, cast

from PySide6 import QtCore

from scarlet.workflow import RunKey, WorkflowContext, initialize_workflow_context_from_raw_directory
from scarlet.workflow.context import Entity, Mode
from scarlet.workflow.pipeline import StitchingOptions

from .jobs import JobController, JobFailure
from .mask_editor import MaskEditorSource, write_mask_bundle
from .scarlet_viewer import build_mask_source
from .settings_data import (
    detect_settings_geometry,
    load_center_from_empty_beam_file,
    load_settings_image,
)


@dataclass(frozen=True)
class AcquisitionPreviewPayload:
    """Detector data and existing masks displayed for one selected run."""

    run_key: RunKey
    source: MaskEditorSource
    masks: dict[int, Any]
    beam_centers: dict[int, tuple[float, float]]
    converted_for_preview: bool
    display_count_time: float | None


class WorkflowController(QtCore.QObject):
    """Coordinate the current workflow without putting domain rules in widgets."""

    workflow_changed = QtCore.Signal(object)
    workflow_saved = QtCore.Signal()
    dirty_changed = QtCore.Signal(bool)
    busy_changed = QtCore.Signal(bool, str)
    activity = QtCore.Signal(str, str)
    failed = QtCore.Signal(str, str)
    settings_image_loaded = QtCore.Signal(object)
    acquisition_preview_loaded = QtCore.Signal(object)
    masks_changed = QtCore.Signal(str)
    flatfield_changed = QtCore.Signal(str)
    transmission_source_changed = QtCore.Signal(str)
    processing_changed = QtCore.Signal()

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
        *,
        jobs: JobController | None = None,
    ) -> None:
        super().__init__(parent)
        self._workflow: WorkflowContext | None = None
        self._workflow_path: Path | None = None
        self._dirty = False
        self._pending_preview_run_key: RunKey | None = None
        self._jobs = jobs or JobController(self)
        self._jobs.started.connect(self._on_job_started)
        self._jobs.succeeded.connect(self._on_job_succeeded)
        self._jobs.failed.connect(self._on_job_failed)
        self._jobs.finished.connect(self._on_job_finished)

    @property
    def workflow(self) -> WorkflowContext | None:
        return self._workflow

    @property
    def workflow_path(self) -> Path | None:
        return self._workflow_path

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def is_busy(self) -> bool:
        return self._jobs.is_running

    def create_workflow(
        self,
        *,
        input_dir: str | Path,
        output_dir: str | Path,
        experiment_id: str,
        instrument_name: str,
        transmission_strategy: str,
    ) -> None:
        input_path = Path(input_dir).resolve()
        output_path = Path(output_dir).resolve()
        self._jobs.start(
            "scan",
            lambda: initialize_workflow_context_from_raw_directory(
                input_path,
                output_dir=output_path,
                experiment_id=experiment_id,
                instrument_name=instrument_name,
                transmission_strategy=transmission_strategy,
            ),
        )

    def open_workflow(self, file_path: str | Path) -> None:
        path = Path(file_path).resolve()
        self._jobs.start("open", lambda: (path, WorkflowContext.load(path)))

    def refresh_workflow(self, *, include_excluded: bool = False) -> None:
        workflow = self._require_workflow()
        kind = "reconsider" if include_excluded else "refresh"
        self._jobs.start(
            kind,
            lambda: workflow.refresh_runs(include_excluded=include_excluded),
        )

    def save_workflow(self, file_path: str | Path, *, overwrite: bool = False) -> None:
        workflow = self._require_workflow()
        path = Path(file_path).resolve()
        self._jobs.start("save", lambda: workflow.save(path, overwrite=overwrite))

    def import_runs_table(self, file_path: str | Path) -> None:
        workflow = self._require_workflow()
        path = Path(file_path).resolve()
        self._jobs.start("import_csv", lambda: workflow.update_from_runs_table_csv(path))

    def export_runs_table(self, file_path: str | Path) -> None:
        workflow = self._require_workflow()
        path = Path(file_path).resolve()
        self._jobs.start(
            "export_csv",
            lambda: workflow.write_runs_table_csv(path, overwrite=True),
        )

    def load_settings_image(
        self,
        config_id: str,
        purpose: str,
        detector_number: int,
    ) -> None:
        workflow = self._require_workflow()
        if purpose not in {"center", "roi"}:
            raise ValueError(f"Invalid Settings image purpose: {purpose!r}")
        self._jobs.start(
            "settings_image",
            lambda: load_settings_image(
                workflow,
                config_id=config_id,
                purpose=cast(Any, purpose),
                detector_number=int(detector_number),
            ),
        )

    def detect_settings_geometry(
        self,
        config_id: str,
        purpose: str,
        detector_number: int,
    ) -> None:
        workflow = self._require_workflow()
        if purpose not in {"center", "roi"}:
            raise ValueError(f"Invalid Settings detection purpose: {purpose!r}")
        self._jobs.start(
            "settings_detect",
            lambda: detect_settings_geometry(
                workflow,
                config_id=config_id,
                purpose=cast(Any, purpose),
                detector_number=int(detector_number),
            ),
        )

    def load_center_from_empty_beam_file(
        self,
        config_id: str,
        detector_number: int,
    ) -> None:
        """Read the beam-center metadata stored in an empty-beam file."""
        workflow = self._require_workflow()
        self._jobs.start(
            "settings_load_center",
            lambda: load_center_from_empty_beam_file(
                workflow,
                config_id=config_id,
                detector_number=int(detector_number),
            ),
        )

    @QtCore.Slot(object)
    def load_acquisition_preview(self, run_key: object) -> None:
        workflow = self._require_workflow()
        if not isinstance(run_key, RunKey):
            raise TypeError("run_key must be a RunKey")
        if self._jobs.is_running:
            self._pending_preview_run_key = run_key
            return

        def load() -> AcquisitionPreviewPayload:
            from scarlet.io import read_count_time_value

            raw_path = workflow.get_run_path(run_key)
            if raw_path is None:
                raise KeyError(f"Unknown acquisition: {run_key.short()}")
            prepared_path = workflow.prepare_run(run_key)
            if prepared_path is None:
                raise ValueError(f"Cannot prepare acquisition: {run_key.short()}")
            source = build_mask_source(raw_path, prepared_path)
            count_time = read_count_time_value(prepared_path)
            display_count_time = None
            if count_time is not None and count_time > 0.0:
                # SANSLLB conversion stores the dead-time-corrected count rate.
                # The acquisition viewer is used to assess counting statistics,
                # so present corrected counts rather than counts per second.
                source = replace(
                    source,
                    detector_data={
                        detector_number: data * count_time
                        for detector_number, data in source.detector_data.items()
                    },
                )
                display_count_time = float(count_time)
            return AcquisitionPreviewPayload(
                run_key=run_key,
                source=source,
                masks=workflow.get_masks(run_key.config_id),
                beam_centers={
                    detector_number: center
                    for detector_number in source.detector_data
                    if (center := workflow.get_beam_center(run_key.config_id, detector_number))
                    is not None
                },
                converted_for_preview=prepared_path.resolve() != raw_path.resolve(),
                display_count_time=display_count_time,
            )

        self._jobs.start("acquisition_preview", load)

    @QtCore.Slot(str, object, object, object, bool)
    def save_acquisition_masks(
        self,
        config_id: str,
        source: object,
        masks: object,
        output_path: object,
        overwrite: bool = False,
    ) -> None:
        workflow = self._require_workflow()
        if not isinstance(source, MaskEditorSource):
            raise TypeError("source must be a MaskEditorSource")
        if not isinstance(masks, dict):
            raise TypeError("masks must be a detector-to-mask mapping")
        path = Path(output_path).resolve()
        normalized_config_id = str(config_id).strip()
        if not normalized_config_id:
            raise ValueError("config_id must not be empty")

        def save() -> tuple[str, Path]:
            path.parent.mkdir(parents=True, exist_ok=True)
            written_path = write_mask_bundle(path, source, masks, overwrite=overwrite)
            workflow.set_mask_file(normalized_config_id, written_path)
            return normalized_config_id, written_path

        self._jobs.start("save_masks", save)

    @QtCore.Slot(str, object)
    def attach_mask_file(self, config_id: str, file_path: object) -> None:
        workflow = self._require_workflow()
        config_id = str(config_id).strip()
        path = Path(file_path).resolve()
        if not config_id:
            raise ValueError("config_id must not be empty")

        def attach() -> tuple[str, Path]:
            attached_config_id = workflow.attach_mask_bundle(path, config_id=config_id)
            return attached_config_id, path

        self._jobs.start("attach_mask", attach)

    @QtCore.Slot(str)
    def find_compatible_mask(self, config_id: str) -> None:
        workflow = self._require_workflow()
        config_id = str(config_id).strip()
        if not config_id:
            raise ValueError("config_id must not be empty")

        def attach() -> tuple[str, Path]:
            path = workflow.find_compatible_mask_bundle(config_id)
            if path is None:
                raise FileNotFoundError(
                    f"No compatible SCARLET_masks bundle found in {workflow.output_dir} "
                    f"for {config_id}."
                )
            workflow.set_mask_file(config_id, path)
            return config_id, path

        self._jobs.start("find_mask", attach)

    @QtCore.Slot(str)
    def build_water_flatfield(self, config_id: str) -> None:
        workflow = self._require_workflow()
        config_id = str(config_id).strip()
        if not config_id:
            raise ValueError("config_id must not be empty")
        if config_id not in {key.config_id for key in workflow.runs}:
            raise KeyError(f"Unknown active configuration: {config_id}")
        self._jobs.start(
            "flatfield",
            lambda: (config_id, workflow.build_water_flatfield(config_id, overwrite=True)),
        )

    @QtCore.Slot(str, object)
    def attach_flatfield_file(self, config_id: str, file_path: object) -> None:
        workflow = self._require_workflow()
        config_id = str(config_id).strip()
        path = Path(file_path).resolve()
        if not config_id:
            raise ValueError("config_id must not be empty")
        if config_id not in {key.config_id for key in workflow.runs}:
            raise KeyError(f"Unknown active configuration: {config_id}")

        def attach() -> tuple[str, Path]:
            from scarlet.workflow import load_flatfield_file

            load_flatfield_file(path)
            workflow.set_flatfield(config_id, path)
            return config_id, path

        self._jobs.start("attach_flatfield", attach)

    @QtCore.Slot(str, str)
    def set_flatfield_source(self, config_id: str, source_config_id: str) -> None:
        workflow = self._require_workflow()
        try:
            resolved = workflow.set_flatfield_source(config_id, source_config_id)
        except (KeyError, TypeError, ValueError) as exc:
            self.activity.emit("ERROR", str(exc))
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self._set_dirty(True)
        self.flatfield_changed.emit(config_id)
        self.activity.emit("INFO", f"Flatfield source for {config_id} is now {resolved}.")

    @QtCore.Slot(str, str)
    def set_transmission_source(self, config_id: str, source_config_id: str) -> None:
        workflow = self._require_workflow()
        try:
            resolved = workflow.set_transmission_source(config_id, source_config_id)
        except (KeyError, TypeError, ValueError) as exc:
            self.activity.emit("ERROR", str(exc))
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self._set_dirty(True)
        self.transmission_source_changed.emit(config_id)
        self.activity.emit(
            "INFO", f"Transmission source for {config_id} is now {resolved}."
        )

    @QtCore.Slot(object)
    def process_selected(
        self,
        run_keys: object,
        *,
        n_bins: object = None,
        q_scale: str = "linear",
    ) -> None:
        """Reduce the exact sample-scattering runs selected in the GUI."""
        workflow = self._require_workflow()
        if not isinstance(run_keys, (list, tuple)) or not all(
            isinstance(run_key, RunKey) for run_key in run_keys
        ):
            raise TypeError("run_keys must be a sequence of RunKey values")
        selected = tuple(dict.fromkeys(run_keys))
        if not selected:
            return
        self._start_reduction(
            workflow,
            selected,
            kind="reduce_selected",
            n_bins=n_bins,
            q_scale=q_scale,
        )

    @QtCore.Slot()
    def process_pending(
        self,
        *,
        n_bins: object = None,
        q_scale: str = "linear",
    ) -> None:
        """Reduce every currently pending sample-scattering run."""
        workflow = self._require_workflow()
        pending = tuple(
            run_key
            for run_key in workflow.runs
            if run_key.entity == "sample"
            and run_key.mode == "scattering"
            and not workflow.is_run_processed(run_key)
        )
        if not pending:
            self.activity.emit("INFO", "No pending sample acquisition to process.")
            return
        self._start_reduction(
            workflow,
            pending,
            kind="reduce_pending",
            n_bins=n_bins,
            q_scale=q_scale,
        )

    @QtCore.Slot(str, object, object)
    def stitch_sample(
        self,
        sample_name: str,
        scale_on: object,
        normalization_factor: object = 1.0,
    ) -> None:
        """Backward-compatible entry point for stitching one sample."""
        self.stitch_selected([sample_name], scale_on, normalization_factor)

    @QtCore.Slot(object, object, object)
    def stitch_selected(
        self,
        sample_names: object,
        scale_on: object,
        normalization_factor: object = 1.0,
    ) -> None:
        """Merge the selected ready sample groups into I(Q) text curves."""
        workflow = self._require_workflow()
        if not isinstance(sample_names, (list, tuple)) or not all(
            isinstance(sample_name, str) for sample_name in sample_names
        ):
            raise TypeError("sample_names must be a sequence of sample names")
        samples = tuple(dict.fromkeys(sample_name.strip() for sample_name in sample_names))
        if not samples or any(not sample_name for sample_name in samples):
            raise ValueError("At least one non-empty sample name is required")
        reference = None if scale_on is None else str(scale_on).strip() or None
        factor = self._validate_normalization_factor(normalization_factor)
        self._start_stitching(
            workflow,
            samples,
            reference,
            factor,
            kind="stitch_selected",
        )

    @QtCore.Slot(object)
    def stitch_pending(self, normalization_factor: object = 1.0) -> None:
        """Merge every ready sample that has no merged text file in ``output_dir``."""
        workflow = self._require_workflow()
        factor = self._validate_normalization_factor(normalization_factor)
        from scarlet.workflow.pipeline import StichingPipeline

        pipeline = StichingPipeline(workflow)
        samples = tuple(
            sample_name
            for sample_name in sorted(
                {
                    run_key.sample_name
                    for run_key in workflow.runs
                    if run_key.entity == "sample"
                    and run_key.mode == "scattering"
                    and run_key.sample_name is not None
                }
            )
            if pipeline.is_sample_ready(sample_name)
            and not (workflow.output_dir / f"{sample_name}_merged.txt").is_file()
        )
        if not samples:
            self.activity.emit("INFO", "No ready sample is waiting for stitching.")
            return
        self._start_stitching(
            workflow,
            samples,
            None,
            factor,
            kind="stitch_pending",
        )

    @staticmethod
    def _validate_normalization_factor(value: object) -> float:
        try:
            factor = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("normalization_factor must be a number") from exc
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("normalization_factor must be finite and strictly positive")
        return factor

    def stitching_options(self) -> StitchingOptions:
        """Return the persisted greedy-stitching options of the open workflow."""
        return StitchingOptions.from_dict(
            self._require_workflow().get("stitching_options")
        )

    @QtCore.Slot(object)
    def set_stitching_options(self, options: object) -> None:
        """Persist GUI stitching options with the workflow context."""
        try:
            stored = (
                options
                if isinstance(options, StitchingOptions)
                else StitchingOptions.from_dict(options)
            )
        except (TypeError, ValueError) as exc:
            self.activity.emit("ERROR", str(exc))
            self.failed.emit(str(exc), traceback.format_exc())
            return
        workflow = self._require_workflow()
        workflow.set("stitching_options", stored.as_dict())
        self._set_dirty(True)
        self.activity.emit("INFO", "Updated stitching algorithm options.")

    def _start_stitching(
        self,
        workflow: WorkflowContext,
        sample_names: tuple[str, ...],
        reference: str | None,
        normalization_factor: float,
        *,
        kind: str,
    ) -> None:
        """Validate and start one background stitching job for sample groups."""

        from scarlet.workflow.pipeline import StichingPipeline

        pipeline = StichingPipeline(workflow)
        options = self.stitching_options()
        not_ready = [
            sample_name
            for sample_name in sample_names
            if not pipeline.is_sample_ready(sample_name)
        ]
        if not_ready:
            message = (
                "Stitching is blocked: every configuration must be processed first "
                f"for {', '.join(not_ready)}."
            )
            self.activity.emit("ERROR", message)
            self.failed.emit(message, "")
            return

        if reference is not None:
            missing_reference = []
            for sample_name in sample_names:
                processed_configurations = {
                    run_key.config_id
                    for run_key in workflow.runs
                    if run_key.entity == "sample"
                    and run_key.mode == "scattering"
                    and run_key.sample_name == sample_name
                    and workflow.is_run_processed(run_key)
                }
                if reference not in processed_configurations:
                    missing_reference.append(sample_name)
            if missing_reference:
                message = (
                    "Reference configuration is not processed for "
                    f"{', '.join(missing_reference)}: {reference}."
                )
                self.activity.emit("ERROR", message)
                self.failed.emit(message, "")
                return

        def stitch() -> tuple[int, int]:
            stitched_count = 0
            failed_count = 0
            for sample_name in sample_names:
                try:
                    data = pipeline.run_for_sample(
                        sample_name,
                        scale_on=reference,
                        normalization_factor=normalization_factor,
                        options=options,
                    )
                    point_count = len(data)
                    stitched_count += 1
                except Exception as exc:
                    failed_count += 1
                    workflow.error(
                        "Failed to stitch sample",
                        where="WorkflowController.stitch_selected",
                        key=sample_name,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    continue
                workflow.info(
                    "Finished GUI stitching",
                    where="WorkflowController.stitch_selected",
                    sample_name=sample_name,
                    reference_config=reference,
                    normalization_factor=normalization_factor,
                    point_count=point_count,
                )
            workflow.info(
                "Finished GUI stitching batch",
                where="WorkflowController.stitch_selected",
                attempted_count=len(sample_names),
                stitched_count=stitched_count,
                failed_count=failed_count,
                normalization_factor=normalization_factor,
            )
            return stitched_count, failed_count

        self._jobs.start(kind, stitch)

    def _start_reduction(
        self,
        workflow: WorkflowContext,
        run_keys: tuple[RunKey, ...],
        *,
        kind: str,
        n_bins: object = None,
        q_scale: str = "linear",
    ) -> None:
        from .readiness import assess_workflow_readiness

        readiness = assess_workflow_readiness(workflow)
        if not readiness.ready:
            message = "Processing is blocked: " + " ".join(readiness.errors)
            self.activity.emit("ERROR", message)
            self.failed.emit(message, "")
            return
        invalid = [
            run_key
            for run_key in run_keys
            if run_key.entity != "sample"
            or run_key.mode != "scattering"
            or run_key.sample_name is None
            or workflow.get_run_path(run_key) is None
        ]
        if invalid:
            message = "Only registered sample scattering acquisitions can be processed."
            self.activity.emit("ERROR", message)
            self.failed.emit(message, "\n".join(run_key.short() for run_key in invalid))
            return

        def reduce() -> tuple[int, int]:
            from scarlet.workflow.pipeline import ReductionPipeline

            pipeline = ReductionPipeline.default(
                workflow,
                n_bins=n_bins,
                q_scale=q_scale,
            )
            processed_count = 0
            failed_count = 0
            for run_key in run_keys:
                try:
                    pipeline.run_for_runkey(run_key)
                    processed_count += 1
                except Exception as exc:
                    failed_count += 1
                    workflow.error(
                        "Failed to process run",
                        where="WorkflowController.process_selected",
                        key=run_key.short(),
                        raw_path=str(workflow.get_run_path(run_key)),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
            workflow.info(
                "Finished GUI reduction",
                where="WorkflowController.process_selected",
                attempted_count=len(run_keys),
                processed_count=processed_count,
                failed_count=failed_count,
                n_bins=pipeline.n_bins,
                q_scale=pipeline.q_scale,
            )
            return processed_count, failed_count

        self._jobs.start(kind, reduce)

    @QtCore.Slot(str, int, object)
    def update_beam_center(
        self,
        config_id: str,
        detector_number: int,
        center: object,
    ) -> None:
        workflow = self._require_workflow()
        try:
            if not isinstance(center, (tuple, list)) or len(center) != 2:
                raise TypeError("center must contain x and y")
            stored = workflow.update_beam_center(
                config_id,
                detector_number,
                (float(center[0]), float(center[1])),
            )
        except (TypeError, ValueError) as exc:
            self.activity.emit("ERROR", str(exc))
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self._set_dirty(True)
        self.workflow_changed.emit(workflow)
        self.activity.emit(
            "INFO",
            f"Updated detector{detector_number} beam center for {config_id}: "
            f"({stored[0]:.3f}, {stored[1]:.3f}).",
        )

    @QtCore.Slot(str, object)
    def update_transmission_roi(self, config_id: str, roi: object) -> None:
        workflow = self._require_workflow()
        try:
            if not isinstance(roi, (tuple, list)) or len(roi) != 4:
                raise TypeError("ROI must contain x0, x1, y0, and y1")
            source_config_id = workflow.resolve_transmission_config(config_id)
            stored = workflow.update_roi(
                source_config_id,
                cast(Any, tuple(int(value) for value in roi)),
            )
        except (TypeError, ValueError) as exc:
            self.activity.emit("ERROR", str(exc))
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self._set_dirty(True)
        self.workflow_changed.emit(workflow)
        self.activity.emit(
            "INFO",
            f"Updated transmission ROI for {source_config_id}: {stored}.",
        )

    @QtCore.Slot(object, str)
    def update_run_mode(self, run_key: object, mode: str) -> None:
        workflow = self._require_workflow()
        if not isinstance(run_key, RunKey):
            raise TypeError("run_key must be a RunKey")
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"scattering", "transmission"}:
            raise ValueError(f"Invalid acquisition mode: {mode!r}")
        new_key = replace(
            run_key,
            mode=cast(Mode, normalized_mode),
            duplicate_index=0,
        )
        self._replace_run_key(run_key, new_key, label="acquisition mode")

    @QtCore.Slot(object, str)
    def update_run_entity(self, run_key: object, entity: str) -> None:
        self._require_workflow()
        if not isinstance(run_key, RunKey):
            raise TypeError("run_key must be a RunKey")
        normalized_entity = entity.strip().lower()
        if normalized_entity not in {"sample", "empty_beam", "empty_cell", "dark", "water"}:
            raise ValueError(f"Invalid acquisition entity: {entity!r}")
        new_key = replace(
            run_key,
            entity=cast(Entity, normalized_entity),
            duplicate_index=0,
        )
        self._replace_run_key(run_key, new_key, label="acquisition entity")

    @QtCore.Slot(object, object)
    def update_run_thickness(self, run_key: object, value: object) -> None:
        workflow = self._require_workflow()
        if not isinstance(run_key, RunKey):
            raise TypeError("run_key must be a RunKey")
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError("thickness must be a number or None")
        try:
            thickness = workflow.update_run_thickness(run_key, value)
        except (KeyError, TypeError, ValueError) as exc:
            self.workflow_changed.emit(workflow)
            self.activity.emit("ERROR", str(exc))
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self._set_dirty(True)
        self.workflow_changed.emit(workflow)
        value_text = "empty" if thickness is None else f"{thickness:g} mm"
        self.activity.emit("INFO", f"Changed thickness for {run_key.short()}: {value_text}.")

    @QtCore.Slot(object, str)
    def rename_sample(self, run_key: object, sample_name: str) -> None:
        workflow = self._require_workflow()
        if not isinstance(run_key, RunKey):
            raise TypeError("run_key must be a RunKey")
        try:
            replacements = workflow.rename_sample(run_key, sample_name)
        except (KeyError, ValueError) as exc:
            self.workflow_changed.emit(workflow)
            self.activity.emit("ERROR", str(exc))
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self._set_dirty(True)
        self.workflow_changed.emit(workflow)
        self.activity.emit(
            "INFO",
            f"Renamed {len(replacements)} acquisition(s) to sample {sample_name.strip()!r}.",
        )

    def _replace_run_key(self, run_key: RunKey, new_key: RunKey, *, label: str) -> None:
        workflow = self._require_workflow()
        try:
            stored_key = workflow.replace_run(run_key, new_key)
        except (KeyError, TypeError, ValueError) as exc:
            self.workflow_changed.emit(workflow)
            self.activity.emit("ERROR", str(exc))
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self._set_dirty(True)
        self.workflow_changed.emit(workflow)
        self.activity.emit(
            "INFO",
            f"Changed {label}: {run_key.short()} → {stored_key.short()}.",
        )

    @QtCore.Slot(object)
    def exclude_runs(self, run_keys: object) -> None:
        workflow = self._require_workflow()
        if not isinstance(run_keys, (list, tuple)) or not all(
            isinstance(run_key, RunKey) for run_key in run_keys
        ):
            raise TypeError("run_keys must be a sequence of RunKey values")
        try:
            file_paths = workflow.exclude_runs(run_keys, source="gui")
        except KeyError as exc:
            self.activity.emit("ERROR", str(exc))
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self._set_dirty(True)
        self.workflow_changed.emit(workflow)
        self.activity.emit("INFO", f"Excluded {len(file_paths)} acquisition(s).")

    def _require_workflow(self) -> WorkflowContext:
        if self._workflow is None:
            raise RuntimeError("No workflow is currently open")
        return self._workflow

    def _set_dirty(self, dirty: bool) -> None:
        dirty = bool(dirty)
        if dirty == self._dirty:
            return
        self._dirty = dirty
        self.dirty_changed.emit(dirty)

    @QtCore.Slot(str)
    def _on_job_started(self, kind: str) -> None:
        labels = {
            "scan": "Scanning acquisitions…",
            "open": "Opening workflow…",
            "refresh": "Refreshing acquisitions…",
            "reconsider": "Reconsidering excluded acquisitions…",
            "save": "Saving workflow…",
            "import_csv": "Importing acquisitions CSV…",
            "export_csv": "Exporting acquisitions CSV…",
            "settings_image": "Loading Settings detector image…",
            "settings_detect": "Detecting Settings geometry…",
            "acquisition_preview": "Loading acquisition detector image…",
            "save_masks": "Saving detector masks…",
            "attach_mask": "Attaching mask file…",
            "find_mask": "Finding compatible mask…",
            "flatfield": "Creating water flatfield…",
            "attach_flatfield": "Attaching flatfield file…",
            "reduce_selected": "Processing selected acquisitions…",
            "reduce_pending": "Processing pending acquisitions…",
            "stitch_selected": "Stitching selected samples…",
            "stitch_pending": "Stitching pending samples…",
        }
        message = labels.get(kind, "Working…")
        self.activity.emit("INFO", message)
        self.busy_changed.emit(True, message)

    @QtCore.Slot(str, object)
    def _on_job_succeeded(self, kind: str, result: Any) -> None:
        if kind == "scan":
            self._workflow = result
            self._workflow_path = None
            self._set_dirty(True)
            self.workflow_changed.emit(self._workflow)
            count = len(self._workflow.runs)
            self.activity.emit("INFO", f"Detected {count} acquisition(s).")
        elif kind == "open":
            path, workflow = result
            self._workflow = workflow
            self._workflow_path = path
            self._set_dirty(False)
            self.workflow_changed.emit(workflow)
            self.activity.emit("INFO", f"Opened workflow {path}.")
        elif kind in {"refresh", "reconsider"}:
            self._workflow = result
            self._set_dirty(True)
            self.workflow_changed.emit(self._workflow)
            self.activity.emit("INFO", f"Workflow now contains {len(self._workflow.runs)} acquisition(s).")
        elif kind == "save":
            self._workflow_path = Path(result).resolve()
            self._set_dirty(False)
            self.workflow_saved.emit()
            self.activity.emit("INFO", f"Saved workflow {self._workflow_path}.")
        elif kind == "import_csv":
            self._workflow = result
            self._set_dirty(True)
            self.workflow_changed.emit(self._workflow)
            self.activity.emit("INFO", "Imported acquisitions from CSV.")
        elif kind == "export_csv":
            exported_path = Path(result).resolve()
            self._set_dirty(True)
            self.activity.emit("INFO", f"Exported acquisitions to {exported_path}.")
        elif kind in {"settings_image", "settings_detect", "settings_load_center"}:
            self._set_dirty(True)
            self.settings_image_loaded.emit(result)
            action = {
                "settings_detect": "Detected geometry from",
                "settings_load_center": "Loaded beam center from",
            }.get(kind, "Loaded Settings image from")
            self.activity.emit("INFO", f"{action} {result.source_path}.")
        elif kind == "acquisition_preview":
            if result.converted_for_preview:
                self._set_dirty(True)
                self.workflow_changed.emit(self._require_workflow())
            self.acquisition_preview_loaded.emit(result)
            self.activity.emit("INFO", f"Loaded detector preview for {result.run_key.short()}.")
        elif kind == "save_masks":
            config_id, path = result
            self._set_dirty(True)
            self.masks_changed.emit(config_id)
            self.activity.emit("INFO", f"Saved and attached mask bundle for {config_id}: {path}.")
        elif kind in {"attach_mask", "find_mask"}:
            config_id, path = result
            self._set_dirty(True)
            self.masks_changed.emit(config_id)
            action = "Attached" if kind == "attach_mask" else "Found and attached"
            self.activity.emit("INFO", f"{action} mask bundle for {config_id}: {path}.")
        elif kind == "flatfield":
            config_id, path = result
            self._set_dirty(True)
            self.flatfield_changed.emit(config_id)
            self.activity.emit("INFO", f"Created water flatfield for {config_id}: {path}.")
        elif kind == "attach_flatfield":
            config_id, path = result
            self._set_dirty(True)
            self.flatfield_changed.emit(config_id)
            self.activity.emit("INFO", f"Attached flatfield for {config_id}: {path}.")
        elif kind in {"reduce_selected", "reduce_pending"}:
            processed_count, failed_count = result
            self._set_dirty(True)
            self.processing_changed.emit()
            if failed_count:
                self.activity.emit(
                    "WARNING",
                    f"Processed {processed_count} acquisition(s); {failed_count} failed."
                )
            else:
                self.activity.emit("INFO", f"Processed {processed_count} acquisition(s).")
        elif kind in {"stitch_selected", "stitch_pending"}:
            stitched_count, failed_count = result
            self._set_dirty(True)
            self.processing_changed.emit()
            level = "WARNING" if failed_count else "INFO"
            self.activity.emit(
                level,
                f"Stitched {stitched_count} sample(s); {failed_count} failed.",
            )

    @QtCore.Slot(str, object)
    def _on_job_failed(self, kind: str, failure: object) -> None:
        if isinstance(failure, JobFailure):
            message = failure.message
            details = failure.traceback_text
        else:
            message = str(failure)
            details = message
        self.activity.emit("ERROR", f"{kind}: {message}")
        self.failed.emit(message, details)

    @QtCore.Slot(str)
    def _on_job_finished(self, kind: str) -> None:
        self.busy_changed.emit(False, "")
        if kind == "acquisition_preview" and self._pending_preview_run_key is not None:
            run_key = self._pending_preview_run_key
            self._pending_preview_run_key = None
            QtCore.QTimer.singleShot(0, lambda: self.load_acquisition_preview(run_key))
