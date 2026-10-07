from __future__ import annotations

from dataclasses import dataclass

from scarlet.workflow import WorkflowContext

from .models import workflow_run_rows


@dataclass(frozen=True)
class WorkflowReadiness:
    """A lightweight, non-destructive check before starting data reduction."""

    sample_count: int
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.sample_count > 0 and not self.errors


def assess_workflow_readiness(workflow: WorkflowContext) -> WorkflowReadiness:
    """Report missing inputs that the default reduction pipeline will need."""
    sample_runs = [
        key
        for key in workflow.runs
        if key.entity == "sample" and key.mode == "scattering" and key.sample_name
    ]
    errors: list[str] = []
    warnings: list[str] = []
    if not sample_runs:
        errors.append("No sample scattering acquisition is available.")
        return WorkflowReadiness(sample_count=0, errors=tuple(errors))

    for key in sample_runs:
        if not workflow.runs[key].exists():
            errors.append(f"Sample scattering file is missing for {key.short()}.")

    unresolved_detections = sum(bool(row.review_warning) for row in workflow_run_rows(workflow))
    if unresolved_detections:
        warnings.append(
            f"{unresolved_detections} acquisition mode detection(s) still need review."
        )

    for sample_name, config_id in sorted(
        {(key.sample_name, key.config_id) for key in sample_runs if key.sample_name}
    ):
        if workflow.get_sample_thickness(sample_name, config_id) is None:
            warnings.append(f"Missing thickness for {sample_name} in {config_id}.")

    source_mode = workflow.get_transmission_source_mode()
    configurations = sorted({key.config_id for key in sample_runs})
    transmission_inputs: set[tuple[str, str]] = set()

    for key in sample_runs:
        if workflow.get_transmission(key.sample_name or "", key.config_id) is None:
            transmission_config = workflow.resolve_transmission_config(key.config_id)
            transmission_inputs.add((transmission_config, source_mode))
            if not _has_run(
                workflow,
                config_id=transmission_config,
                entity="sample",
                mode=source_mode,
                sample_name=key.sample_name,
            ):
                errors.append(
                    f"Missing {source_mode} acquisition for {key.sample_name} "
                    f"in {transmission_config}."
                )

    for config_id in configurations:
        flatfield = workflow.get_flatfield(config_id)
        if flatfield is not None and flatfield.exists():
            continue
        flatfield_config = workflow.resolve_flatfield_config(config_id)
        water_runs = [
            key
            for key, path in workflow.runs.items()
            if key.config_id == flatfield_config
            and key.entity == "water"
            and key.mode == "scattering"
            and key.sample_name
            and path.exists()
        ]
        if not water_runs:
            errors.append(
                f"Missing water scattering acquisition or prepared flatfield for {config_id}."
            )
            continue

        water_key = water_runs[0]
        if workflow.get_transmission(water_key.sample_name or "", flatfield_config) is None:
            transmission_config = workflow.resolve_transmission_config(flatfield_config)
            transmission_inputs.add((transmission_config, source_mode))
            if not _has_run(
                workflow,
                config_id=transmission_config,
                entity="water",
                mode=source_mode,
            ):
                errors.append(
                    f"Missing water {source_mode} acquisition for {transmission_config}."
                )

    for config_id, mode in sorted(transmission_inputs):
        if not _has_run(
            workflow,
            config_id=config_id,
            entity="empty_beam",
            mode=mode,
        ):
            errors.append(f"Missing empty-beam {mode} acquisition for {config_id}.")

    return WorkflowReadiness(
        sample_count=len(sample_runs),
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _has_run(
    workflow: WorkflowContext,
    *,
    config_id: str,
    entity: str,
    mode: str,
    sample_name: str | None = None,
) -> bool:
    return any(
        key.config_id == config_id
        and key.entity == entity
        and key.mode == mode
        and (sample_name is None or key.sample_name == sample_name)
        and path.exists()
        for key, path in workflow.runs.items()
    )
