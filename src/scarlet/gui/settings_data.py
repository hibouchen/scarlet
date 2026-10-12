from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from scarlet.io import nexus_reader
from scarlet.reduction.transmission import compute_beam_center, compute_transmission_roi
from scarlet.workflow import WorkflowContext

SettingsImagePurpose = Literal["center", "roi"]


@dataclass(frozen=True)
class SettingsImagePayload:
    config_id: str
    source_config_id: str
    purpose: SettingsImagePurpose
    detector_number: int
    source_path: Path
    image: np.ndarray
    center: tuple[float, float] | None
    roi: tuple[int, int, int, int] | None


def load_settings_image(
    workflow: WorkflowContext,
    *,
    config_id: str,
    purpose: SettingsImagePurpose,
    detector_number: int,
) -> SettingsImagePayload:
    """Prepare and read the empty-beam image used by one Settings editor."""
    if purpose == "center":
        source_config_id = config_id
        # Keep this aligned with workflow initialization: beam centers are
        # defined from the empty-beam transmission acquisition.
        mode = "transmission"
    elif purpose == "roi":
        source_config_id = workflow.resolve_transmission_config(config_id)
        mode = workflow.get_transmission_source_mode()
    else:
        raise ValueError(f"Unsupported Settings image purpose: {purpose!r}")

    source_path = workflow.prepare_reference("empty_beam", mode, source_config_id)
    if source_path is None:
        raise ValueError(
            f"Missing empty-beam {mode} acquisition for configuration {source_config_id!r}"
        )
    detector = nexus_reader.read_detector_data(
        source_path,
        detector_number,
        normalize_by_monitor=True,
        correct_deadtime=True,
    )
    image = np.asarray(getattr(detector, "values", detector), dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(
            f"Detector{detector_number} image must be 2D, got shape {image.shape}"
        )
    return SettingsImagePayload(
        config_id=config_id,
        source_config_id=source_config_id,
        purpose=purpose,
        detector_number=detector_number,
        source_path=source_path,
        image=image,
        center=workflow.get_beam_center(config_id, detector_number),
        roi=workflow.get_roi(source_config_id),
    )


def detect_settings_geometry(
    workflow: WorkflowContext,
    *,
    config_id: str,
    purpose: SettingsImagePurpose,
    detector_number: int,
) -> SettingsImagePayload:
    """Load the relevant empty beam and automatically detect its overlay."""
    payload = load_settings_image(
        workflow,
        config_id=config_id,
        purpose=purpose,
        detector_number=detector_number,
    )
    if purpose == "center":
        center = compute_beam_center(
            payload.source_path,
            detector_number=detector_number,
        )
        return replace(payload, center=center)
    roi = compute_transmission_roi(
        payload.source_path,
        detector_number=detector_number,
    )
    return replace(payload, roi=roi)


def load_center_from_empty_beam_file(
    workflow: WorkflowContext,
    *,
    config_id: str,
    detector_number: int,
) -> SettingsImagePayload:
    """Load a detector's stored beam center from its empty-beam file."""
    payload = load_settings_image(
        workflow,
        config_id=config_id,
        purpose="center",
        detector_number=detector_number,
    )
    center = nexus_reader.read_detector_beam_center(
        payload.source_path,
        detector_number,
    )
    return replace(payload, center=center)
