from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from scarlet.reduction import correct_detector_dead_time


def correct_detector_data_for_deadtime(
    data: Any,
    *,
    acquisition_time: float | None,
    dead_time: float | None,
    detector_name: str,
    warnings: List[str],
) -> Tuple[np.ndarray, bool]:
    """Apply detector dead-time correction when the acquisition time is usable."""
    corrected = np.asarray(data, dtype=np.float64)

    if acquisition_time is None:
        warnings.append(
            f"{detector_name}: missing acquisition time; detector data left uncorrected for dead time."
        )
        return corrected, False

    acq_time = float(acquisition_time)
    if not np.isfinite(acq_time) or acq_time <= 0.0:
        warnings.append(
            f"{detector_name}: invalid acquisition time {acquisition_time!r}; detector data left uncorrected for dead time."
        )
        return corrected, False

    effective_dead_time = 0.0 if dead_time is None else float(dead_time)
    if not np.isfinite(effective_dead_time):
        effective_dead_time = 0.0

    return (
        correct_detector_dead_time(
            corrected,
            acq_time=acq_time,
            deadtime=effective_dead_time,
        ),
        True,
    )
