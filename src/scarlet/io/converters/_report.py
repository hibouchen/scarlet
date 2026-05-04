from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConvertReport:
    """Standard conversion result shared by apparatus-specific converters."""

    input_file: Path
    output_file: Path
    entry_in: str
    notes: list[str]
    warnings: list[str]
