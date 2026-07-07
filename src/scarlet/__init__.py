"""
SCARLET: SANS data reduction framework.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scarlet")
except PackageNotFoundError:
    # Fallback for source checkouts used without installed package metadata.
    __version__ = "0.1.0"

__all__ = ["__version__", "ValidationReport", "ValidationMessage", "validate_nexus_file"]

from .validation.schema_validator import ValidationMessage, ValidationReport, validate_nexus_file
