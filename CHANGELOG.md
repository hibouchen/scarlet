# Changelog

All notable changes to this project should be documented in this file.

The project follows a simple Semantic Versioning policy:

- `MAJOR` for incompatible public API changes;
- `MINOR` for backward-compatible feature additions;
- `PATCH` for backward-compatible fixes and documentation-only release corrections.

## [Unreleased]

- Documentation: README aligned with the currently exposed CLI and Python API.
- Project metadata: package version is now exposed as `scarlet.__version__`.
- Process: added a project changelog and a minimal release checklist.

## [0.1.0] - 2026-07-07

Initial tagged project version for the current SCARLET packaging layout.

- Packaged Python project with `pyproject.toml`.
- CLI entry points for schema inspection, validation, conversion, reduction-related utilities, and viewers.
- Schema packaging for SCARLET NeXus conventions.
- Workflow context, mask handling, reference helpers, and flatfield preparation code present in the distribution.
- Low-level reduction helpers for monitor normalization, dead-time correction, transmission, geometry, resolution, subtraction, and azimuthal averaging.
