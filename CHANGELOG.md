# Changelog

All notable changes to this project should be documented in this file.

The project follows a simple Semantic Versioning policy:

- `MAJOR` for incompatible public API changes;
- `MINOR` for backward-compatible feature additions;
- `PATCH` for backward-compatible fixes and documentation-only release corrections.

## [Unreleased]

## [0.2.3] - 2026-09-04

- Feature: added a graphical `scarlet notebook` launcher to create a notebook from the packaged tutorial or open an existing `.ipynb` file before starting JupyterLab.
- Packaging: added the `scarlet-notebook` console script and packaged the tutorial notebook with the Python distribution.
- Windows: updated `run_scarlet.cmd` to launch the graphical notebook selector.

## [0.2.2] - 2026-08-18

- SAM: improved raw-file conversion to populate sample name, thickness, detector distance, pixel sizes, and slit/aperture metadata from the source NeXus fields.
- SAM: aligned converted detector orientation with the raw beam-center coordinates and updated mask compatibility for the transposed detector images.
- Workflow: relaxed mode inference fallback for non-2D files and updated the reduction pipeline for SAM pinhole resolution handling.
- Documentation: refreshed the SAM tutorial notebook and generated workflow artifacts for the current reduction flow.

## [0.2.0] - 2026-07-24

- Feature: implemented curve stitching utilities for combining scattering curves across configurations.
- Workflow: integrated stitching support into the reduction pipeline.
- Documentation: README aligned with the currently exposed CLI and Python API.
- Project metadata: package version is now exposed as `scarlet.__version__`.
- Cleanup: removed deprecated stitching example scripts that were no longer maintained.
- Cleanup: removed outdated stitching notebooks that no longer reflected the current workflow.
- Maintenance: trimmed obsolete experimental and generated files to keep the repository focused on supported workflows.
- Process: added a project changelog and a minimal release checklist.

## [0.1.0] - 2026-07-07

Initial tagged project version for the current SCARLET packaging layout.

- Packaged Python project with `pyproject.toml`.
- CLI entry points for schema inspection, validation, conversion, reduction-related utilities, and viewers.
- Schema packaging for SCARLET NeXus conventions.
- Workflow context, mask handling, reference helpers, and flatfield preparation code present in the distribution.
- Low-level reduction helpers for monitor normalization, dead-time correction, transmission, geometry, resolution, subtraction, and azimuthal averaging.
