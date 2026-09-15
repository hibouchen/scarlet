# SCARLET

**SCA**ttering **R**eduction and ana**L**ysis **E**nvironmen**T** is a Python framework for SANS data-reduction workflows. It uses NeXus/HDF5 as its primary exchange format and provides command-line tools, a Python API, and graphical inspection interfaces.

## What SCARLET does

- Convert instrument files to the SCARLET `NXsas_raw` profile.
- Validate NeXus/HDF5 files against the supplied schemas.
- Prepare and inspect detector masks.
- Apply reduction building blocks, including azimuthal integration.
- Export an `I(Q)` curve from a reduced output.
- Work in a Jupyter notebook using the supplied tutorial.

## Quick start

```bash
git clone https://github.com/hibouchen/scarlet.git
cd scarlet
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
scarlet --help
```

Continue with [Installation](installation.md), then the [tutorial](tutorial.md).

## Status

SCARLET is under active development. Some high-level reduction functionality is still evolving; review the [project changelog](https://github.com/hibouchen/scarlet/blob/main/CHANGELOG.md) before updating a production workflow.
