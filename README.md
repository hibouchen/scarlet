# SCARLET

**SCA**ttering **R**eduction and ana**L**ysis **E**nvironmen**T**

SCARLET is a NeXus-native framework for SANS data workflows.

At the current stage, SCARLET mainly provides:

- conversion of selected instrument files to a common SCARLET `NXsas_raw` profile;
- YAML-driven validation of SCARLET NeXus/HDF5 files;
- extraction and comparison of instrument configurations;
- a `WorkflowContext` API to organize converted runs, masks, beam centers, ROIs, transmissions, reference files, and flatfields;
- low-level reduction helpers for dead-time correction, monitor normalization, transmission calculation, geometry, `Q`-resolution helpers, reference subtraction, and azimuthal averaging;
- GUI tools for mask editing and NeXus inspection.

Some workflow pieces around `refs_*`, flatfields, and 2D reduction are present in the codebase, but the high-level public reduction interface is still evolving.

---

## Installation for development

From the project root:

```bash
pip install -e .
```

For notebooks:

```bash
pip install -e .[notebook]
```

For tests:

```bash
pip install -e .[dev]
pytest -q
```

If the package is not installed in editable mode, tests can also be run with:

```bash
PYTHONPATH=src pytest -q
```

### GUI note

The mask editor currently uses `tkinter`.

- `tkinter` is not a pip dependency managed through `pyproject.toml`
- it is usually provided by the Python installation or by a system package
- on some Linux distributions you may need to install `python3-tk`

For the most portable setup, the core package, CLI, and notebooks should be considered the primary interfaces; the `tkinter` GUI remains optional.

---

## Geometry convention

SCARLET uses a sample-centred coordinate system:

- origin at the sample centre;
- `+z` downstream, along the direct beam;
- `+x` horizontal, beam-right;
- `+y` vertical up;
- distances in metres unless stated otherwise.

Instrument components are described with NeXus groups and, where possible, `NXtransformations`.

---

## Current architecture

```text
src/scarlet/
  cli.py                         # command-line utilities
  io/
    converters/                  # instrument -> SCARLET NXsas_raw converters
      sam.py
      sansllb.py
  schemas/                       # packaged YAML schemas and schema notes
  validation/                    # YAML schema loading and HDF5 validation
  reduction/                     # low-level reduction helpers
  workflow/
    configuration.py             # configuration extraction/comparison
    context.py                   # workflow state container
```

The main schema currently used for converted monochromatic SANS files is:

```text
scarlet_nxsas_raw_v1.3_mono.yaml
```

Additional packaged schemas include:

```text
scarlet_masks_v1.0.yaml
scarlet_refs_sub_v1.0.yaml
scarlet_refs_norm_v1.0.yaml
scarlet_workflow_context_v1.0.yaml
```

---

## Command-line interface

List available schemas:

```bash
scarlet schema list
```

Validate a converted SCARLET raw file:

```bash
scarlet validate data/SANSLLB/processed/run_001.nxs \
  --schema scarlet_nxsas_raw_v1.3_mono.yaml
```

Validate a reference bundle:

```bash
scarlet validate data/SANSLLB/processed/refs_sub_config_1.nxs \
  --schema scarlet_refs_sub_v1.0.yaml
```

List known converters:

```bash
scarlet convert list
```

Convert an instrument file to SCARLET `NXsas_raw`:

```bash
scarlet convert sansllb data/SANSLLB/raw/run_001.nxs \
  data/SANSLLB/processed/run_001_scarlet.nxs \
  --overwrite \
  --validate
```

The currently registered converters are:

- `sansllb` with aliases `SANSLLB`, `sans-llb`, `sans_llb`;
- `sam`.

Export one azimuthal detector curve from a reduced file:

```bash
scarlet azimuthal-average \
  data/SANSLLB/processed/reduced_2d/sample_reduced_2d.nxs \
  data/SANSLLB/processed/azimuthal_average/sample_iq.csv \
  --overwrite
```

Open the mask editor GUI:

```bash
scarlet mask-gui data/SANSLLB/processed/sample_scattering.nxs
```

Open the NeXus viewer:

```bash
scarlet nxsas-gui data/SANSLLB/processed
```

Open the silx-based viewer:

```bash
scarlet viewer data/SANSLLB/processed --instrument sansllb
```

Notes on the current CLI state:

- `scarlet azimuthal-average` currently exports one detector at a time from an existing reduced file.
- The parser also exposes `scarlet reduce-2d`, but that path should still be treated as in-progress until the underlying public reduction API is finalized.
- The repository ships `refs_*` schemas and workflow helpers, but it does not currently expose public `scarlet refs-sub ...` or `scarlet refs-norm ...` commands.

---

## Python API currently available

### Convert files

```python
from scarlet.io.converters import convert_to_scarlet_nxsas_raw

report = convert_to_scarlet_nxsas_raw(
    "sansllb",
    "data/SANSLLB/raw/run_001.nxs",
    "data/SANSLLB/processed/run_001_scarlet.nxs",
    overwrite=True,
)

print(report.warnings)
```

### Validate files

```python
from pathlib import Path

from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import validate_nexus_file

schema = load_schema("scarlet_nxsas_raw_v1.3_mono.yaml")
report = validate_nexus_file(Path("data/SANSLLB/processed/run_001_scarlet.nxs"), schema)

for line in report.format_lines():
    print(line)
```

### Extract and compare configurations

```python
from scarlet.workflow.configuration import (
    configuration_from_nexus,
    compare_configurations,
)

cfg_a, issues_a = configuration_from_nexus("run_a_scarlet.nxs")
cfg_b, issues_b = configuration_from_nexus("run_b_scarlet.nxs")

same, diffs = compare_configurations(cfg_a, cfg_b)
```

### Initialize a workflow context from a raw directory

```python
from scarlet.workflow import initialize_workflow_context_from_raw_directory

ctx = initialize_workflow_context_from_raw_directory(
    "data/SANSLLB/raw",
    output_dir="data/SANSLLB/processed",
    instrument_name="sansllb",
)

print(len(ctx.runs))
print(len(ctx.configurations))
print(ctx.runs_table())
```

### Attach a mask bundle and prepare a flatfield

```python
from scarlet.workflow import WorkflowContext

ctx = WorkflowContext(output_dir="data/SANSLLB/processed")
config_id = ctx.attach_mask_bundle("data/SANSLLB/processed/config_1_masks.nxs")
print(config_id)

# When the corresponding water / dark / empty-cell references are registered
# in the workflow context, a flatfield artifact can be built on demand.
# flatfield_path = ctx.build_water_flatfield(config_id)
```

### Compute a transmission from two images

```python
from scarlet.reduction import compute_transmission

transmission = compute_transmission(
    "sample_transmission.nxs",
    "empty_beam_transmission.nxs",
    roi=(100, 140, 95, 135),
)

print(transmission)
```

### Compute an azimuthal average from arrays

```python
import numpy as np

from scarlet.reduction import azimuthal_average, compute_q_norm_map

image = np.random.random((128, 128))
q_map = compute_q_norm_map(
    image,
    beam_center=(63.5, 63.5),
    detector_distance=4.2,
    pixel_size=(0.001, 0.001),
    wavelength=6.0,
)
iq = azimuthal_average(
    image,
    q_map,
    n_bins=200,
)

print(iq.q.shape)
print(iq.intensity.shape)
```

Implemented operations currently cover conversion, validation, workflow bookkeeping, mask-bundle attachment, flatfield preparation helpers, detector dead-time correction, monitor normalization, transmission calculation, geometry / `Q` helpers, and azimuthal regrouping to `I(Q)`.

---

## Local data convention

There is no mandatory data folder layout, but the examples use:

```text
data/<instrument>/raw/        # original instrument files
data/<instrument>/processed/  # converted SCARLET files and generated outputs
```

The `data/` directory is ignored by git so that raw and processed experimental files are not committed accidentally.

---

## Roadmap

Near-term priorities:

1. stabilize the higher-level workflow around references, flatfields, and reduced outputs;
2. align the public CLI and Python API with the currently implemented reduction pieces;
3. add formal uncertainty propagation to the correction chain;
4. add multi-distance stitching and Q-resolution handling.

Longer-term goals:

- multi-distance stitching;
- Q-resolution handling;
- TOF-aware workflows;
- AI-assisted masking and quality checks;
- instrument-agnostic validated release.

---

## Why SCARLET?

- NeXus-native;
- explicit geometry conventions;
- reproducible metadata and validation;
- designed for multi-configuration SANS workflows;
- prepared for modern reactor, TOF and compact-source instruments.
