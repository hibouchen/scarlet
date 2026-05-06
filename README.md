# SCARLET

**SCA**ttering **R**eduction and ana**L**ysis **E**nvironmen**T**

SCARLET is a NeXus-native framework for SANS data workflows.

At the current stage, SCARLET provides the **infrastructure layer** needed before a full deterministic reduction pipeline:

- conversion of selected instrument files to a common SCARLET `NXsas_raw` profile;
- YAML-driven validation of SCARLET NeXus/HDF5 files;
- extraction and comparison of instrument configurations;
- generation of subtraction-reference bundles, `SCARLET_refs_sub`;
- generation of normalization-reference bundles, `SCARLET_refs_norm`, including water references that may come from another configuration.

The high-level physical reduction API, for example `sc.load(...)`, `sc.Reduction(...)`, and `scarlet reduce`, is **planned but not implemented yet**.

---

## Installation for development

From the project root:

```bash
pip install -e .
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
  workflow/
    configuration.py             # configuration extraction/comparison + refs files
    context.py                   # early workflow state container
```

The main schema currently used for converted monochromatic SANS files is:

```text
scarlet_nxsas_raw_v1.3_mono.yaml
```

Additional packaged schemas include:

```text
scarlet_refs_sub_v1.0.yaml
scarlet_refs_norm_v1.0.yaml
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

Generate subtraction-reference bundles from the run-configuration Excel file:

```bash
scarlet refs-sub from-excel \
  data/SANSLLB/processed/run_configuration.xlsx \
  data/SANSLLB/processed \
  data/SANSLLB/processed \
  --overwrite \
  --validate
```

Generate normalization-reference bundles, including water references:

```bash
scarlet refs-norm from-excel \
  data/SANSLLB/processed/run_configuration.xlsx \
  data/SANSLLB/processed \
  data/SANSLLB/processed \
  --overwrite \
  --validate
```

The normalization generator prefers water files measured in the same configuration. If no local water reference is available, it can borrow a water file from another configuration and records the corresponding `source_config_id` in the output file.

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

### Generate reference bundles from Excel

```python
from scarlet.workflow.configuration import (
    write_refs_sub_files_from_excel,
    write_refs_norm_files_from_excel,
)

write_refs_sub_files_from_excel(
    "run_configuration.xlsx",
    data_dir="data/SANSLLB/processed",
    output_dir="data/SANSLLB/processed",
    overwrite=True,
)

write_refs_norm_files_from_excel(
    "run_configuration.xlsx",
    data_dir="data/SANSLLB/processed",
    output_dir="data/SANSLLB/processed",
    overwrite=True,
)
```

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

1. keep the documentation and CLI aligned with the implemented code;
2. stabilize the ROI convention used for transmission bundles;
3. add an end-to-end test: convert -> refs_sub -> refs_norm -> validate;
4. implement the first deterministic 2D correction pipeline;
5. implement azimuthal integration and export of reduced `I(Q)`.

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
