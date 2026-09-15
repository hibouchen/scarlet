# SCARLET

<img src="logo_scarlet.png" alt="SCARLET logo" width="220">

**SCA**ttering **R**eduction and ana**L**ysis **E**nvironmen**T**

SCARLET is a NeXus-native framework for SANS data workflows. It provides command-line tools and Python APIs for converting raw instrument files, validating SCARLET NeXus/HDF5 files, inspecting data, preparing masks, and running reduction helpers.

Documentation: https://hibouchen.github.io/scarlet/

The high-level workflow around reduced outputs is still evolving.

## Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

Clone the repository and install SCARLET:

```bash
git clone https://github.com/hibouchen/scarlet.git
cd scarlet
pip install -e .
```

Alternative isolated CLI installation with `pipx`:

```bash
pipx install git+https://github.com/hibouchen/scarlet.git
```

From a local clone:

```bash
pipx install .
```

This creates a dedicated environment for SCARLET and exposes the command-line
entry points (`scarlet`, `scarlet-notebook`, and `viewer`) on your PATH.

For development and tests:

```bash
pip install -e .[dev]
pytest -q
```

Main dependencies, including `jupyterlab`, `silx`, and `PySide6`, are installed with the base package.

## Command-Line Interface

SCARLET installs the main command:

```bash
scarlet --help
```

### Notebook

Open the SCARLET tutorial notebook launcher:

```bash
scarlet notebook
```

The graphical launcher lets you create a new notebook from the packaged tutorial or open an existing `.ipynb` file. A direct entry point is also available:

```bash
scarlet-notebook
```

For scripted use, pass a destination path:

```bash
scarlet notebook tutorial_sessions
```

### Viewer

Open the silx-based viewer for NeXus inspection, detector display, and mask export:

```bash
scarlet viewer data/SANSLLB/processed --instrument sansllb
```

Supported instruments for temporary viewer conversion:

- `sansllb`
- `sam`

### Schemas

List packaged validation schemas:

```bash
scarlet schema list
```

Common schemas include:

- `scarlet_nxsas_raw_v1.3_mono.yaml`
- `scarlet_masks_v1.0.yaml`
- `scarlet_workflow_context_v1.0.yaml`

### Validation

Validate a SCARLET NeXus/HDF5 file:

```bash
scarlet validate data/SANSLLB/processed/run_001_scarlet.nxs \
  --schema scarlet_nxsas_raw_v1.3_mono.yaml
```

Validate a mask bundle by selecting the corresponding schema:

```bash
scarlet validate data/SANSLLB/processed/config_1_masks.nxs \
  --schema scarlet_masks_v1.0.yaml
```

### Conversion

List available converters:

```bash
scarlet convert list
```

Convert a raw instrument file to SCARLET `NXsas_raw`:

```bash
scarlet convert sansllb data/SANSLLB/raw/run_001.nxs \
  data/SANSLLB/processed/run_001_scarlet.nxs \
  --overwrite \
  --validate
```

Registered converters:

- `d11` with alias `D11`
- `sansllb` with aliases `SANSLLB`, `sans-llb`, `sans_llb`
- `sam`

### Azimuthal Average

Export one azimuthal `I(Q)` curve from a reduced SCARLET file:

```bash
scarlet azimuthal-average reduced_2d.nxs iq.csv --overwrite
```

Useful options include:

- `--processed-entry`
- `--detector`
- `--q-min`
- `--q-max`

## Data Layout

There is no mandatory data folder layout. Examples in this repository generally use:

```text
data/<instrument>/raw/
data/<instrument>/processed/
```

The `data/` directory is ignored by git to avoid committing experimental raw and processed files.
