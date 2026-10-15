# SCARLET

<img src="logo_scarlet.png" alt="SCARLET logo" width="220">

**SCA**ttering **R**eduction and ana**L**ysis **E**nvironmen**T**

SCARLET is a NeXus-native framework for SANS data workflows. It provides command-line tools and Python APIs for converting raw instrument files, validating SCARLET NeXus/HDF5 files, inspecting data, preparing masks, and running reduction helpers.

Documentation: https://hibouchen.github.io/scarlet/

The high-level workflow around reduced outputs is still evolving.

## Python Workflow

Initialize a context directly from instrument raw files:

```python
from scarlet.workflow import initialize_workflow_context_from_raw_directory

workflow = initialize_workflow_context_from_raw_directory(
    "data/SANSLLB/raw",
    output_dir="data/SANSLLB/processed",
    instrument_name="sansllb",
)
```

Initialization records the original raw paths and does not eagerly convert the
directory. Conversion to SCARLET NeXus occurs only when the reduction pipeline
requests a run.

```python
workflow.refresh_runs()                 # Discover new acquisitions
workflow.excluded_files_table()         # Current exclusions
workflow.processed_runs_table()         # Runs containing /processed
workflow.runs_status_table()            # Status of every run
```

Create a reduction pipeline bound to this context and process only the
unprocessed sample runs already registered in the workflow:

```python
from scarlet.workflow.pipeline import ReductionPipeline

pipeline = ReductionPipeline.default(
    workflow,
    n_bins={0: 120, 1: 30, 2: 30},
    q_scale="linear",
)
states = pipeline.run_new()
```

The default pipeline produces absolute intensities in `cm^-1`. Its water
flatfield must contain the attenuation-corrected direct flux and one
`normalisation_factor` per detector. Flatfields built by the current workflow
store this calibration metadata automatically. `n_bins` may be a single
integer or a detector-number mapping; `q_scale` accepts `"linear"` or `"log"`.

One failing run is recorded in `workflow.logs` without interrupting the other
runs. `run_new()` does not refresh or change the workflow run registry. Call
`workflow.refresh_runs()` explicitly beforehand when new acquisitions must be
discovered. `pipeline.run_for_sample(...)` and `pipeline.run_all()` also use the
bound workflow and therefore do not take it as a method argument.

Incremental stitching uses the same workflow-owned API:

```python
from scarlet.workflow.pipeline import StichingPipeline

stitching = StichingPipeline(workflow)
merged = stitching.run_new(
    scale_on="config_1",
    normalization_factor=1.0,
)
```

Only samples whose registered scattering runs are all processed are eligible.
Existing merged outputs are reused until one of their processed NeXus inputs
becomes newer.

The detailed workflow, including editable CSV filtering and persistence, is
documented in the [Workflow context guide](https://hibouchen.github.io/scarlet/workflow/).

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

### Workflow application

Open the desktop application and optionally preselect a raw data directory:

```bash
scarlet gui data/SANSLLB/raw_june --instrument sansllb
```

The application can create, inspect, refresh, save, and reopen a workflow
context. Sample names, acquisition modes, entities, and positive optional
thicknesses can be corrected directly in the table; configuration identifiers
remain read-only. Rows can be searched, filtered, sorted from any column, and
excluded in a multi-row selection. SCARLET records an automatic exclusion
reason without prompting the user. CSV import/export supports larger edits,
and a readiness indicator reports missing inputs before processing. The
selected acquisition is also displayed below the table with the same detector
mask-selection tools as the standalone NeXus viewer. Saving there creates a
mask bundle in the workflow output folder and attaches it to the acquisition
configuration. The standalone viewer remains available for independent NeXus
inspection and mask editing.

The **Settings** step sits between acquisition review and processing. It
groups beam-center and transmission-ROI verification, mask assignment, and
flatfield preparation by configuration.
Its configuration overview already reports defined, missing, invalid, stale,
or unavailable inputs and shows their detailed values when a row is selected.
The beam-center editor uses the empty-beam transmission image, while the ROI
editor uses the empty beam selected by the transmission strategy. Both load in
the background. A center can be positioned by clicking the image and a ROI by
dragging a rectangle; both remain editable numerically before saving. Separate
automatic-detection buttons can propose a center or ROI without saving it
until the user confirms the result. The selected configuration offers
**Create flatfield** or **Rebuild flatfield** to prepare its water flatfield
in the background using the current settings and masks. The generated file
also stores the attenuation-corrected direct flux and the absolute-calibration
factor of each detector. An existing validated flatfield file can be attached,
or a configuration can reuse the flatfield of another active configuration.
The **Transmission** tab can likewise declare
that a configuration reuses transmissions from another active configuration;
its transmission ROI follows that source, and any locally built flatfield is
then invalidated.

The **Processing** page lists sample-scattering reduction candidates, their
pending/processed status, and the current readiness diagnostic. Select one or
more rows to reduce them, or use **Process pending** to reduce all currently
pending sample acquisitions. Reductions run in the background and the table is
updated when the job finishes. The page exposes a separate radial-bin count
for every detector present on the instrument and linear or logarithmic q-bin
spacing. The separate **Stitching** page merges every
processed configuration of a selected sample. Its control table identifies
already stitched samples from the presence of `<sample>_merged.txt` in the
output folder. Select one or more ready samples to stitch them; a reference
configuration common to every selection can be chosen. The per-run
**Normalization factor** multiplies the stitched intensity and its uncertainty;
it does not change q or its uncertainty.
Use **Settings → Stitching options…** to adjust and save the greedy-stitching
algorithm parameters used for the workflow.

The **Results** page discovers reduced and stitched I(Q) text files in the
workflow output folder. Its search field filters by filename or result type,
and column headers sort the visible files. Select multiple rows to overlay
their curves in the silx plot, including their available intensity
uncertainties.

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
