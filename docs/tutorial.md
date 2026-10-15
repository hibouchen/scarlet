# Jupyter tutorial

The bundled tutorial provides a hands-on introduction to SCARLET data and tools.

## Open the tutorial

From a SCARLET installation:

```bash
scarlet notebook
```

This command opens the graphical launcher and lets you create an editable copy of the tutorial notebook. A destination can also be provided for scripted use:

```bash
scarlet notebook tutorial_sessions
```

The following direct entry point is equivalent:

```bash
scarlet-notebook
```

## Working principles

Keep experimental data outside the repository and work on notebook copies. By convention, the project examples use this layout:

```text
data/<instrument>/raw/
data/<instrument>/processed/
```

The `data/` directory is ignored by Git to avoid versioning raw files and local reduction outputs.

## Start a workflow from raw files

```python
from scarlet.workflow import initialize_workflow_context_from_raw_directory

workflow = initialize_workflow_context_from_raw_directory(
    "data/SANSLLB/raw",
    output_dir="data/SANSLLB/processed",
    instrument_name="sansllb",
)

workflow.runs_table()
```

The context registers the original raw acquisition paths. It does not convert
all files during initialization; each required run is converted only when the
pipeline requests it.

If new acquisitions are added to the raw directory, update the registry with:

```python
workflow.refresh_runs()
```

Useful notebook views include:

```python
workflow.excluded_files_table()
workflow.processed_runs_table()
workflow.runs_status_table()
```

## Prepare absolute-calibration flatfields

Build the water flatfield for each configuration that supplies one:

```python
workflow.build_water_flatfield("config_1")
workflow.build_water_flatfield("config_2")
```

Each generated flatfield stores the attenuation-corrected direct flux and one
`normalisation_factor` per detector. Masked, non-finite, and non-positive
pixels are excluded from the detector mean. A configuration may reuse another
configuration's flatfield with `workflow.set_flatfield_source(...)`; the
pipeline compensates for the ratio between their direct fluxes.

Converted files store an optional `NXattenuator` under the instrument with
`selection` and `attenuator_transmission`. The transmission is the physical
ratio `I_attenuated / I_incident` and must be in `(0, 1]`. Old files without an
attenuator group are treated as unattenuated. An existing group without a
valid transmission cannot be used for direct-flux calibration.

Create a pipeline bound to the workflow and reduce only its registered,
unprocessed runs:

```python
from scarlet.workflow.pipeline import ReductionPipeline

pipeline = ReductionPipeline.default(
    workflow,
    n_bins={0: 120, 1: 30, 2: 30},
    q_scale="linear",
)
states = pipeline.run_new()
```

The default pipeline writes absolute intensities in `cm^-1`. Use one `n_bins`
value per detector, or pass a single integer for all detectors. Set `q_scale`
to `"log"` for logarithmic q-bin boundaries.

The method does not refresh the workflow. Call `workflow.refresh_runs()` first
when required. Failures are logged per run and do not stop the remaining
reductions.

Once the required runs have been reduced, stitch only ready samples without a
current merged output:

```python
from scarlet.workflow.pipeline import StichingPipeline

stitching = StichingPipeline(workflow)
merged = stitching.run_new(
    scale_on="config_1",
    normalization_factor=1.0,
)
```

The optional normalization factor multiplies the stitched `I` and `I_error`
values. It does not alter q or `q_error`.

See [Workflow context](workflow.md) for CSV filtering, exclusion history,
on-demand conversion, processing status, and context persistence.
