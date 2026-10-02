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

Create a pipeline bound to the workflow and reduce only new, unprocessed runs:

```python
from scarlet.workflow.pipeline import ReductionPipeline

pipeline = ReductionPipeline.default(workflow)
states = pipeline.refresh_and_run_new()
```

Failures are logged per run and do not stop the remaining reductions.

See [Workflow context](workflow.md) for CSV filtering, exclusion history,
on-demand conversion, processing status, and context persistence.
