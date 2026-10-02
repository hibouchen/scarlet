# Workflow context

`WorkflowContext` is the registry and execution trace for a SCARLET reduction.
It links logical run identifiers to raw acquisition files, stores workflow
metadata, records excluded files, and reports which runs have been processed.

## Initialize from raw acquisitions

```python
from scarlet.workflow import initialize_workflow_context_from_raw_directory

workflow = initialize_workflow_context_from_raw_directory(
    "data/SANSLLB/raw",
    output_dir="data/SANSLLB/processed",
    instrument_name="sansllb",
)
```

Initialization inspects the raw files to identify samples, references, modes,
and instrument configurations. The paths registered in `workflow.runs` remain
the paths of the original raw files, such as `.hdf` acquisitions.

No conversion to the SCARLET NeXus format is performed during initialization.
Conversion occurs only when a run is requested by the reduction pipeline. The
converted NeXus file is written under `output_dir` and then reused by later
pipeline calls.

```python
workflow.runs_table()
```

The returned `TableView` is rendered as an HTML table in Jupyter.

## Review and filter the run table

Export the detected runs when they need to be reviewed or filtered manually:

```python
csv_path = workflow.write_runs_table_csv(
    "notebooks/runs_filtered.csv",
    overwrite=True,
)
```

Edit the CSV, keeping `file_path` pointed at the raw acquisitions rather than
at pre-converted `.nxs` files, and apply it to the context:

```python
workflow.update_from_runs_table_csv(csv_path)
```

Files removed from the CSV are recorded in the exclusion history with the
source `runs_table`.

## Refresh runs after new acquisitions arrive

Rescan the context's `root_dir` without rebuilding the context:

```python
workflow.refresh_runs()
```

The refresh operation:

- adds newly discovered raw acquisitions;
- does not convert them;
- does not duplicate already registered runs;
- preserves existing run entries, even if their files are no longer present;
- keeps files excluded manually or through the run-table CSV excluded.

Files previously rejected by the directory scan are reconsidered. This allows
an acquisition that was incomplete or invalid during an earlier scan to be
registered once it becomes readable.

To explicitly reconsider every valid raw file, including files excluded by the
CSV or manually:

```python
workflow.refresh_runs(include_excluded=True)
```

## Inspect excluded files

Display the files that are currently excluded:

```python
workflow.excluded_files_table()
```

Display the append-only history, including files that were later reinstated:

```python
workflow.excluded_files_table(include_history=True)
```

Each event contains:

- `action`: `excluded` or `reinstated`;
- `file_path`;
- `reason`;
- `source`, such as `directory_scan`, `runs_table`, or `manual`;
- the optional logical `run_key`;
- the UTC timestamp.

A caller can also record an explicit exclusion:

```python
workflow.exclude_file(
    "data/SANSLLB/raw/run_042.hdf",
    reason="Acquisition interrupted",
    source="manual",
)
```

Calling `add_run()` for an excluded file records a `reinstated` event. The
history is persisted in `/entry/excluded_files` when the context is saved.

## Run the pipeline

```python
from scarlet.workflow.pipeline import ReductionPipeline

pipeline = ReductionPipeline.default(workflow)
state = pipeline.run_for_sample(
    sample_name="sample_a",
    config_id="config_1",
)
```

The first pipeline access converts the requested raw run when necessary. Runs
that are never requested are never converted. Processed detector data are
written into the associated NeXus file under the top-level `/processed` entry.

The context is an attribute of the pipeline. It therefore does not need to be
passed again to `run_for_sample()` or `run_all()`.

To process only registered sample-scattering runs that do not yet contain
`/processed`:

```python
states = pipeline.run_new()
```

This method handles runs independently. If one run cannot be reduced, the
exception is recorded as a structured `ERROR` log on the workflow and the
pipeline continues with the remaining runs. The returned list contains only
the successful reduction states. Calling the method again retries failed runs
because their NeXus files still do not contain `/processed`.

`run_new()` does not rescan directories and does not add or remove workflow
runs. When new acquisitions must first be discovered, refresh the context
explicitly before invoking the pipeline:

```python
workflow.refresh_runs()
states = pipeline.run_new()
```

## Watch processing status

List only runs whose associated NeXus file contains `/processed`:

```python
workflow.processed_runs_table()
```

Display the status of every run:

```python
workflow.runs_status_table()
```

The possible states are:

| Status | Meaning |
| --- | --- |
| `processed` | The associated NeXus file contains `/processed`. |
| `unprocessed` | The NeXus file exists but does not contain `/processed`. |
| `missing` | The expected converted NeXus file does not exist yet. |
| `unreadable` | The associated file exists but cannot be opened as HDF5. |

Check one run directly:

```python
from scarlet.workflow import RunKey

key = RunKey(
    config_id="config_1",
    entity="sample",
    mode="scattering",
    sample_name="sample_a",
)

workflow.is_run_processed(key)
workflow.get_run_nexus_path(key)
```

These methods inspect the files each time they are called. They do not maintain
a background process and do not rely on a cached processing flag, so a notebook
view reflects the result of the latest completed pipeline execution.

## Stitch newly reduced samples

`StichingPipeline` follows the same incremental principle and uses the workflow
provided at construction time:

```python
from scarlet.workflow.pipeline import StichingPipeline

stitching = StichingPipeline(workflow)
merged = stitching.run_new(scale_on="config_1")
```

`run_new()` does not refresh the workflow and never converts raw acquisitions.
An eligible sample must have `/processed` in every registered scattering run.
The method skips a sample when its `<sample_name>_merged.txt` output is newer
than all its processed NeXus inputs. It runs the stitching again when a
processed input is newer than the existing merged output.

As with `ReductionPipeline.run_new()`, samples are handled independently. A
stitching failure is stored in the workflow log, and processing continues with
the remaining ready samples. The returned dictionary maps successfully
stitched sample names to their merged arrays.

## Save and restore the context

```python
from scarlet.workflow import WorkflowContext

workflow.save("notebooks/workflow.nxs", overwrite=True)
workflow = WorkflowContext.load("notebooks/workflow.nxs")
```

The saved context includes run registrations, configurations, references,
artifacts, logs, issues, timings, and the exclusion history. Raw and processed
scientific data remain in their own files.
