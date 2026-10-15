# Graphical interface design

This document defines the first implementation plan for the SCARLET desktop
application. It deliberately starts from the existing workflow API and GUI
helpers instead of introducing a second reduction model.

## Product goal

The application should let an instrument scientist reduce a SANS experiment
without writing Python while keeping every decision visible and reproducible.
The main user journey is:

1. create or open a workflow;
2. review the automatically detected acquisitions;
3. correct metadata and exclude invalid acquisitions;
4. run reduction and stitching;
5. inspect outputs and processing logs.

The first supported instruments are SAM and SANSLLB. The initial release is a
desktop application based on PySide6 and silx, consistent with the existing
viewer and installed project dependencies.

The MVP does not aim to replace every advanced notebook workflow. Batch
comparison tools, interactive fitting, publication layout, remote execution,
and a plugin system are outside its initial scope.

## Existing foundations

| Existing component | Reuse in the application | Current gap |
| --- | --- | --- |
| `WorkflowContext` | Canonical runs, configuration, exclusions, masks, logs, persistence, and status | No direct API yet for replacing one run key after an inline table edit |
| `ReductionPipeline` | Reduction of one exact run with `run_for_runkey()` and incremental processing with `run_new()` | No progress or cancellation callback |
| `StichingPipeline` | Per-sample and incremental stitching | The public class currently keeps the historical `Stiching` spelling |
| `nxsas_viewer.py` | File discovery, temporary conversion, NeXus inspection, and dataset loading | Helpers are not exposed as reusable widgets |
| `mask_editor.py` | Detector loading and mask-bundle writing | Mask interaction currently lives in the viewer window |
| `scarlet_viewer.py` | Proven silx `Plot2D` display and mask-selection behavior | Its detector tab is reused by the acquisition mask editor; the full file browser remains standalone |
| `tutorial.py` | Small PySide6 launcher example | It uses a separate Qt loading path and is not a workflow application |

The existing viewer remains useful during development. The workflow GUI should
extract reusable widgets from it progressively; it should not rewrite all of
the visualization code before the first useful workflow is available.

## MVP information architecture

One main window contains six pages and a persistent activity area:

- **Experiment** creates a workflow from a raw directory or opens a saved
  workflow context.
- **Acquisitions** reviews detection results, validation issues, exclusions,
  processing status, and per-file detector masks.
- **Settings** verifies beam centers and transmission ROIs, assigns detector
  masks, declares transmission reuse, and creates or assigns flatfields for
  each configuration.
- **Processing** starts reduction, configures detector-specific azimuthal bins
  and q spacing, and reports progress.
- **Stitching** lists the merged text outputs found in the output directory
  and merges selected ready samples with a common reference configuration and
  a multiplicative normalization factor.
- **Results** overlays selected reduced or stitched I(Q) text curves using
  silx and provides filename/type search and table sorting. Detector-image
  review is a later extension.
- **Activity** is a dock containing workflow logs, task messages, and errors.

Navigation is gated by available state, not by a rigid wizard. A user can
return to the acquisition table, make a correction, and process the affected
run again.

### Main window

```text
+--------------------------------------------------------------------------+
| SCARLET [Experiment] [Acquisitions] [Settings] [Processing] [Stitching] [Results] |
+----------------+---------------------------------------------------------+
| Workflow       | Current page                                            |
| experiment-42  |                                                         |
| SANSLLB        |                                                         |
| 37 acquisitions|                                                         |
| 12 / 18 done   |                                                         |
|                |                                                         |
| [Save]         |                                                         |
+----------------+---------------------------------------------------------+
| Activity:  info  warning  error                         [clear] [details] |
+--------------------------------------------------------------------------+
```

### Experiment page

```text
+--------------------------------------------------------------------------+
| Start a workflow                                                         |
|                                                                          |
| Raw data       [/data/SANSLLB/raw_june                   ] [Browse]       |
| Output         [/data/SANSLLB/processed                  ] [Browse]       |
| Instrument     [SANSLLB v]                                               |
| Experiment ID  [june_campaign                            ]                |
| Transmission   [opaque beamstop v]                                       |
|                                                                          |
|                                      [Open workflow] [Scan acquisitions] |
+--------------------------------------------------------------------------+
```

The transmission strategy is a workflow setting. Choosing
`semi_transparent_beamstop` must not force acquisition modes; mode detection
and user corrections remain independent.

### Acquisitions page

```text
+--------------------------------------------------------------------------+
| [Refresh folder] [Import CSV] [Export CSV]   Filter [____________]       |
| [ ] Show excluded   [ ] Show only warnings                              |
+--------------------------------------------------------------------------+
| Use | Sample | Config | Mode         | Entity     | Status      | File   |
|  x  | sample | cfg-1  | scattering   | sample     | missing     | ...hdf |
|  x  | sample | cfg-1  | transmission | sample     | missing     | ...hdf |
|  !  |        | cfg-1  | scattering   | empty_beam | processed   | ...hdf |
+--------------------------------------------------------------------------+
| Selection: detected from intensity distribution                          |
| Warning: acquisition mode confidence is low                 [Open image] |
+--------------------------------------------------------------------------+
| Detector preview and mask                                                 |
| [Detector 0] [Detector 1]                                      [Log v]   |
|                 silx Plot2D with selection-mask tools                    |
+--------------------------------------------------------------------------+
```

Editable cells use constrained editors for sample name, mode, entity, and
strictly positive optional thickness. Configuration identifiers stay
read-only. Rows can be searched, sorted, and filtered by status, mode, entity,
configuration, or uncertain detection. Shift/Ctrl selection allows several
rows to be excluded together without requesting a reason; SCARLET records an
automatic audit event. CSV import and export remain available for large edits.
The page also reports whether the required sample, transmission, empty-beam,
and water/flatfield inputs are ready for the default processing pipeline.
Selecting a row loads its detector images in the same silx mask editor used by
the standalone viewer, positioned to the right of the acquisition table.
Saving writes a `SCARLET_masks` bundle and registers it for that row's
configuration, invalidating its flatfield so it can be rebuilt.

### Settings page

The selected configuration remains summarized above the editor. The editor is
split into four tabs: **Beam center and ROI** for the empty-beam image and
numeric geometry controls, **Mask** for the attached-mask status and a link
back to the Acquisitions editor, manual mask-bundle attachment, or compatible
mask discovery in the output folder; **Flatfield** for creating, attaching,
or reusing a flatfield from another configuration; and **Transmission** for
declaring that the selected configuration reuses transmissions (and the
transmission ROI) from another active configuration. Changing the
transmission source invalidates the selected configuration's flatfield.

### Processing, stitching, and results pages

```text
+--------------------------------+-----------------------------------------+
| Processing                     | Stitching                               |
|                                |                                         |
| Sample runs and status table   | Ready sample groups table               |
|                                |                                         |
| detector0 bins [120]           | Reference [automatic v]                 |
| detector1 bins [ 30]           | Normalization factor [1.0]              |
| q scale        [linear v]      |                                         |
|                                |                                         |
| [Process selected] [pending]   | [Stitch selected] [pending]             |
+--------------------------------+-----------------------------------------+
```

Long operations must not freeze the window. While a job is running, the UI
shows the current acquisition, successful and failed counts, and a cancel
request when the operation supports a safe boundary between runs.

The **Settings** application menu opens a dedicated stitching-options dialog.
It exposes every parameter forwarded to the greedy stitching algorithm,
excluding the reference configuration and normalization factor, which remain
specific to an individual stitching run. The selected values are saved with
the workflow context for reproducibility.

## Application architecture

The UI is a client of the workflow domain model. Scientific decisions stay in
`scarlet.workflow`; widgets only present state and dispatch commands.

```text
ScarletApplication
└── ScarletMainWindow
    ├── WorkflowController ─────────────── WorkflowContext
    ├── ExperimentPage                    save/load/refresh/edit APIs
    ├── AcquisitionsPage
    │   └── RunsTableModel
    ├── SettingsPage
    ├── ProcessingPage ─── JobController ─ ReductionPipeline
    ├── StitchingPage ──── JobController ─ StichingPipeline
    ├── ResultsPage
    │   ├── DetectorView (silx Plot2D)
    │   └── CurveView
    └── ActivityDock ───────────────────── workflow logs and task events
```

The proposed source layout is:

```text
src/scarlet/gui/
├── application.py       # application creation and command entry point
├── main_window.py       # navigation, actions, and page composition
├── workflow_controller.py
├── jobs.py              # background job protocol and Qt implementation
├── models/
│   └── runs_table.py    # QAbstractTableModel adapter
├── pages/
│   ├── experiment.py
│   ├── acquisitions.py
│   ├── settings.py
│   ├── processing.py
│   └── results.py
└── widgets/
    ├── detector_mask_editor.py
    ├── curve_view.py
    └── activity_log.py
```

### State ownership

`WorkflowContext` is the source of truth. A controller owns the current
context and the path of its saved file. Pages receive immutable view data or
table adapters and never keep an independent copy of scientific metadata.

The workflow API should gain a validated operation for changing one registered
run, for example `replace_run(old_key, new_key, file_path)`. The operation must
preserve duplicate-key allocation, synchronize reference stores, update
sample thickness/transmission metadata where relevant, and record the change.
The GUI must not reproduce these rules or implement inline editing by silently
round-tripping through a temporary CSV.

Saving is explicit in the first vertical slice. The title bar indicates a
dirty workflow, and closing a dirty workflow asks whether it should be saved.
Autosave can be considered after file-recovery behavior is specified.

### Background jobs

Directory scans, raw conversion, reduction, stitching, and non-trivial file
loading run outside the GUI thread. `JobController` executes domain calls in a
Qt worker thread and exposes signals for:

- job start and completion;
- current run or sample;
- progress counters;
- structured failure information;
- cancellation requested at a safe run boundary.

The first reduction adapter can enumerate pending `RunKey` objects and invoke
`ReductionPipeline.run_for_runkey()` one by one. This gives useful progress and
safe cancellation without changing numerical steps. A later domain-level
progress callback can replace that adapter if command-line clients also need
fine-grained progress.

Exceptions are caught at the worker boundary, recorded in the workflow when
appropriate, and rendered in the activity dock. The GUI thread only updates
widgets in response to signals.

## Delivery plan

### Milestone 1: application shell and workflow setup

- add a `scarlet gui` command while keeping `scarlet viewer` available;
- create the main window, navigation, and activity dock;
- create a workflow from a raw directory in a background job;
- open and save a `WorkflowContext` file;
- display a read-only acquisitions table.

This is the first vertical slice. It is complete when a user can launch the
application, scan SAM or SANSLLB data, inspect the detected acquisitions, save
the workflow, close the application, and reopen the same workflow.

### Milestone 2: acquisition review

- add validated inline editing through a workflow-domain mutation API (mode
  and entity editing are available, as are sample renaming and optional
  strictly positive sample thickness editing; configuration stays read-only);
- add multi-row exclusion with an automatic audit reason and reinstate excluded
  files through a deliberate rescan;
- show processing state, uncertain mode-detection warnings, and processing
  readiness diagnostics;
- keep CSV import/export, search/filter controls, sorting, and folder refresh
  available.

### Milestone 3: settings

- summarize preparation state for every configuration (implemented, including
  selected-configuration details; configurations represented only by excluded
  files are omitted);
- verify and edit beam centers and transmission ROIs on the relevant
  empty-beam images (implemented with graphical, numerical, and automatic
  detection tools whose results require explicit confirmation);
- assign existing masks or create them with the mask editor;
- create, assign, or reuse water flatfields;
- gate reduction on blocking preparation issues.

### Milestone 4: reduction

- display sample-scattering candidates with pending, processed, or unavailable
  state and support multiple-row selection (implemented as the first slice);
- process a selection or all new sample-scattering runs;
- report per-run progress and failures;
- refresh statuses without rescanning the raw directory;
- preserve a responsive interface during conversion and reduction.

### Milestone 5: results and stitching

- extract reusable detector display from the existing viewer;
- display azimuthal curves and merged curves;
- stitch ready samples with a selectable reference configuration;
- expose output files and exports.

### Milestone 5: masks and refinement

- integrate mask creation and mask-bundle attachment;
- add configuration and transmission diagnostics;
- refine keyboard navigation, accessibility, recovery, and packaging;
- add optional autosave once recovery semantics are tested.

## Verification strategy

The GUI should remain testable without a display server wherever possible:

- test workflow mutations and job selection as pure Python behavior;
- test table-model roles, validation, and edits with a Qt application fixture;
- mock dialogs and domain services in page/controller tests;
- reuse small NeXus fixtures for application-level workflow tests;
- keep numerical pipeline tests independent from Qt;
- add one offscreen smoke test for creating and closing the main window.

Each milestone should include a manual check with representative SAM and
SANSLLB folders, including uncertain SANSLLB acquisition-mode detections and a
semi-transparent beamstop workflow.

## Decisions for the first implementation

The proposed defaults are:

- SAM and SANSLLB have equal priority in the application shell;
- the interface language is English, consistent with the existing package and
  documentation;
- workflow saving is explicit, with dirty-state protection;
- acquisition metadata is editable before reduction;
- the current viewer stays available as a separate command until its reusable
  widgets cover the same features.

These choices can be revised without changing the domain architecture. The
next implementation step is Milestone 1 only.
