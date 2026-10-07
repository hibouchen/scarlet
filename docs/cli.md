# Command line

The main command is `scarlet`.

```bash
scarlet --help
```

## Schemas

List the bundled schemas:

```bash
scarlet schema list
```

Validate a NeXus/HDF5 file:

```bash
scarlet validate data/SANSLLB/processed/run_001_scarlet.nxs \
  --schema scarlet_nxsas_raw_v1.3_mono.yaml
```

The `--strict` option treats warnings as errors. `--entry` explicitly selects the NeXus entry to validate.

## Conversion

List available converters:

```bash
scarlet convert list
```

Convert a raw file and validate the result:

```bash
scarlet convert sansllb data/SANSLLB/raw/run_001.nxs \
  data/SANSLLB/processed/run_001_scarlet.nxs \
  --overwrite \
  --validate
```

Registered converters include `d11`, `sansllb`, and `sam`.

## Azimuthal integration

Export an `I(Q)` curve from a processed output:

```bash
scarlet azimuthal-average reduced_2d.nxs iq.csv --overwrite
```

Use `--processed-entry`, `--detector`, `--q-min`, and `--q-max` to select the data to export.

## Viewer

Open the NeXus and mask viewer:

```bash
scarlet viewer data/SANSLLB/processed --instrument sansllb
```

For temporary conversions performed by the viewer, the supported instruments are `sansllb` and `sam`.

## Workflow application

Open the SCARLET desktop application:

```bash
scarlet gui
```

Preselect a raw acquisition directory and instrument:

```bash
scarlet gui data/SANSLLB/raw_june --instrument sansllb
```

Open a previously saved workflow context:

```bash
scarlet gui --workflow data/SANSLLB/processed/workflow.nxs
```

The application supports creating, opening, refreshing, and saving workflows
and reviewing detected acquisitions. Sample names, acquisition modes, and
entities can be corrected in the table; configuration identifiers remain
read-only. Uncertain mode detections are highlighted for review. Use `Shift`
or `Ctrl` to select several rows, then exclude the complete selection without
an additional dialog. SCARLET records an automatic audit reason. The
**Reconsider excluded** action deliberately scans excluded files again.
Clicking a column header sorts the rows; clicking it again reverses the order.
Search and filters narrow the table without changing the workflow. Sample and
water thicknesses can be edited in millimeters; the value must be strictly
positive or empty. CSV import/export handles larger edits, and the readiness
indicator lists missing inputs needed by the default processing pipeline.
Selecting an acquisition also opens its detector image below the table. The
same silx selection tools as the standalone viewer can be used to draw a
detector mask; **Save and attach mask** proposes
`<configuration>_masks.nxs` in the workflow output folder, lets you choose a
different filename, then attaches the saved bundle to the selected
configuration. When the acquisition time is available, this preview displays
corrected counts (the stored rate multiplied by `count_time`) to make counting
statistics easier to assess.
The **Settings** step precedes processing and groups configuration-level beam
centers, transmission ROIs, detector masks, and flatfields. Its overview table
shows the preparation state of every configuration, with exact values and file
sources in the selected-row details. Its visual editor loads the transmission
empty beam for detector centers and the strategy-dependent empty beam for the
transmission ROI. Click to move a center or drag to draw a ROI, then save the
validated coordinates. The automatic center and ROI buttons calculate a
proposal in the background and leave the final save under user control. The
selected configuration also provides **Create flatfield** or **Rebuild
flatfield**. It prepares the water flatfield in the background, using the
current centers, ROI, and detector masks. **Choose flatfield file** validates
and attaches an existing `SCARLET_flatfield` bundle; the flatfield-source menu
can instead reuse the flatfield of another active configuration.
The **Transmission** tab can declare that a configuration reuses the
transmissions of another active configuration. Its transmission ROI then
follows the selected source and any locally built flatfield is invalidated.
The **Mask** tab similarly provides **Choose mask file** to attach a selected
`SCARLET_masks` bundle and **Find compatible mask** to attach the newest
compatible bundle found in the workflow output folder.
The **Processing** page lists only sample-scattering candidates, reports their
pending, processed, or unavailable status, and supports multiple-row
selection. **Process selected** reduces the chosen rows, while **Process
pending** reduces every pending sample acquisition. Both run in the background
and remain disabled until the readiness diagnostic has no blocking issue.
The separate **Stitching** page merges all processed configurations for one
selected sample. Its control table marks samples as stitched from the presence
of their `<sample>_merged.txt` output file. It supports selecting multiple
ready samples, or every pending one; a single selected sample can be rebased
to one of its processed configurations. Result visualization is introduced in
a subsequent milestone.
The application menu **Settings → Stitching options…** exposes the complete
set of greedy-stitching algorithm parameters (except the per-run reference and
normalization factor) and stores them in the workflow context.
The **Results** page discovers reduced and stitched I(Q) text files in the
workflow output folder. Multiple selected rows are overlaid in a silx curve
plot, including available intensity uncertainties.
