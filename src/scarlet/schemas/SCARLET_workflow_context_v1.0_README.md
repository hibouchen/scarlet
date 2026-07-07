# SCARLET_workflow_context v1.0

This file describes a serialized **SCARLET workflow context**, stored as one NeXus/HDF5 file.

The schema is intended to track the current in-memory shape of `scarlet.workflow.context.WorkflowContext`, so a saved project can be reloaded later without recomputing run classification, reference attachment, detector masks, beam centers, ROI selection, or flatfield bookkeeping.

This file contains:
- workflow metadata and paths
- the full run registry, including duplicate logical runs
- saved instrumental configurations
- per-configuration beam centers and transmission ROI
- reference file paths grouped by entity and acquisition mode
- sample transmissions, empty-cell transmissions, and sample thicknesses
- mask bundles, in-memory detector masks, flatfields, flatfield-source mappings, and stale-flatfield markers
- logs, issues, artifacts, timings, and a safe serialized subset of `WorkflowContext.store`

Heavy transient caches such as open HDF5 handles, detector frames, or frame-error arrays are intentionally excluded.

---

## Global structure

```text
/entry (NXentry)
  definition = "SCARLET_workflow_context"
  schema_version = "1.0"

  /metadata (NXcollection)
  /runs (NXcollection)
  /configurations (NXcollection)
  /beam_centers (NXcollection)
  /rois (NXcollection)
  /references (NXcollection)
  /transmissions (NXcollection)
  /sample_thicknesses (NXcollection)
  /masks (NXcollection)
  /flatfield_sources (NXcollection)
  /stale_flatfields (NXcollection)
  /artifacts (NXcollection)
  /logs (NXcollection)
  /issues (NXcollection)
  /timings (NXcollection)
  /store (NXcollection)
```

---

## `/entry/metadata`

Required datasets:
- `experiment_id`
- `instrument_name`
- `root_dir`
- `output_dir`
- `created_utc`

Optional datasets:
- `scarlet_version`
- `schema_raw`
- `schema_refs_sub`
- `schema_refs_norm`
- `schema_masks`

`schema_*` entries are legacy compatibility metadata; they are no longer required by the current `WorkflowContext` object.

---

## `/entry/runs`

This group stores the full logical run registry.

Required parallel datasets:
- `sample_name`
- `config_id`
- `mode`
- `entity`
- `duplicate_index`
- `file_path`

Each row corresponds to one registered `RunKey -> file_path`.

The `duplicate_index` dataset is required because `WorkflowContext` now preserves multiple files with the same logical `(config_id, entity, mode, sample_name)` key.

---

## `/entry/configurations`

One subgroup per `config_id`.

Each configuration stores:
- `wavelength`
- `sample_detector_distance`
- optional `config_id`
- optional `notes`
- optional `/collimation`

If `/collimation` is present, it may contain:
- `collimation_distance`
- `last_aperture_to_sample_distance`
- optional `/aperture1`
- optional `/aperture2`

Each aperture subgroup stores:
- `type`
- optional `x_gap`
- optional `y_gap`
- optional `diameter`

This mirrors the current `Configuration`, `Collimation`, and `Aperture` dataclasses used by the workflow layer.

---

## `/entry/beam_centers`

One subgroup per `config_id`, then one subgroup per detector:

```text
/entry/beam_centers/config_1/detector0
/entry/beam_centers/config_1/detector1
```

Each detector subgroup stores:
- `beam_center_x`
- `beam_center_y`

---

## `/entry/rois`

One subgroup per `config_id`.

Required datasets:
- `x0`
- `x1`
- `y0`
- `y1`

Optional datasets:
- `detector_number`
- `method`
- `notes`

The ROI follows the workflow convention `(x0, x1, y0, y1)`, with `x1` and `y1` exclusive.

---

## `/entry/references`

This group stores file paths derived from dedicated workflow attributes.

Supported subgroups:
- `dark`
- `empty_beam`
- `empty_cell`
- `water`
- `mask_files`
- `flatfields`

Layout:
- `/entry/references/dark/<config_id>`: scalar file path
- `/entry/references/empty_beam/<config_id>/scattering`: scalar file path
- `/entry/references/empty_beam/<config_id>/transmission`: scalar file path
- same layout for `empty_cell` and `water`
- `/entry/references/mask_files/<config_id>`: scalar `SCARLET_masks` bundle path
- `/entry/references/flatfields/<config_id>`: scalar flatfield artifact path

---

## `/entry/transmissions`

This group stores the two transmission caches of the current workflow object.

`/entry/transmissions/sample` stores the sample transmission table with required parallel datasets:
- `sample_name`
- `config_id`
- `value`

`/entry/transmissions/empty_cell` stores the empty-cell transmission table with required parallel datasets:
- `config_id`
- `value`

---

## `/entry/sample_thicknesses`

`/entry/sample_thicknesses/sample` stores the sample-thickness table with required parallel datasets:
- `sample_name`
- `config_id`
- `value`

---

## `/entry/masks`

This group stores the in-memory detector masks currently attached to the workflow context.

Suggested layout:
- `/entry/masks/config_1/detector0`
- `/entry/masks/config_1/detector1`
- `/entry/masks/config_2/detector0`

Each dataset is a 2D mask array following the SCARLET convention `1=masked, 0=valid`.

---

## `/entry/flatfield_sources`

This group stores the optional mapping from one target configuration to the configuration providing the reusable flatfield.

Suggested layout:
- `/entry/flatfield_sources/config_2 = "config_1"`

---

## `/entry/stale_flatfields`

This group stores the set of configurations whose flatfield artifact must be rebuilt before reuse.

Suggested layout:
- `/entry/stale_flatfields/config_1 = true`
- `/entry/stale_flatfields/config_3 = true`

Only the presence of the config id matters semantically; the dataset payload may simply be a boolean marker.

---

## `/entry/artifacts`

Required parallel datasets:
- `name`
- `path`
- `kind`
- `created_utc`

---

## `/entry/logs`

Required parallel datasets:
- `level`
- `message`
- `where`
- `when_utc`
- `meta_json`

`meta_json` stores the structured log metadata as JSON.

---

## `/entry/issues`

Required parallel datasets:
- `level`
- `message`
- `where`
- `key`
- `when_utc`
- `meta_json`

`meta_json` stores the structured issue metadata as JSON.

---

## `/entry/timings`

One scalar dataset per timing key, storing the elapsed time in seconds.

---

## `/entry/store`

This group stores a safe serialized subset of `WorkflowContext.store`.

Rules:
- only JSON-compatible values are persisted
- heavy cache entries such as `frames`, `frame_errors`, `masks`, and `transmissions` are excluded
- unsupported keys may be omitted; if so, `_skipped_keys` may list them as JSON

---

## Notes

- This file is intended for workflow resume, inspection, and deferred finalization.
- It is not a substitute for the original raw data, converted files, or reference files.
- It should stay lightweight enough to save frequently during interactive work.
