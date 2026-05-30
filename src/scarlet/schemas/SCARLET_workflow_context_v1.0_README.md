# SCARLET_workflow_context v1.0

This file describes a lightweight serialized **SCARLET workflow context**, stored as one NeXus/HDF5 file.

It is intended to save enough workflow state to reload a project later and continue or finalize the treatment without rebuilding all classification and reference-selection steps from scratch.

This file contains:
- workflow metadata and paths
- the run table
- saved instrumental configurations
- generated reference file paths
- mask file paths and in-memory detector masks
- saved transmission scalars
- logs, issues, artifacts and timings
- a safe serialized subset of `WorkflowContext.store`

Heavy transient caches such as open HDF5 handles, detector frames and frame errors are intentionally excluded.

---

## Global structure

```text
/entry (NXentry)
  definition = "SCARLET_workflow_context"
  schema_version = "1.0"

  /metadata (NXcollection)
  /runs (NXcollection)
  /configurations (NXcollection)
  /references (NXcollection)
  /masks (NXcollection)
  /transmissions (NXcollection)
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
- `schema_raw`
- `schema_refs_sub`
- `schema_refs_norm`
- `schema_masks`
- `created_utc`

Paths may be stored relative to the workflow-context file location when possible.

---

## `/entry/runs`

This group stores the same logical content as `runs_report.csv`.

Required parallel datasets:
- `sample_name`
- `config_id`
- `mode`
- `entity`
- `file_path`

Each row corresponds to one registered `RunKey -> file_path`.

---

## `/entry/configurations`

One subgroup per `config_id`.

Each configuration stores:
- `wavelength`
- `sample_detector_distance`
- optional `config_id`
- optional `notes`
- optional `/collimation`

The stored structure mirrors the `Configuration`, `Collimation` and `Aperture` dataclasses used by the workflow layer.

---

## `/entry/references`

This group stores three subgroups:
- `refs_sub_files`
- `refs_norm_files`
- `masks_files`

Each subgroup contains one scalar dataset per `config_id`, whose value is the corresponding file path.

---

## `/entry/masks`

This group stores the in-memory detector masks currently attached to the workflow context.

Suggested layout:
- `/entry/masks/config_1/detector0`
- `/entry/masks/config_1/detector1`
- `/entry/masks/config_2/detector0`

Each dataset is a 2D mask array.

---

## `/entry/transmissions`

Required parallel datasets:
- `sample_name`
- `config_id`
- `value`

Each row corresponds to one entry of the workflow transmission cache.

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

`meta_json` stores the log metadata as JSON.

---

## `/entry/issues`

Required parallel datasets:
- `level`
- `message`
- `where`
- `key`
- `when_utc`
- `meta_json`

`meta_json` stores the issue metadata as JSON.

---

## `/entry/timings`

One scalar dataset per timing key, storing the elapsed time in seconds.

---

## `/entry/store`

This group stores a safe serialized subset of `WorkflowContext.store`.

Rules:
- only JSON-compatible values are persisted
- heavy cache entries such as `frames`, `frame_errors`, `masks` and `transmissions` are excluded
- unsupported keys may be omitted; if so, `_skipped_keys` lists them as JSON

---

## Notes

- This file is intended for workflow resume, inspection and deferred finalization.
- It is not a substitute for the original raw data or reference files.
- It should remain lightweight enough to save frequently during interactive work.
