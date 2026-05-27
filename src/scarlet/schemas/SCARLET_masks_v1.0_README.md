# SCARLET_masks v1.0

This file describes a **SCARLET mask bundle**, stored as one NeXus/HDF5 file.

It is intended to save the detector masks drawn by a user in the graphical mask editor,
together with a snapshot of the instrumental configuration inferred from the source NeXus file.

This file contains:
- detector masks, one dataset per detector
- the inferred configuration block of the source file
- traceability metadata pointing back to the source file

---

## Global structure

```text
/entry (NXentry)
  definition = "SCARLET_masks"
  schema_version = "1.0"
  config_id = "<string>"                 # optional

  /configuration (NXcollection)
  /mask (NXcollection)
  /meta (NXcollection)
```

---

## `/entry/configuration`

This group stores the instrumental configuration inferred from the source NeXus file.

Possible datasets:
- `wavelength` (float, angstrom)
- `sample_detector_distance` (float, m)
- `notes` (string)

Optional subgroup:
- `/entry/configuration/collimation`

### `/entry/configuration/collimation`

If present, this subgroup stores the inferred collimation:

- `collimation_distance` (m)
- `last_aperture_to_sample_distance` (m)
- `/entry/configuration/collimation/aperture1`
- `/entry/configuration/collimation/aperture2`

Each aperture must be either:
- `NXslit`
- `NXpinhole`

If aperture is `NXslit`:
- `x_gap` (m)
- `y_gap` (m)

If aperture is `NXpinhole`:
- `diameter` (m)

---

## `/entry/mask`

This group stores the detector masks drawn by the user.

Possible datasets:
- `mask_detector0`
- `mask_detector1`
- ...

### Convention
- `1 = masked`
- `0 = valid`

Each mask dataset is a 2D array matching the detector image shape used during editing.

---

## `/entry/meta`

This group stores traceability information for the saved mask bundle.

Required datasets:
- `created_utc` (ISO8601 string)
- `mask_convention` (recommended value: `"1=masked, 0=valid"`)
- `source_file` (absolute path of the source NeXus file)
- `source_entry_path` (entry path used when reading the source file)

Optional datasets:
- `configuration_issues` (newline-separated issues encountered while inferring the configuration)

---

## Notes

- This file is independent from `SCARLET_refs_sub` and `SCARLET_refs_norm`.
- It is designed as an intermediate artifact for user-drawn detector masks.
- The saved configuration is a snapshot of the source file used during mask creation.
