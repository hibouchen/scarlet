# SCARLET_refs_norm v1.0

This file describes the **SCARLET normalization reference file**, stored as one NeXus/HDF5 file **per target instrumental configuration**.

It is designed to keep, in a **portable and self-contained way**, everything needed to prepare or apply normalization for a given SANS configuration, while keeping the same overall structure as `SCARLET_refs_sub`.

This file contains:
- the same configuration description as `SCARLET_refs_sub`
- optional subtraction-related references if needed for consistency
- beam center coordinates
- detector masks
- transmission ROI definition
- optional transmission attenuation factor
- and, most importantly, the two water reference files:
  - `water_scattering`
  - `water_transmission`

A key point of this schema is that the **target configuration** is identified by `/entry/config_id`,
but the water references may optionally come from **another configuration** if better statistics are needed.

---

## Global structure

```text
/entry (NXentry)
  definition = "SCARLET_refs_norm"
  schema_version = "1.0"
  config_id = "<target configuration id>"

  /configuration (NXcollection)
  /references (NXcollection)
  /beam_center (NXcollection)        # optional
  /mask (NXcollection)
  /transmission_roi (NXcollection)
  /transmission_setup (NXcollection)   # optional
  /meta (NXcollection)
```

---

## `/entry/configuration`

This group stores the **target instrumental configuration** for which normalization will be applied.

### Required fields
- `wavelength` (float, angstrom)
- `sample_detector_distance` (float, m)

### Optional fields
- `notes` (string)

### `/entry/configuration/collimation`

This subgroup describes the collimation using **two apertures** and two distances:

- `collimation_distance`: distance between `aperture1` and `aperture2`
- `last_aperture_to_sample_distance`: distance between `aperture2` and the sample

Both distances are stored in **meters**.

### Apertures
Two apertures are defined:

- `/entry/configuration/collimation/aperture1`
- `/entry/configuration/collimation/aperture2`

Each aperture must be either:

- `NXslit`
- `NXpinhole`

#### If aperture is `NXslit`
Required fields:
- `x_gap` (m)
- `y_gap` (m)

#### If aperture is `NXpinhole`
Required field:
- `diameter` (m)

---

## `/entry/references`

This group stores copied reference measurements.

Each reference is stored as an `NXcollection` containing a subgroup named `entry`,
which is a deep copy of the original source file `/entry`.

### Optional subtraction-related references
These are allowed so that a normalization lot can remain structurally parallel to a subtraction lot:

- `/entry/references/dark`
- `/entry/references/empty_beam_transmission`
- `/entry/references/empty_beam_scattering`
- `/entry/references/empty_cell_transmission`
- `/entry/references/empty_cell_scattering`

All of them are optional.

### Required normalization references
These two groups are required:

- `/entry/references/water_scattering`
- `/entry/references/water_transmission`

Each must contain:
- `/entry` : deep copy of the source file `/entry`

Each may also contain:
- `source_config_id` (string, optional)

This optional `source_config_id` is used when the water reference comes from another instrumental configuration
than the one described by `/entry/config_id`.

### Example

```text
/entry/references/water_scattering
  source_config_id = "long"
  /entry
    ...
```

This means:
- the normalization file targets one configuration (for example `medium`)
- but the water scattering reference was borrowed from configuration `long`

---

## `/entry/beam_center`

This optional group stores beam center coordinates for each detector.

Suggested layout:
- `/entry/beam_center/detector0/beam_center_x`
- `/entry/beam_center/detector0/beam_center_y`
- `/entry/beam_center/detector1/beam_center_x`
- `/entry/beam_center/detector1/beam_center_y`

### Convention
- coordinates are stored in detector pixel units
- `beam_center_x` is the horizontal pixel coordinate
- `beam_center_y` is the vertical pixel coordinate

---

## `/entry/mask`

This group stores detector masks.

Possible datasets:
- `mask_detector0`
- `mask_detector1`
- ...

### Convention
- `1 = masked`
- `0 = valid`
- beamstop-masked pixels are included directly in these detector masks

Masks are optional.

---

## `/entry/transmission_roi`

This group defines the detector ROI used to calculate transmission.

### Required fields
- `detector` (int, default usually `0`)
- `roi_type` (currently `"rectangle"`)
- `x0`
- `x1`
- `y0`
- `y1`

### Optional fields
- `method` (string, e.g. `"sum"`)
- `notes` (string)

### Convention
Python-style bounds:
- `x0 <= x < x1`
- `y0 <= y < y1`

So `x1` and `y1` are **exclusive**.

---

## `/entry/transmission_setup` (optional)

This group stores the attenuation factor used for transmission measurements.

### `/entry/transmission_setup/attenuator` (NXattenuator)

Required (if present):
- `attenuation_factor` (float, unitless)

Convention:
- `I_corrected = I_measured * attenuation_factor`

---

## `/entry/meta`

This group stores traceability information.

### Required fields
- `created_utc` (ISO8601 string)
- `mask_convention` (recommended value: `"1=masked"`)

### Optional fields
- `dark_source_file`
- `empty_beam_transmission_source_file`
- `empty_beam_scattering_source_file`
- `empty_cell_transmission_source_file`
- `empty_cell_scattering_source_file`
- `water_scattering_source_file`
- `water_transmission_source_file`
- `scarlet_version`

---

## Notes

- This file is defined for a **target configuration**.
- It is intended to be reused for the normalization of many sample files acquired under the same target configuration.
- Water references may optionally come from **another configuration** if better counting statistics are needed.
- This schema keeps the same overall organization as `SCARLET_refs_sub` to simplify tooling and workflows.
