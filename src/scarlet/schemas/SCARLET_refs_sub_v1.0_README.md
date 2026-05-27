# SCARLET_refs_sub v1.0

This file describes the **SCARLET subtraction reference file**, stored as one NeXus/HDF5 file **per instrumental configuration**.

It is designed to keep, in a **portable and self-contained way**, all information needed to apply subtraction corrections for a given SANS configuration:

- copied reference measurements
- beam center coordinates per detector
- detector masks
- transmission ROI definition
- optional attenuation factor used during transmission measurements
- configuration metadata

This file may contain:
- both subtraction references and masks
- only subtraction references
- only masks

---

## Global structure

```text
/entry (NXentry)
  definition = "SCARLET_refs_sub"
  schema_version = "1.0"
  config_id = "<string>"

  /configuration (NXcollection)
  /references (NXcollection)
  /beam_center (NXcollection)        # OPTIONAL
  /mask (NXcollection)
  /transmission_roi (NXcollection)
  /transmission_setup (NXcollection)   # OPTIONAL
  /meta (NXcollection)
```

---

## `/entry/configuration`

This group stores the instrumental configuration associated with the reference file.

### Required fields
- `wavelength` (float, angstrom)
- `sample_detector_distance` (float, m)

### Optional fields
- `notes` (string)

### `/entry/configuration/collimation`

This subgroup describes the collimation using **two apertures** and two distances:

- `collimation_distance`: distance between aperture1 and aperture2
- `last_aperture_to_sample_distance`: distance between aperture2 and the sample

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

Possible references are:

- `/entry/references/dark`
- `/entry/references/empty_beam_transmission`
- `/entry/references/empty_beam_scattering`
- `/entry/references/empty_cell_transmission`
- `/entry/references/empty_cell_scattering`

All of them are optional.

This design ensures that the file stays portable across computers without broken external links.

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
- `mask_detector0`, `mask_detector1`, ...

### Convention
- `1 = masked`
- `0 = valid`

Masks are optional.

---

## `/entry/transmission_roi`

This group defines the detector ROI used to calculate transmission.

### Required fields
- `detector` (int, default usually `0`)
- `roi_type` (currently `"rectangle"`)
- `x0`, `x1`, `y0`, `y1`

### Optional fields
- `notes` (string)

### Convention
Python-style bounds:
- `x0 <= x < x1`
- `y0 <= y < y1`

So `x1` and `y1` are **exclusive**.

---

## `/entry/transmission_setup` (OPTIONAL)

This group stores the attenuation factor used for transmission measurements.

### `/entry/transmission_setup/attenuator` (NXattenuator)

Required (if present):
- `attenuation_factor` (float, unitless)

**Convention**: the attenuated beam is corrected by multiplying by `attenuation_factor`

---

## `/entry/meta`

This group stores traceability information.

Required:
- `created_utc` (ISO8601 string)
- `mask_convention` (recommended value: `"1=masked"`)

Optional:
- source file paths
- `scarlet_version`

---

## Notes

- This file is defined **per configuration**.
- It is intended to be reused for the reduction of many sample files acquired under the same configuration.
- Detector efficiency normalization is **not included yet** in this specification.
- The workflow should inspect what is present in the file and apply only the available corrections.
