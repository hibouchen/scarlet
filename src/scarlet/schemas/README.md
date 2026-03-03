# SCARLET NXsas_raw v1.0 (Monochromatic Profile)
## Raw Data Format Specification for SANS (no TOF)

---

## 1. Purpose

**NXsas_raw** is the SCARLET raw data convention built on top of NeXus.
It defines a unified, instrument-agnostic structure for storing **raw SANS data** as a common starting point for reduction.

This format:

- Stores raw detector counts only (no reduced data)
- Is strictly NeXus-compliant
- Provides a single reduction entry point
- Uses a fixed geometric convention
- Supports multi-detector instruments

---

## 2. Global Geometry Convention (SCARLET)

All instrument components are expressed in the following coordinate system:

- Origin: sample center
- +z: downstream (beam direction)
- +x: horizontal (beam-right)
- +y: vertical (upwards)
- Length units: meters
- Angle units: degrees

All positions use `NXtransformations`.

---

## 3. File Structure

```
/entry (NXentry)
    definition = "NXsas_raw"
    schema_version = "1.0"

    /sample (NXsample)
    /instrument (NXinstrument)
    /control (NXmonitor)        # optional
    /data0 (NXdata)
    /data1 (NXdata)
    ...
```

---

## 4. NXsample

```
/entry/sample (NXsample)
    name
    thickness [m]                (recommended)
    transmission [1]             (optional)
```

Sample environments may be stored using `NXlog`.

---

## 5. NXinstrument

### 5.1 Geometry

```
/entry/instrument/geometry (NXcollection)
    origin_definition = "sample center"
    axis_convention   = "+z downstream, +x beam-right, +y up"
    units_length      = "m"
    units_angle       = "deg"
```

### 5.2 Source (optional)

```
/entry/instrument/source (NXsource)
    name
    type
    probe
```

### 5.3 Monochromator (required)

```
/entry/instrument/monochromator (NXmonochromator)
    wavelength [angstrom]
    wavelength_error [angstrom]  (optional)

    /velocity_selector (NXvelocity_selector) (optional)
```

### 5.4 Collimation (detailed, required)

Collimation is described as an ordered chain of physical elements upstream of the sample.

```
/entry/instrument/collimation (NXcollection)
    description        (optional)
    element_order      (required, array of strings)

    /elements (NXcollection)
        /<element_name> (NXaperture | NXslit | NXguide)
            ... geometry fields ...
            /transformations (NXtransformations)
                translation [m] = (x,y,z)   # typically z < 0 upstream
                rotation [deg] (optional)
```

Allowed element types in v1.0:
- `NXaperture`
- `NXslit`
- `NXguide`

Element order is defined by `element_order` (beamline order, typically increasing z).

### 5.5 Beamstop (optional)

```
/entry/instrument/beamstop (NXbeamstop)
    shape
    dimensions
    /transformations
        translation [m]
```

---

## 6. NXdetector

Each detector lives under:

```
/entry/instrument/detector0 (NXdetector)
/entry/instrument/detector1 (NXdetector)
...
```

### 6.1 Mandatory fields (SCARLET)

- `data` (raw counts)
- `x_pixel_size` [m]
- `y_pixel_size` [m]
- `beam_center_x` [pixel]
- `beam_center_y` [pixel]
- `dead_time` [s]
- `/transformations/translation`

### 6.2 Optional fields (SCARLET)

Identification:
- `local_name`
- `type`
- `description`
- `serial_number`

Corrections & calibration:
- `data_errors`
- `pixel_mask`
- `pixel_mask_applied` (bool)
- `flatfield`
- `flatfield_errors`
- `flatfield_applied` (bool)
- `countrate_correction_applied` (bool)

Efficiency:
- `efficiency` (scalar or array)
- `wavelength` (only if efficiency depends on wavelength)

Timing:
- `count_time`
- `real_time`

Advanced geometry:
- `distance`
- `x_pixel_offset`
- `y_pixel_offset`
- `z_pixel_offset`

---

## 7. NXmonitor (optional but recommended)

Monitors are used for beam normalization and acquisition control.

```
/entry/instrument/monitor0 (NXmonitor)
/entry/instrument/monitor1 (NXmonitor)
/entry/control (NXmonitor)
```

Required fields:
- `mode` = "monitor" | "timer"
- `preset`
- `integral` OR `data`

Optional fields:
- `count_time`
- `start_time`
- `end_time`
- `/transformations/translation`

---

## 8. NXdata (NXlink views)

Raw detector data are stored in:

```
/entry/instrument/detectorX/data
```

Standard reduction entry points are `NXdata` groups at `/entry/dataN`, which link to detector datasets:

```
/entry/data0 (NXdata)
    counts        -> link to /entry/instrument/detector0/data
    counts_errors -> link to /entry/instrument/detector0/data_errors (optional)
    @signal = "counts"   (recommended)
    @axes   = [...]      (recommended)
```

This avoids data duplication.

---

## 9. What this format guarantees

- Strict NeXus compliance
- Instrument-agnostic raw structure
- Explicit geometry
- Multi-detector support
- Clean separation of instrument and data
- Deterministic reduction starting point

---

## 10. What this format does NOT contain

- No I(Q)
- No background subtraction
- No normalization
- No resolution convolution

This is a harmonized raw data container.
