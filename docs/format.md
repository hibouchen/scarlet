# Generic data format (SCARLET baseline)

The goal is to define a NeXus/HDF5 container that is **close to NXsas**, while providing a more usable **collimation** description for reduction: multiple elements, positions, and apertures.

This document describes a SCARLET convention, or profile. It retains the NXsas structure (`NXentry`, `instrument/`, `sample/`, and `data/`) while adding a richer collimation model.

## General conventions

- Origin: sample center.
- Axes: `+z` downstream, `+x` horizontal (beam-right), `+y` vertical (up).
- Elements upstream of the sample have `z < 0`.

## Minimal organization (raw profile)

Simplified paths:

```
/raw_data (NXentry)
  definition = "NXsas_raw"            # SCARLET convention for raw data
  /sample (NXsample)
  /instrument (NXinstrument)
    /geometry                          # Existing reserved SCARLET group
    /collimation (NXcollection)        # SCARLET extension, detailed below
    /attenuator (NXattenuator)         # Optional
    /detector0 (NXdetector)
  /control (NXmonitor)
  /data
    /data0/data                        # Raw 2D image (SCARLET v0.1)
```

The current implementation converges on the `scarlet_nxsas_raw_v1.3_mono.yaml` profile: raw data are stored in `NXdetector` groups (`/raw_data/instrument/detector0`, `/raw_data/instrument/detector1`, and so on) and exposed through `NXdata` groups (`/raw_data/data0`, `/raw_data/data1`, and so on) containing links to those detectors.

## Attenuator

The optional attenuator group is stored below the active raw entry:

```text
/<entry>/instrument/attenuator (NXattenuator)
  selection = "..."
  attenuator_transmission = 0.217
```

`selection` identifies the instrument setting. `attenuator_transmission` is
the physical ratio `I_attenuated / I_incident`, so it must satisfy
`0 < attenuator_transmission <= 1`. It is not the inverse correction factor.

The SANSLLB converter accepts current and transitional raw field names. A raw
`attenuation_factor` is inverted before being written as
`attenuator_transmission`. For SAM, the factor between parentheses in the raw
attenuator state is interpreted as the inverse correction factor and is also
inverted. The converted group is optional so older files without attenuator
metadata remain valid. If the group exists but no valid transmission can be
read, direct-flux calibration reports an error rather than assuming unity.

## Collimation (SCARLET extension)

In NXsas, collimation is often summarized as a single collimator object and/or a collimation length. For SANS, this is insufficient: reduction needs to know the successive elements, such as guides, diaphragms, pinholes, slits, and sollers, together with their positions and apertures.

### Principle

Create the following group:

```
/raw_data/instrument/collimation (NXcollection)
```

It contains a sequence of collimation elements. Each element is a NeXus group with a meaningful `NX_class`, for example `NXslit`, `NXpinhole`, `NXguide`, or `NXcollimator`.

### Example tree

A typical SANS setup with two slits, a pinhole, and a guide can be represented as follows. Group names are arbitrary.

```
/raw_data/instrument/collimation (NXcollection)
  order = ["slit_1", "guide_1", "slit_2", "pinhole_1"]

  /slit_1 (NXslit)
    distance = -8.0            # m, relative to the sample center
    x_gap = 0.010              # m
    y_gap = 0.010              # m

  /guide_1 (NXguide)
    distance = -6.0            # m
    # Instrument-specific parameters, such as cross section or coating

  /slit_2 (NXslit)
    distance = -2.5            # m
    x_gap = 0.006              # m
    y_gap = 0.006              # m

  /pinhole_1 (NXpinhole)
    distance = -1.2            # m
    diameter = 0.008           # m
```

In this example:

- all elements are upstream of the sample, so their distances are negative;
- the order is made explicit with the optional `order` dataset; otherwise it can be reconstructed by sorting on `distance`;
- the aperture fields (`x_gap`, `y_gap`, and `diameter`) are expressed in meters.

### Element positions

To make the format immediately usable without requiring a complete `NXtransformations` chain, SCARLET introduces the following convention:

- Every collimation element **must** include a scalar floating-point `distance` field in meters. It represents the position along `z` relative to the sample center.
  - `distance < 0`: upstream of the sample.
  - `distance > 0`: downstream of the sample.

The `distance` field is compatible with a future move to `depends_on` and `NXtransformations`, the more NeXus-native representation.

### Recommended fields by type

Exact fields depend on the instrument, but the following are recommended:

- `NXslit`: `x_gap`, `y_gap` in meters, plus `distance`.
- `NXpinhole`: `diameter` in meters, plus `distance`.
- `NXguide`: `state` (`"in"` or `"out"`), geometry or section as needed, plus `distance`.
- `NXcollimator`: soller or divergence parameters when available, plus `distance`.

### Ordering

Ordering can be:

- implicit through `distance`, sorted in ascending order; or
- explicit through an optional `order` dataset in `instrument/collimation`, containing group names.

### h5py creation example

The following Python excerpt creates the collimation structure only:

```python
import h5py

with h5py.File("my_file.nxs", "a") as f:
    inst = f["raw_data/instrument"]
    coll = inst.require_group("collimation")
    coll.attrs["NX_class"] = "NXcollection"

    # Optional, but useful to preserve the sequence explicitly.
    coll.create_dataset("order", data=[b"slit_1", b"guide_1", b"slit_2", b"pinhole_1"])

    s1 = coll.require_group("slit_1")
    s1.attrs["NX_class"] = "NXslit"
    s1.create_dataset("distance", data=-8.0).attrs["units"] = "m"
    s1.create_dataset("x_gap", data=0.010).attrs["units"] = "m"
    s1.create_dataset("y_gap", data=0.010).attrs["units"] = "m"

    g1 = coll.require_group("guide_1")
    g1.attrs["NX_class"] = "NXguide"
    g1.create_dataset("distance", data=-6.0).attrs["units"] = "m"
    g1.create_dataset("state", data=b"in")

    s2 = coll.require_group("slit_2")
    s2.attrs["NX_class"] = "NXslit"
    s2.create_dataset("distance", data=-2.5).attrs["units"] = "m"
    s2.create_dataset("x_gap", data=0.006).attrs["units"] = "m"
    s2.create_dataset("y_gap", data=0.006).attrs["units"] = "m"

    p1 = coll.require_group("pinhole_1")
    p1.attrs["NX_class"] = "NXpinhole"
    p1.create_dataset("distance", data=-1.2).attrs["units"] = "m"
    p1.create_dataset("diameter", data=0.008).attrs["units"] = "m"
```

## Reduction products in the same file

The goal is to write reduced results as a **new `NXentry`** in the same logical NeXus file, keeping raw and processed data together.

The Python workflow keeps instrument raw files unchanged during initialization.
When the pipeline first needs a run, it creates or reuses the associated SCARLET
NeXus file in the workflow output directory. The workflow pipeline writes its
result under the top-level entry:

```
/processed (NXentry)
  definition = "SCARLET_processed"
  /meta (NXcollection)
  /data (NXcollection)
    /detector0 (NXdata)
    /detector1 (NXdata)   # Optional
  /data0 (NXdata)         # NeXus view for detector0
  /data1 (NXdata)         # Optional
```

The presence of `/processed` is also the criterion used by
`WorkflowContext.is_run_processed()` and the processing-status tables. See
[Workflow context](workflow.md) for the complete lifecycle.

The workflow pipeline writes one azimuthal text curve per detector and records
the processed detector arrays in `/processed`. Bin counts may differ by
detector and bin boundaries may be linear or logarithmic in q. Text outputs
contain q, `I`, `I_error`, and `q_error`.

## Flatfield calibration format

Water flatfields use a separate `SCARLET_flatfield` NeXus file:

```text
/entry (NXentry)
  definition = "SCARLET_flatfield"
  direct_flux                         # counts/monitor, full detector
  /instrument (NXinstrument)
    /detector0 (NXdetector)
      flatfield
      flatfield_errors
      pixel_mask
      normalisation_factor            # target units cm^-1
    /detector1 (NXdetector)            # Optional
      ...
```

For detector `N`, the stored factor is `10 * mean(valid flatfield_N) /
direct_flux`. Pixels marked in `pixel_mask`, as well as non-finite or
non-positive pixels, do not contribute to the mean. During reduction SCARLET
multiplies the corrected detector by this factor and by the ratio between the
flatfield direct flux and the active configuration direct flux. See
[Workflow context](workflow.md#prepare-flatfields-and-absolute-calibration)
for the complete formula and failure behavior.
