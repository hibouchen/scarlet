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
    /detector
  /control (NXmonitor)
  /data
    /data0/data                        # Raw 2D image (SCARLET v0.1)
```

The current implementation converges on the `scarlet_nxsas_raw_v1.3_mono.yaml` profile: raw data are stored in `NXdetector` groups (`/raw_data/instrument/detector0`, `/raw_data/instrument/detector1`, and so on) and exposed through `NXdata` groups (`/raw_data/data0`, `/raw_data/data1`, and so on) containing links to those detectors.

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

The high-level reduction command, for example `scarlet reduce`, is not yet implemented in the current repository. Available commands currently cover conversion, validation, generation of `refs_sub` and `refs_norm` reference bundles, and an initial deterministic 2D correction through `scarlet reduce-2d`.

When needed, `scarlet reduce-2d` copies the raw file and then adds:

```
/raw_data (NXentry)   # Present in recent converted files
  definition = "NXsas_raw"
  ...

/processed_data (NXentry)
  definition = "SCARLET_azimuthal_iq"
  schema_version = "0.2"
  /data (NXdata)
    # Alias to the first reduced detector
  /data0 (NXdata)
    I                 # 1D azimuthal curve for detector0
    Q                 # Bin centers in 1/angstrom
    Q_edges           # Bin edges
    n_pixels          # Number of accumulated pixels per bin
  /data1 (NXdata)     # Optional when detector1 exists
    I
    Q
    Q_edges
    n_pixels
  /detector0 (NXcollection)
    I_2d              # Corrected 2D image for detector0
    Qx
    Qy
    sample_corrected
    water_corrected   # Optional
    mask              # Optional, 1 means masked
  /detector1 (NXcollection)   # Optional when detector1 exists
    I_2d
    Qx
    Qy
    sample_corrected
    water_corrected   # Optional when refs_norm is supplied
    mask              # Optional, 1 means masked
  /reduction (NXprocess)
    /detector_indices
    /sample_transmission/value
    /water_transmission/value   # Optional
    /inputs/...
```

The `SCARLET_azimuthal_iq` product remains deliberately preliminary: it does not yet include uncertainty propagation or multi-detector merging.
