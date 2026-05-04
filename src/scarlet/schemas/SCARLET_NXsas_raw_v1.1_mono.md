# SCARLET NXsas_raw v1.1 (Monochromatic Profile)

This document highlights the **v1.1 update**: the collimation definition in raw files now includes the same
two-aperture summary as in `SCARLET_refs_sub`.

## Required collimation summary (aligned with refs_sub)

```
/entry/instrument/collimation (NXcollection)
    collimation_distance [m]                # distance between aperture1 and aperture2
    last_aperture_to_sample_distance [m]    # distance between aperture2 and sample (z=0)

    /aperture1 (NXslit or NXpinhole)
      NXslit: x_gap [m], y_gap [m]
      NXpinhole: diameter [m]

    /aperture2 (NXslit or NXpinhole)
      NXslit: x_gap [m], y_gap [m]
      NXpinhole: diameter [m]
```

Optionally, a more detailed collimation chain may still be stored under:

- `/entry/instrument/collimation/element_order`
- `/entry/instrument/collimation/elements/...`

but the two-aperture summary is the required baseline.
