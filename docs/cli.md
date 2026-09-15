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
