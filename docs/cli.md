# Ligne de commande

La commande principale est `scarlet`.

```bash
scarlet --help
```

## Schemas

Listez les schemas embarques :

```bash
scarlet schema list
```

Validez un fichier NeXus/HDF5 :

```bash
scarlet validate data/SANSLLB/processed/run_001_scarlet.nxs \
  --schema scarlet_nxsas_raw_v1.3_mono.yaml
```

L'option `--strict` traite aussi les avertissements comme des erreurs. `--entry` permet de designer explicitement l'entree NeXus a valider.

## Conversion

Affichez les convertisseurs disponibles :

```bash
scarlet convert list
```

Convertissez un fichier brut et validez le resultat :

```bash
scarlet convert sansllb data/SANSLLB/raw/run_001.nxs \
  data/SANSLLB/processed/run_001_scarlet.nxs \
  --overwrite \
  --validate
```

Les convertisseurs enregistres comprennent `d11`, `sansllb` et `sam`.

## Integration azimutale

Exportez une courbe `I(Q)` depuis une sortie traitee :

```bash
scarlet azimuthal-average reduced_2d.nxs iq.csv --overwrite
```

Les options `--processed-entry`, `--detector`, `--q-min` et `--q-max` permettent de selectionner les donnees a exporter.

## Visualiseur

Ouvrez le visualiseur NeXus et de masques :

```bash
scarlet viewer data/SANSLLB/processed --instrument sansllb
```

Pour les conversions temporaires effectuees par le visualiseur, les instruments pris en charge sont `sansllb` et `sam`.
