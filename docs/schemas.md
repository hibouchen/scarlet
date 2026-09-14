# Schemas NeXus

SCARLET distribue des schemas YAML pour valider les fichiers NeXus/HDF5 crees ou utilises par le workflow. La commande suivante affiche les noms exacts disponibles dans l'installation :

```bash
scarlet schema list
```

Les schemas principaux sont :

| Schema | Role |
| --- | --- |
| `scarlet_nxsas_raw_v1.3_mono.yaml` | Donnees brutes SCARLET au profil `NXsas_raw` mono-longueur d'onde. |
| `scarlet_masks_v1.0.yaml` | Paquet de masques de detecteur. |
| `scarlet_refs_sub_v1.0.yaml` | References necessaires a la soustraction. |
| `scarlet_refs_norm_v1.0.yaml` | References necessaires a la normalisation. |
| `scarlet_workflow_context_v1.0.yaml` | Contexte et artefacts du workflow. |

Les explications fonctionnelles des profils sont conservees avec le paquet : [schemas SCARLET](https://github.com/hibouchen/scarlet/tree/main/src/scarlet/schemas).

Pour la structure du format brut et la convention de collimation, consultez le [format de donnees](format.md).
