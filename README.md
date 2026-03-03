# SCARLET
### SCA ttering R eduction and ana L ysis E nvironmen T

**SCARLET** is a NeXus-native, instrument-agnostic framework for small-angle neutron scattering (SANS) data reduction.

It provides a unified pipeline from raw instrument files to absolute I(Q), with strict geometry conventions and reproducible metadata handling.

---

## 🔬 Vision

Modern SANS experiments are performed across multiple facilities and instrument geometries (reactor, TOF, multi-detector, VSANS).
However, data reduction pipelines remain instrument-specific and heterogeneous.

SCARLET aims to:

- Define a common NeXus raw container (NXsas_raw)
- Provide a geometry-consistent reduction framework
- Ensure reproducibility and validation
- Enable future AI-assisted reduction
- Harmonize multi-instrument workflows

---

## 📐 Geometry Convention

- Origin at sample center
- +z downstream
- +x horizontal (beam-right)
- +y vertical up

All collimation and detector elements are defined using NXtransformations.

---

## 🧱 Architecture

SCARLET is structured around:

- A unified raw container (NXsas_raw)
- A core dataset abstraction (ScarletDataset)
- Modular reduction steps
- Validation and provenance tracking

See `docs/format.md` for the proposed generic data format (NXsas-like) and the extended collimation model.

---

## 🔁 Minimal Pipeline

```python
import scarlet as sc

dataset = sc.load("raw_file.nxs")

iq = (
    sc.Reduction(dataset)
      .normalize()
      .transmission()
      .background()
      .azimuthal()
      .absolute_scale()
      .run()
)

iq.save("result.nxs")
```

---

## 🗂️ Données locales (pour tests)

Il n’y a pas de dossier imposé : vous passez simplement le chemin du fichier à `scarlet`.

Par convention :

- `data/<instrument>/raw/` : **données brutes instrument**
- `data/<instrument>/processed/` : **données converties** (ex. `NXsas_raw`) + **sorties générées**

Le contenu de `data/` est ignoré par git via `.gitignore` (seuls les `README.md`/`.gitkeep` restent suivis).

Exemples :

```bash
scarlet info data/D11/raw/mon_fichier.nxs
scarlet reduce data/D11/processed/nxsas_raw.nxs -o data/D11/processed/result.nxs
```

Note : la réduction (`scarlet reduce`) attend un fichier au format `NXsas_raw`. Si vos fichiers “bruts instrument” ne sont pas déjà en `NXsas_raw`, il faut les convertir via un adaptateur (voir `src/scarlet/io/adapters/`).

---

## 🧾 Résultats dans le même fichier (multi-entry)

Pour éviter de multiplier les fichiers et pouvoir relancer une réduction, vous pouvez écrire les résultats dans **une nouvelle entry** du même fichier NeXus/HDF5 :

```bash
scarlet reduce data/D11/processed/nxsas_raw.nxs --inplace --entry reduced --overwrite-entry
```

Cela ajoute (ou remplace) une entry `reduced` contenant un `definition=NXsas` et un `NXdata` minimal (`Q`, `I`, `I_errors`).

---

## 🔄 Conversion SANSLLB → NXsas_raw

Si vos fichiers bruts SANSLLB ne sont pas déjà au format `NXsas_raw`, utilisez l’adaptateur :

```python
from scarlet.io.adapters.sansllb import convert
convert("data/SANSLLB/raw/mon_fichier.nxs", "data/SANSLLB/processed/nxsas_raw.nxs")
```

Note : par défaut, l’adaptateur utilise `nexusformat` si installé (extra `scarlet-sans[nexus]`), sinon il retombe sur un backend `h5py`. Pour forcer un backend : `convert_sansllb(..., backend="nexusformat")`. Certaines correspondances restent marquées `TODO` dans `src/scarlet/io/adapters/sansllb.py`.
Dans les fichiers SANSLLB présents ici, la collimation est récupérée depuis `/.../collimator` (slits/guides) quand disponible (conversion mm→m) ; les distances manquantes restent à `NaN` avec un `TODO`.

---

## 🚀 Roadmap

- v0.1 — Core deterministic reduction  
- v0.2 — Multi-distance stitching  
- v0.3 — Resolution function handling  
- v0.4 — AI-assisted masking & QA  
- v1.0 — Instrument-agnostic validated release  

---

## 🔴 Why SCARLET?

- NeXus-native  
- Physically rigorous  
- Instrument-agnostic  
- Designed for future neutron sources  
