# SCARLET `NXsas_raw` v1.3 (profil monochromatique)

Ce document décrit **complètement** le schéma NeXus/HDF5 **SCARLET `NXsas_raw` v1.3 (mono)**, qui sert de **point de départ unique** pour la réduction de données SANS (Small-Angle Neutron Scattering), quel que soit l’instrument source, après conversion vers ce format.

> **Idée clé (v1.3)** : la collimation est décrite de façon **autoritative** par une **chaîne ordonnée d’éléments** (`element_order` + `elements`), et certaines grandeurs (ex. `collimation_distance`) peuvent être **dérivées** des transformations.  
> La valeur `last_aperture_to_sample_distance` doit être fournie par la **conversion** depuis le fichier original.

---

## 1. Objectif du format

`NXsas_raw` (convention SCARLET) vise à :

- stocker les **données brutes** détecteur(s) (counts, erreurs éventuelles),
- fournir une description instrumentale homogène (géométrie, collimation, longueur d’onde…),
- permettre une réduction reproductible (même structure d’entrée, indépendamment de l’instrument d’origine),
- supporter plusieurs détecteurs et plusieurs moniteurs,
- offrir des vues `NXdata` standardisées (liens vers les datasets des détecteurs).

Ce format **ne contient pas** :
- de courbes I(Q) (réduction),
- ni de corrections finales (fond, normalisation absolue, convolution de résolution…).

---

## 2. Conventions globales SCARLET

### 2.1 Repère géométrique
- Origine : **centre de l’échantillon**
- Axes :  
  - **+z** : aval (direction du faisceau)  
  - **+x** : horizontal (à droite en regardant dans le sens du faisceau)  
  - **+y** : vertical (vers le haut)

### 2.2 Unités
- Longueurs : **m**
- Angles : **deg**
- Longueur d’onde : **Å**

---

## 3. Structure globale du fichier

```
/entry (NXentry)
  definition = "NXsas_raw"
  schema_version = "1.3"            (optionnel)

  /sample (NXsample)

  /instrument (NXinstrument)
    /geometry                         (réservé SCARLET)
    /source (NXsource)                (optionnel)
    /monochromator (NXmonochromator)  (requis en profil mono)
    /collimation (NXcollection)       (requis)
    /beamstop (NXbeamstop)            (optionnel)

    /detector0 (NXdetector)           (au moins un, requis)
    /detector1 (NXdetector)           (optionnel)
    ...

    /monitor0 (NXmonitor)             (optionnel)
    /monitor1 (NXmonitor)             (optionnel)
    ...

  /control (NXmonitor)                (optionnel)
  /data0 (NXdata)                     (requis)
  /data1 (NXdata)                     (requis si detector1)
  ...
```

---

## 4. `/entry` (NXentry)

### Requis
- `/entry` (NXentry)
- `/entry/definition` = `"NXsas_raw"`

### Optionnel (recommandé)
- `/entry/schema_version` = `"1.3"`

---

## 5. `/entry/sample` (NXsample)

SCARLET impose l’existence de `NXsample`, mais **ne force pas** un ensemble minimal de champs (les instruments varient beaucoup).  
Recommandé :
- `name`
- `thickness` (m)
- `transmission` (sans unité, 0–1) si déjà mesurée/connue (sinon calculée dans le workflow)

---

## 6. `/entry/instrument` (NXinstrument)

### 6.1 `/entry/instrument/geometry` (groupe réservé SCARLET)
Groupe “réservé” à des métadonnées de convention, par exemple :
- `origin_definition = "sample center"`
- `axis_convention = "+z downstream, +x beam-right, +y up"`

Ces champs sont **recommandés** mais pas bloquants pour la validation.

### 6.2 `/entry/instrument/source` (NXsource) — optionnel
Copie d’informations source (selon disponibilité) : type, probe, etc.

### 6.3 `/entry/instrument/monochromator` (NXmonochromator) — requis (profil mono)
Requis :
- `wavelength` (Å)

Optionnel :
- `wavelength_error` (Å) si disponible (écart-type ou équivalent instrument)

Sous-groupe optionnel :
- `/velocity_selector (NXvelocity_selector)`

---

## 7. Collimation (v1.3) — **point clé**

### 7.1 Groupe autoritatif
`/entry/instrument/collimation` est un `NXcollection` qui doit contenir :

#### Requis
- `element_order` (liste ordonnée de noms d’éléments)
- `last_aperture_to_sample_distance` (m)  
  → **doit venir du fichier original lors de la conversion**
- `/elements` (NXcollection)

#### Optionnel
- `collimation_distance` (m)  
  → peut être **calculée** à partir des transformations des apertures dérivées (voir §7.4)
- `/aperture1` et `/aperture2` (snapshots optionnels)

### 7.2 Chaîne d’éléments `element_order` + `elements`
Structure :

```
/entry/instrument/collimation
  element_order = ["A1", "G1", "A2", ...]     (requis)

  /elements (NXcollection)                   (requis)
    /A1 (NXslit | NXpinhole | NXaperture)
      /transformations (NXtransformations)
        translation = [x,y,z]   (m)           (requis)
      x_gap, y_gap (m)          si NXslit
      diameter (m)              si NXpinhole

    /G1 (NXguide)
      state = "in" | "out"      (requis)
      /transformations/translation (m) (requis)
```

**Règles v1.3** :
- Chaque élément de `element_order` doit exister sous `/elements/<name>`.
- Chaque élément sous `/elements/<name>` doit avoir :
  - `NX_class` dans : `{NXslit, NXpinhole, NXaperture, NXguide}`
  - `/transformations/translation` (vecteur, en mètres)
- Pour `NXguide` : dataset `state` requis, valeurs autorisées : `"in"` ou `"out"`.
- Pour `NXslit` : datasets `x_gap` et `y_gap` requis (m).
- Pour `NXpinhole` : dataset `diameter` requis (m).

### 7.3 Déduction de `aperture2` (à partir de `element_order`)
On définit les **apertures** comme tout élément dont `NX_class ∈ {NXslit, NXpinhole, NXaperture}`.

- **aperture2** = **dernière** aperture rencontrée dans `element_order`  
  (donc la plus proche de l’échantillon, en termes d’ordre du faisceau).

### 7.4 Déduction de `aperture1` (à partir de `element_order` + `NXguide/state`)
Objectif : repérer la fente correspondant au point où le guide “rentre” dans le faisceau.

Règle recommandée :
- trouver le **premier** `NXguide` avec `state="in"` (ou la première transition `"out" → "in"` si on veut être plus strict),
- `aperture1` = l’aperture la plus proche **en amont** dans `element_order` (en remontant dans la liste).

En cas d’ambiguïté (pas de guide “in”, pas d’aperture en amont), le logiciel peut :
- prendre une stratégie de repli (nearest downstream),
- et émettre un warning.

### 7.5 Calcul possible de `collimation_distance`
Une fois `aperture1` et `aperture2` identifiées, on peut calculer :

- `collimation_distance = z(aperture2) − z(aperture1)`

où `z()` est la composante z de `/transformations/translation` de chaque aperture.

> Remarque : c’est pourquoi `translation` est requis pour tous les éléments.

---

## 8. Détecteurs (NXdetector)

Le format supporte plusieurs détecteurs : `detector0`, `detector1`, …

Chaque détecteur doit contenir **au minimum** :

### Champs requis
- `data` (counts bruts)
- `x_pixel_size` (m)
- `y_pixel_size` (m)
- `beam_center_x` (pixel)
- `beam_center_y` (pixel)
- `dead_time` (s)
- `/transformations/translation` (m)

### Champs optionnels
- `data_errors`
- `local_name` (ex. nom instrument)
- `type` (type détecteur)
- `pixel_mask` + `pixel_mask_applied`
- `countrate_correction_applied`
- `flatfield` + `flatfield_applied`
- `efficiency` (+ `wavelength` si efficacité dépendante de λ)
- offsets pixel : `x_pixel_offset`, `y_pixel_offset`, `z_pixel_offset`
- `distance` (m) (si fourni séparément)

---

## 9. Moniteurs (NXmonitor)

Deux emplacements possibles :
- `/entry/control` (NXmonitor) : optionnel
- `/entry/instrument/monitorN` (NXmonitor) : optionnels

Règles minimales :
- datasets requis : `mode`, `preset`
- et au moins l’un de : `integral` **ou** `data`
- `mode ∈ {"monitor","timer"}`

---

## 10. Vues `NXdata` (NXlink)

Pour chaque détecteur `detectorN`, on fournit un groupe :

- `/entry/dataN` (NXdata) — **requis**

Avec :
- `counts` : lien vers `/entry/instrument/detectorN/data`
- `counts_errors` : lien vers `/entry/instrument/detectorN/data_errors` (si disponible)

Attributs recommandés sur `NXdata` :
- `@signal = "counts"`
- `@axes = ...`

---

## 11. Profil “mono”
Cette version (v1.3 mono) impose :
- `NXmonochromator/wavelength` (monochromatique)
- pas de structure TOF (pas de frames/choppers obligatoires)

---

## 12. Checklist “minimum valid file” (v1.3)

Pour être validé au minimum :
- `/entry/definition="NXsas_raw"`
- `/entry/sample` (NXsample)
- `/entry/instrument` (NXinstrument)
- `/entry/instrument/monochromator/wavelength`
- `/entry/instrument/collimation/element_order`
- `/entry/instrument/collimation/last_aperture_to_sample_distance`
- `/entry/instrument/collimation/elements/*/transformations/translation`
- `NXguide/state` si des guides existent dans `elements`
- au moins un détecteur `detector0` avec les champs requis
- au moins `data0` (NXdata) pointant sur `detector0/data`

---

## 13. Notes d’implémentation (conseillées)

- Les convertisseurs (instrument → SCARLET) devraient toujours remplir :
  - `element_order`
  - `elements/<name>/transformations/translation`
  - `NXguide/state` (“in/out”)
  - `last_aperture_to_sample_distance` (issu de l’instrument d’origine)
- Les étapes de workflow peuvent ensuite dériver automatiquement :
  - `aperture1`, `aperture2`
  - `collimation_distance` (et éventuellement l’écrire si on veut un “snapshot”).

