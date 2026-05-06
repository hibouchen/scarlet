# Format de données générique (base SCARLET)

Objectif : définir un conteneur NeXus/HDF5 **proche de NXsas**, mais avec une description de **collimation** plus exploitable pour la réduction (multi-éléments, positions, ouvertures).

Ce document décrit une *convention SCARLET* (profil) : on conserve l’esprit de NXsas (groupes `entry/`, `instrument/`, `sample/`, `data/`) tout en ajoutant une modélisation de collimation plus riche.

## Conventions générales

- Origine : centre de l’échantillon
- Axes : `+z` aval (downstream), `+x` horizontal (beam-right), `+y` vertical (up)
- Les éléments en amont de l’échantillon ont une position `z < 0`

## Organisation minimale (profil “raw”)

Chemins (vue simplifiée) :

```
/entry (NXentry)
  definition = "NXsas_raw"            # convention SCARLET pour les données brutes
  /sample (NXsample)
  /instrument (NXinstrument)
    /geometry                         # groupe existant SCARLET (réservé)
    /collimation (NXcollection)       # extension SCARLET (détaillée ci-dessous)
    /detector
  /control (NXmonitor)   
  /data
    /data0/data              # image 2D brute (SCARLET v0.1)
```

Remarque : l’implémentation actuelle converge vers le profil `scarlet_nxsas_raw_v1.3_mono.yaml` : les données brutes sont stockées dans les groupes `NXdetector` (`/entry/instrument/detector0`, `/entry/instrument/detector1`, ...) et exposées via des groupes `NXdata` (`/entry/data0`, `/entry/data1`, ...) contenant des liens vers les détecteurs.

## Collimation (extension SCARLET)

Dans NXsas, la collimation est souvent résumée par un seul objet “collimator” (et/ou une longueur de collimation). Pour SANS, cela ne suffit pas : la réduction a besoin de connaître les **éléments successifs** (guides, diaphragmes/pinhole, slits, sollers, etc.), leurs **positions** et leurs **ouvertures**.

### Principe

Créer un groupe :

```
/entry/instrument/collimation (NXcollection)
```

Il contient une suite d’éléments de collimation, chacun étant un groupe NeXus avec un `NX_class` parlant (par ex. `NXslit`, `NXpinhole`, `NXguide`, `NXcollimator`).

### Exemple (arborescence)

Exemple typique SANS “2 slits + pinhole + guide” (noms libres) :

```
/entry/instrument/collimation (NXcollection)
  order = ["slit_1", "guide_1", "slit_2", "pinhole_1"]

  /slit_1 (NXslit)
    distance = -8.0            # m (z par rapport au centre échantillon)
    x_gap = 0.010              # m
    y_gap = 0.010              # m

  /guide_1 (NXguide)
    distance = -6.0            # m
    # paramètres selon instrument (ex: section, revêtement, etc.)

  /slit_2 (NXslit)
    distance = -2.5            # m
    x_gap = 0.006              # m
    y_gap = 0.006              # m

  /pinhole_1 (NXpinhole)
    distance = -1.2            # m
    diameter = 0.008           # m
```

Dans cet exemple :

- tous les éléments sont **en amont** de l’échantillon (distances négatives),
- l’ordre est explicité via `order` (optionnel) ; sinon, on peut reconstituer une séquence via le tri sur `distance`,
- les champs d’ouverture (`x_gap`, `y_gap`, `diameter`) sont en mètres.

### Position des éléments

Pour rendre le format utilisable rapidement (sans exiger immédiatement une chaîne complète `NXtransformations`), SCARLET introduit la convention suivante :

- Chaque élément de collimation **doit** contenir un champ scalaire `distance` (float) en mètres, représentant la position **le long de z** par rapport au centre échantillon.
  - `distance < 0` : en amont (avant l’échantillon)
  - `distance > 0` : en aval

Le champ `distance` est compatible avec une transition future vers `depends_on` / `NXtransformations` (qui sera la représentation la plus “NeXus-pure”).

### Champs recommandés par type

Les champs exacts dépendent des instruments, mais on recommande :

- `NXslit` : `x_gap`, `y_gap` (en mètres) + `distance`
- `NXpinhole` : `diameter` (mètres) + `distance`
- `NXguide` : `state` ("in" | "out") + géométrie/section (selon vos besoins) + `distance`
- `NXcollimator` : paramètres de soller/divergence (si disponible) + `distance`

### Ordre

L’ordre peut être :

- implicite via `distance` (tri croissant),
- ou explicite via un dataset optionnel `order` dans `instrument/collimation` (liste de noms de groupes).

### Exemple (création en h5py)

Extrait Python (création uniquement de la collimation) :

```python
import h5py

with h5py.File("mon_fichier.nxs", "a") as f:
    inst = f["entry/instrument"]
    coll = inst.require_group("collimation")
    coll.attrs["NX_class"] = "NXcollection"

    # Optionnel mais pratique pour figer la séquence
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

## Produits de réduction dans le même fichier

Objectif prévu : écrire les résultats réduits sous forme d’une **nouvelle entry** `NXsas` dans le même fichier (multi-entry), afin de relancer la réduction sans multiplier les fichiers.

La commande de réduction haut niveau, par exemple `scarlet reduce`, n’est pas encore implémentée dans l’état actuel du dépôt. Les commandes disponibles couvrent aujourd’hui la conversion, la validation et la génération des lots de références `refs_sub` / `refs_norm`.
