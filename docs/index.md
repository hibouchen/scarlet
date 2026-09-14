# SCARLET

**SCA**ttering **R**eduction and ana**L**ysis **E**nvironmen**T** est un framework Python pour les workflows de reduction de donnees SANS. Il utilise le format NeXus/HDF5 comme format d'echange principal et fournit des outils en ligne de commande, une API Python et des interfaces graphiques d'inspection.

## Ce que SCARLET permet de faire

- convertir des fichiers instrumentaux vers le profil `NXsas_raw` de SCARLET ;
- valider des fichiers NeXus/HDF5 avec les schemas fournis ;
- preparer et inspecter des masques de detecteur ;
- appliquer des briques de reduction, dont l'integration azimutale ;
- exporter une courbe `I(Q)` depuis une sortie reduite ;
- travailler dans un notebook Jupyter avec le tutoriel fourni.

## Demarrage rapide

```bash
git clone https://github.com/hibouchen/scarlet.git
cd scarlet
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
scarlet --help
```

Poursuivez avec la page [Installation](installation.md), puis le [tutoriel](tutorial.md).

## Statut

SCARLET est en cours de developpement. Certaines fonctions de reduction haut niveau evoluent encore ; consultez le [changelog du projet](https://github.com/hibouchen/scarlet/blob/main/CHANGELOG.md) avant de mettre a jour un workflow de production.
