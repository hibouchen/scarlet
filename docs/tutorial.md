# Tutoriel Jupyter

Le tutoriel embarque un exemple de prise en main des donnees et des outils SCARLET.

## Ouvrir le tutoriel

Depuis une installation de SCARLET :

```bash
scarlet notebook
```

Cette commande ouvre le lanceur graphique et permet de creer une copie modifiable du notebook de tutoriel. Une destination peut aussi etre fournie pour une utilisation scriptable :

```bash
scarlet notebook tutorial_sessions
```

L'entree directe suivante est equivalente :

```bash
scarlet-notebook
```

## Principes de travail

Conservez les donnees experimentales hors du depot et travaillez sur des copies de notebook. Par convention, les exemples du projet utilisent cette organisation :

```text
data/<instrument>/raw/
data/<instrument>/processed/
```

Le repertoire `data/` est ignore par Git pour eviter de versionner les fichiers bruts et les produits de reduction locaux.
