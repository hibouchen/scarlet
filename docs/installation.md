# Installation

SCARLET demande Python 3.11 ou une version plus recente.

## Installation depuis le depot

```bash
git clone https://github.com/hibouchen/scarlet.git
cd scarlet
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Sous Windows, activez l'environnement avec :

```bat
.venv\Scripts\activate.bat
```

Verifiez ensuite l'installation :

```bash
scarlet --help
```

## Installation pour le developpement

Installez les dependances de test, puis executez la suite :

```bash
pip install -e .[dev]
pytest -q
```

## Construire cette documentation

Les outils de documentation sont optionnels et n'alourdissent pas l'installation utilisateur :

```bash
pip install -e .[docs]
mkdocs serve
```

Le serveur de previsualisation affiche l'adresse locale dans le terminal. Pour verifier un build sans servir le site :

```bash
mkdocs build --strict
```

## Installation isolee avec pipx

Pour installer uniquement les commandes dans un environnement dedie :

```bash
pipx install git+https://github.com/hibouchen/scarlet.git
```

Les executables `scarlet`, `scarlet-notebook` et `viewer` deviennent alors accessibles depuis le terminal.
