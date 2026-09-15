# Installation

SCARLET requires Python 3.11 or later.

## Install from the repository

```bash
git clone https://github.com/hibouchen/scarlet.git
cd scarlet
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows, activate the environment with:

```bat
.venv\Scripts\activate.bat
```

Then verify the installation:

```bash
scarlet --help
```

## Development installation

Install the test dependencies and run the test suite:

```bash
pip install -e .[dev]
pytest -q
```

## Build this documentation

The documentation tools are optional and do not affect a regular user installation:

```bash
pip install -e .[docs]
mkdocs serve
```

The preview server prints its local address in the terminal. To validate a build without serving the site:

```bash
mkdocs build --strict
```

## Isolated installation with pipx

To install the commands in a dedicated environment:

```bash
pipx install git+https://github.com/hibouchen/scarlet.git
```

The `scarlet`, `scarlet-notebook`, and `viewer` executables are then available from the terminal.
