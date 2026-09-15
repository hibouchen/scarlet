# Jupyter tutorial

The bundled tutorial provides a hands-on introduction to SCARLET data and tools.

## Open the tutorial

From a SCARLET installation:

```bash
scarlet notebook
```

This command opens the graphical launcher and lets you create an editable copy of the tutorial notebook. A destination can also be provided for scripted use:

```bash
scarlet notebook tutorial_sessions
```

The following direct entry point is equivalent:

```bash
scarlet-notebook
```

## Working principles

Keep experimental data outside the repository and work on notebook copies. By convention, the project examples use this layout:

```text
data/<instrument>/raw/
data/<instrument>/processed/
```

The `data/` directory is ignored by Git to avoid versioning raw files and local reduction outputs.
