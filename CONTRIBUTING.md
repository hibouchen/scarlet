# Contributing to SCARLET

## Development setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

## Notes
- Keep functions small and deterministic
- Prefer physics-first implementations

## Release checklist
1. Update `pyproject.toml` version when preparing a release.
2. Move relevant entries from `CHANGELOG.md` under `Unreleased` into a dated release section.
3. Run the project test suite and build checks.
4. Commit the release changes.
5. Create a Git tag matching the package version, for example `v0.1.0`.
