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
