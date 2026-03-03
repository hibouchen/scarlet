from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_schema(schema: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a schema YAML.

    - If `schema` is an existing path on disk, load it.
    - Otherwise, treat it as a filename inside the `scarlet.schemas` package.
    """
    p = Path(schema)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        schema_file = resources.files("scarlet.schemas").joinpath(str(schema))
        with schema_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

    if not isinstance(data, dict) or "nodes" not in data:
        raise ValueError("Invalid schema YAML: expected a mapping with a top-level 'nodes' key.")

    return data