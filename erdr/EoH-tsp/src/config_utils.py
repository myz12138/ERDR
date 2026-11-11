from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix == ".json":
        obj = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "PyYAML is required to read YAML configs. Install pyyaml or use JSON config files."
            ) from e
        obj = yaml.safe_load(text)
    if not isinstance(obj, dict):
        raise ValueError(f"Config file must parse to a mapping: {p}")
    return obj
