from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}

    parent_name = config.pop("extends", None)
    if parent_name is None:
        return config

    parent_config = load_config(path.parent / parent_name)
    return deep_merge(parent_config, config)
