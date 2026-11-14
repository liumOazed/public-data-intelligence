"""The source registry: one entry per source, read by collectors, exports and the site."""

from __future__ import annotations

from functools import lru_cache

import yaml

from pdi.common.paths import REGISTRY

REQUIRED = (
    "name",
    "url",
    "owner",
    "licence",
    "frequency",
    "typical_lag",
    "status",
    "terms_checked",
)
STATUSES = ("green", "amber", "red")


@lru_cache
def load() -> dict[str, dict]:
    sources = yaml.safe_load(REGISTRY.read_text())["sources"]
    for key, src in sources.items():
        missing = [f for f in REQUIRED if f not in src]
        if missing:
            raise ValueError(f"registry entry {key!r} is missing {missing}")
        if src["status"] not in STATUSES:
            raise ValueError(f"registry entry {key!r} has unknown status {src['status']!r}")
    return sources


def get(key: str) -> dict:
    return load()[key]
