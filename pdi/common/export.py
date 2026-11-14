"""Export step: write the small files the site reads, each with its own label.

Every file under site/data/v1/ has the same envelope:

    {"meta": {...}, "data": [ {column: value, ...}, ... ]}

`meta` carries what rule 7 of the method requires on every number: source, retrieval date,
frequency, typical publication lag, whether the figure is preliminary, and known limitations.
A later API can serve the same envelope from the same paths and the pages will not change.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from pdi.common import registry
from pdi.common.fetch import utc_now
from pdi.common.paths import SITE_DATA


def _clean(value):
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _clean(value.item())
    return value


def write_table(
    module: str,
    table: str,
    df: pd.DataFrame,
    *,
    sources: list[str],
    description: str,
    retrieved_at: dict[str, str | None] | None = None,
    limitations: list[str] | None = None,
    preliminary: bool = False,
    extra: dict | None = None,
    out_dir: Path | None = None,
) -> Path:
    retrieved_at = retrieved_at or {}
    src_meta = []
    for key in sources:
        src = registry.get(key)
        src_meta.append(
            {
                "id": key,
                "name": src["name"],
                "url": src["url"],
                "licence": src["licence"],
                "frequency": src["frequency"],
                "typical_lag": src["typical_lag"],
                "retrieved_at": retrieved_at.get(key),
            }
        )
    payload = {
        "meta": {
            "module": module,
            "table": table,
            "description": description,
            "generated_at": utc_now().isoformat(),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "sources": src_meta,
            "preliminary": preliminary,
            "limitations": limitations or [],
            **(extra or {}),
        },
        "data": [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")],
    }
    path = (out_dir or SITE_DATA) / module / f"{table}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    return path
