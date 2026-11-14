"""Helpers the Quarto pages use. Pages read only from site/data/v1/ and only through here."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from pdi.common import registry
from pdi.common.paths import SITE_DATA


def load(module: str, table: str) -> tuple[pd.DataFrame, dict]:
    """Read one exported table. A table that was never exported is an empty one."""
    path = SITE_DATA / module / f"{table}.json"
    if not path.exists():
        return pd.DataFrame(), {"status": "This table has not been generated yet.", "sources": []}
    payload = json.loads(path.read_text())
    return pd.DataFrame(payload["data"], columns=payload["meta"]["columns"]), payload["meta"]


def _day(stamp: str | None) -> str:
    if not stamp:
        return "not collected yet"
    return datetime.fromisoformat(stamp).strftime("%-d %b %Y")


def source_note(meta: dict) -> str:
    """The label every chart and table carries: source, retrieval date, frequency, lag, limits."""
    lines = []
    for src in meta.get("sources", []):
        lines.append(
            f"**Source:** [{src['name']}]({src['url']}). "
            f"Retrieved {_day(src.get('retrieved_at'))}. "
            f"Frequency: {src['frequency']}. Typical lag: {src['typical_lag']}."
        )
    if meta.get("preliminary"):
        lines.append("**Preliminary figures.**")
    for limit in meta.get("limitations", []):
        lines.append(f"- {limit}")
    body = "\n".join(line if line.startswith("- ") else line + "\n" for line in lines)
    return f"\n::: {{.source-note}}\n{body}\n:::\n"


def empty_state(meta: dict, what: str) -> str:
    reason = meta.get("status") or "No rows yet."
    return f"\n::: {{.callout-note}}\n## {what}: nothing to show yet\n{reason}\n:::\n"


def registry_table() -> pd.DataFrame:
    rows = []
    for key, src in registry.load().items():
        rows.append(
            {
                "Source": f"[{src['name']}]({src['url']})",
                "Owner": src["owner"],
                "Status": src["status"],
                "Frequency": src["frequency"],
                "Typical lag": src["typical_lag"],
                "Licence and reuse": " ".join(str(src["licence"]).split()),
                "Terms last read": src["terms_checked"],
                "Used by": src.get("module", ""),
                "id": key,
            }
        )
    return pd.DataFrame(rows)
