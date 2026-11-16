"""What's unusual this week: every module's flagged items on one page, written by code.

Each module's export writes site/data/v1/<module>/findings.json in the shared format of
pdi.common.findings (one row per item its own fixed rule marks as unusual). This step reads
those files and nothing else, then:

1. stops the run if any item is labelled a prediction without a passed out-of-sample test
   (the exit criterion's second half, checked at every build);
2. records when each item was first seen, in data/store/unusual/seen.json, so that a later
   build can tell an item that is new this week from one that has been standing;
3. writes site/data/v1/unusual/week.json, which the page reads.

Items present at the very first build are marked as standing: without an earlier build,
nothing can honestly be called new.

Run as `python -m pdi.unusual`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from pdi.common import findings
from pdi.common.export import write_table
from pdi.common.paths import SITE_DATA, STORE

M = "unusual"
SEEN = STORE / "unusual" / "seen.json"
WINDOW_DAYS = 7
# Page order. The key is the folder under site/data/v1/.
MODULES = {
    "macro": "Macro Monitor",
    "energy": "Energy Demand",
    "trade": "Trade",
    "procurement": "Federal Procurement",
    "tech-value": "Tech Value",
}
FINDING = findings.observation(
    "Each item comes from its module's own fixed rule, written before the page was first "
    "built; the label beside each item is the module's own."
)


class UntestedPrediction(RuntimeError):
    pass


def read_module(site_data: Path, module: str) -> tuple[pd.DataFrame, dict]:
    path = site_data / module / "findings.json"
    if not path.exists():
        return pd.DataFrame(columns=findings.ITEM_COLUMNS), {"status": "not exported yet"}
    payload = json.loads(path.read_text())
    df = pd.DataFrame(payload["data"], columns=payload["meta"]["columns"])
    return df.reindex(columns=findings.ITEM_COLUMNS), payload["meta"]


def check_labels(items: pd.DataFrame) -> None:
    """The build stops on an unknown label, or a prediction without a passed test."""
    bad_label = items[~items["label"].isin(findings.LABELS)]
    if len(bad_label):
        raise UntestedPrediction(f"unknown finding label on: {', '.join(bad_label['key'])}")
    pred = items[items["label"] == findings.PREDICTION]
    untested = pred[pred["test_passed"].map(lambda v: v is not True)]
    if len(untested):
        raise UntestedPrediction(
            "labelled a prediction without a passed out-of-sample test: "
            + ", ".join(untested["key"])
        )


def update_seen(items: pd.DataFrame, seen: dict, now: datetime) -> dict:
    """First-seen time per item key; the first build's items are marked as baseline."""
    first_build = not seen
    out = dict(seen)
    for key in items["key"]:
        if key not in out:
            out[key] = {"first_seen": now.isoformat(), "baseline": first_build}
    return out


def build(site_data: Path, seen: dict, now: datetime) -> tuple[pd.DataFrame, dict, dict]:
    frames, status = [], []
    for module, name in MODULES.items():
        df, meta = read_module(site_data, module)
        status.append({
            "module": module, "name": name, "items": int(len(df)),
            "generated_at": meta.get("generated_at"), "rule": meta.get("rule_text"),
            "status": meta.get("status"),
        })  # fmt: skip
        if len(df):
            frames.append(df.assign(module_name=name))
    cols = [*findings.ITEM_COLUMNS, "module_name"]
    items = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    check_labels(items)
    first_build = not seen
    seen = update_seen(items, seen, now)
    items["first_seen"] = items["key"].map(lambda k: seen[k]["first_seen"])
    since = now - timedelta(days=WINDOW_DAYS)
    items["new"] = [
        (not seen[k]["baseline"]) and datetime.fromisoformat(seen[k]["first_seen"]) >= since
        for k in items["key"]
    ]
    order = {m: i for i, m in enumerate(MODULES)}
    items = items.sort_values(
        ["new", "module", "period"], ascending=[False, True, False],
        key=lambda c: c.map(order) if c.name == "module" else c,
    ).reset_index(drop=True)  # fmt: skip
    meta = {
        "finding": FINDING,
        "window_days": WINDOW_DAYS,
        "first_build": first_build,
        "built_at": now.isoformat(),
        "modules": status,
        "new_items": int(items["new"].sum()),
        "predictions": int((items["label"] == findings.PREDICTION).sum()),
    }
    return items, meta, seen


def main() -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    seen = json.loads(SEEN.read_text()) if SEEN.exists() else {}
    items, meta, seen = build(SITE_DATA, seen, now)
    SEEN.parent.mkdir(parents=True, exist_ok=True)
    SEEN.write_text(json.dumps(seen, indent=1, sort_keys=True) + "\n")
    write_table(
        M, "week", items, sources=[], extra=meta,
        description="Every module's items flagged by its own fixed rule, with when each was "
        "first seen by a build and whether that was in the last seven days.",
    )  # fmt: skip
    print(f"unusual: {len(items)} items, {meta['new_items']} new in the last {WINDOW_DAYS} days")


if __name__ == "__main__":
    main()
