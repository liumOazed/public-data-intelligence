"""One hardware catalogue for the three Tech Value pages, with a coverage status per product.

Every CPU and GPU that appears anywhere (a Blender benchmark, a retail price, a SPEC CPU 2017
result or a SPEC GWPG disclosure) gets one row keyed by the same model key the Value ranking and
Fair price pages use. Each row says which pages show the product, and, for the over- and
under-performer page, why it is or is not ranked:

  Measured                 ranked, with several results from more than one OEM
  Limited evidence         ranked, but on few results or a single OEM, or below the minimum count
  Missing specifications   SPEC results exist, but the card is not in the specification table
  Parsing failure          a SPEC row names the product, but no score could be read from it
  Results not found        no SPEC result for the product in the collected suites

``reason`` says which of the five causes applies: missing measurements, unsupported source,
parsing failure, missing specifications or insufficient disclosures.
"""

from __future__ import annotations

import re

import pandas as pd

from pdi.tech_value import config, names

MEASURED = "Measured"
LIMITED = "Limited evidence"
MISSING_SPECS = "Missing specifications"
PARSE_FAIL = "Parsing failure"
NOT_FOUND = "Results not found"

_WORKSTATION = re.compile(
    r"\b(quadro|rtx a\d|rtx pro|ada|radeon pro|radeon ai pro|firepro|arc pro|t\d{3,4})\b", re.I
)


def key_of(kind: str, name: str) -> str:
    """The model key the other pages use, or a normalised name when they have none."""
    hit = names.model_key(name)
    if hit and hit[0] == kind:
        return hit[1]
    text = names.normalise(name)
    text = re.sub(r"\b(geforce|processor|graphics)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _day(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)[:10]


def _limited(n_results: int, n_oems: int) -> bool:
    return n_oems <= 1 or n_results < 4


def build(
    *,
    value: pd.DataFrame,
    fair: pd.DataFrame,
    benchmarks: pd.DataFrame,
    cpu_rank: pd.DataFrame,
    gpu_rank: pd.DataFrame,
    spec: pd.DataFrame,
    gwpg: pd.DataFrame,
    dates: dict[str, str | None],
) -> pd.DataFrame:
    rows: dict[tuple[str, str], dict] = {}

    def row(kind: str, key: str, name: str) -> dict:
        r = rows.setdefault(
            (kind, key),
            {
                "kind": kind,
                "model_key": key,
                "name": name,
                "on_value_ranking": False,
                "on_fair_price": False,
                "on_residuals": False,
                "blender_score": None,
                "price_usd": None,
                "price_date": None,
                "spec_results": 0,
                "spec_oems": 0,
                "spec_rows_unmatched": 0,
                "spec_rows_unparsed": 0,
                "residual_pct": None,
                "consensus_verdict": None,
            },
        )
        # Prefer a SPEC or retail display name over a bare model key.
        if r["name"] == key and name != key:
            r["name"] = name
        return r

    for b in benchmarks.itertuples():
        r = row(b.kind, b.model_key, getattr(b, "device_name", None) or b.model_key)
        r["blender_score"] = float(b.median_score)
    for v in value.itertuples():
        r = row(v.kind, v.model_key, v.model_key)
        r["on_value_ranking"] = True
        r["price_usd"] = float(v.price_usd)
        r["price_date"] = _day(getattr(v, "price_date", None)) or _day(
            getattr(v, "collected_date", None)
        )
    for f in fair.itertuples():
        row(f.kind, f.model_key, f.model_key)["on_fair_price"] = True

    # SPEC CPU: every processor with a result, ranked or not.
    if not spec.empty:
        counts = spec.groupby("processor").agg(
            n=("processor", "size"), oems=("hardware_vendor", "nunique")
        )
        for proc, c in counts.iterrows():
            r = row("cpu", key_of("cpu", proc), proc)
            r["name"] = proc  # SPEC's name is the most precise one
            r["spec_results"] += int(c["n"])
            r["spec_oems"] = max(r["spec_oems"], int(c["oems"]))
    for c in cpu_rank.itertuples():
        r = row("cpu", key_of("cpu", c.processor), c.processor)
        r["on_residuals"] = True
        r["residual_pct"] = float(c.residual_pct)
        r["consensus_verdict"] = c.consensus_verdict

    # SPEC GWPG: every disclosure, including cards without specifications or readable scores.
    if not gwpg.empty:
        status = gwpg["spec_status"] if "spec_status" in gwpg.columns else "ok"
        g = gwpg.assign(spec_status=status)
        for proc, part in g.groupby("processor"):
            r = row("gpu", key_of("gpu", proc), proc)
            r["name"] = proc
            ok = part[part["spec_status"] == "ok"]
            r["spec_results"] += int(len(ok))
            r["spec_oems"] = max(r["spec_oems"], int(ok["oem"].nunique()) if len(ok) else 0)
            r["spec_rows_unmatched"] += int((part["spec_status"] == "missing").sum())
            r["spec_rows_unparsed"] += int((part["spec_status"] == "unparsed").sum())
    for c in gpu_rank.itertuples():
        r = row("gpu", key_of("gpu", c.processor), c.processor)
        r["on_residuals"] = True
        r["residual_pct"] = float(c.residual_pct)
        r["consensus_verdict"] = c.consensus_verdict

    out = pd.DataFrame(rows.values())
    if out.empty:
        return out
    status, reason = zip(*(_status(r) for r in out.to_dict("records")), strict=True)
    out["coverage_status"] = status
    out["reason"] = reason
    out["benchmarks_collected"] = out["kind"].map(
        {"cpu": dates.get("spec_cpu2017"), "gpu": dates.get("spec_gwpg")}
    )
    out["blender_collected"] = dates.get("blender_opendata")
    out["spec_source"] = out["kind"].map(
        {
            "cpu": "SPEC CPU 2017 disclosures",
            "gpu": "Specification table in pdi/tech_value/gwpg.py",
        }
    )
    order = {MEASURED: 0, LIMITED: 1, MISSING_SPECS: 2, PARSE_FAIL: 3, NOT_FOUND: 4}
    return (
        out.assign(_o=out["coverage_status"].map(order))
        .sort_values(["kind", "_o", "name"])
        .drop(columns="_o")
        .reset_index(drop=True)
    )


def _status(r: dict) -> tuple[str, str]:
    min_n = config.MIN_RESULTS_FOR_RANK if r["kind"] == "cpu" else config.GWPG_MIN_RESULTS_FOR_RANK
    if r["on_residuals"]:
        if _limited(r["spec_results"], r["spec_oems"]):
            return LIMITED, "Insufficient disclosures"
        return MEASURED, ""
    if r["spec_results"] > 0:
        return LIMITED, f"Insufficient disclosures ({r['spec_results']} of {min_n} needed)"
    if r["spec_rows_unmatched"] > 0:
        return MISSING_SPECS, "Missing specifications"
    if r["spec_rows_unparsed"] > 0:
        return PARSE_FAIL, "Parsing failure"
    if (
        r["kind"] == "gpu"
        and not _WORKSTATION.search(r["name"])
        and not _WORKSTATION.search(r["model_key"])
    ):
        # SPEC GWPG results are workstation submissions; consumer cards are rarely submitted.
        return NOT_FOUND, "Unsupported source (consumer card, SPEC GWPG is workstation-only)"
    return NOT_FOUND, "Missing measurements (no SPEC result published)"
