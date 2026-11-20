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

A product is **tracked** when Value ranking or Fair price shows it (it has a retail price that
day). Every tracked product also gets:

  measured_value / measured_metric / measured_n   any SPEC measurement, even one disclosure
  assessment                                      Assessed, Measured but not assessed, or No
                                                  measurement (model eligibility, kept apart
                                                  from the measurement itself)
  next_source                                     where a missing measurement could come from

``history_rows`` and ``first_seen`` keep a dated count of tracked and assessed products so the
page can show coverage over time and mark products that entered the catalogue recently.
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

# Workstation lines that SPEC GWPG covers. "rtx a\d+" (not "rtx a\d") so that the RTX A400 and
# A4500 are recognised as workstation cards rather than consumer ones.
_WORKSTATION = re.compile(
    r"\b(quadro|rtx a\d+|rtx pro|ada|radeon pro|radeon ai pro|firepro|arc pro|t\d{3,4})\b", re.I
)

ASSESSED = "Assessed"
MEASURED_ONLY = "Measured, not assessed"
NO_MEASUREMENT = "No measurement"

CPU_METRIC = "SPECrate 2017 Int base per socket"
GPU_RANKED_METRIC = "Calibrated SPEC GWPG score (site estimate)"

# OpenBenchmarking.org answers every automated request with a Cloudflare challenge, from a
# cloud sandbox and from a home PC alike. The collectors do nothing to get past it.
OPENBENCHMARKING_BLOCKED = (
    "OpenBenchmarking.org blocks automated clients (Cloudflare challenge, checked 2026-11-20)"
)


# Sources checked on 2026-11-20 and not collected (see sources/registry.yaml).
NOT_COLLECTED = (
    "Geekbench, PassMark and TechPowerUp do not allow reuse of their results, and "
    "OpenBenchmarking.org blocks automated clients"
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


def _num(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


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
    ws_cpu: pd.DataFrame | None = None,
    blender_rank: pd.DataFrame | None = None,
    blender_spec_keys: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """``blender_rank`` holds the Blender residual rows of the kinds whose Blender model passed
    its checks (see blender_model.passes); ``blender_spec_keys`` the model keys per kind that
    have specifications for it, so a missing one can be named."""
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
                "spec_rows_combinable": None,
                "score_per_dollar": None,
                "value_rank": None,
                "value_n": None,
                "fair_price_usd": None,
                "fair_gap_pct": None,
                "spec_score": None,
                "residual_pct": None,
                "consensus_verdict": None,
                "blender_runs": None,
                "measured_value": None,
                "measured_metric": None,
                "measured_n": 0,
                "measured_groups": 0,
                "ws_cpu_results": 0,
                "ws_cpu_score": None,
                "ws_cpu_metric": None,
                "on_blender_residuals": False,
                "blender_residual_pct": None,
                "blender_predicted": None,
                "blender_verdict": None,
            },
        )
        # Prefer a SPEC or retail display name over a bare model key.
        if r["name"] == key and name != key:
            r["name"] = name
        return r

    for b in benchmarks.itertuples():
        r = row(b.kind, b.model_key, getattr(b, "device_name", None) or b.model_key)
        r["blender_score"] = float(b.median_score)
        runs = getattr(b, "n_runs", None)
        r["blender_runs"] = None if runs is None or pd.isna(runs) else int(runs)
    for v in value.itertuples():
        r = row(v.kind, v.model_key, v.model_key)
        r["on_value_ranking"] = True
        r["price_usd"] = float(v.price_usd)
        r["price_date"] = _day(getattr(v, "price_date", None)) or _day(
            getattr(v, "collected_date", None)
        )
    # Rank by Blender points per dollar, as the Value ranking page does.
    if "score_per_dollar" in value.columns and not value.empty:
        ranked = value.assign(
            rank_=value.groupby("kind")["score_per_dollar"].rank(ascending=False, method="min"),
            n_=value.groupby("kind")["model_key"].transform("size"),
        )
        for v in ranked.itertuples():
            r = row(v.kind, v.model_key, v.model_key)
            r["score_per_dollar"] = float(v.score_per_dollar)
            r["value_rank"] = int(v.rank_)
            r["value_n"] = int(v.n_)
    for f in fair.itertuples():
        r = row(f.kind, f.model_key, f.model_key)
        r["on_fair_price"] = True
        if getattr(f, "fair_price_usd", None) is not None and pd.notna(f.fair_price_usd):
            r["fair_price_usd"] = float(f.fair_price_usd)
            r["fair_gap_pct"] = float(f.gap_pct)

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
        # Every result counts as a measurement, ranked or not: median base score per socket,
        # the same quantity the ranking uses.
        if {config.SPEC_METRIC, "chips"} <= set(spec.columns):
            per_socket = spec[config.SPEC_METRIC] / spec["chips"].where(spec["chips"] > 0)
            med = per_socket.groupby(spec["processor"]).median()
            for proc, v in med.items():
                if pd.isna(v):
                    continue
                r = row("cpu", key_of("cpu", proc), proc)
                r["measured_value"] = float(v)
                r["measured_metric"] = CPU_METRIC
                r["measured_n"] = r["spec_results"]
                r["measured_groups"] = 1
    # SPECworkstation CPU subsystem scores: a second, separate CPU measurement. It fills
    # measured_value only where SPEC CPU 2017 has none, and is never used by the CPU models.
    if ws_cpu is not None and not ws_cpu.empty:
        single = (
            ws_cpu[~ws_cpu["multi_socket"].astype(bool)] if "multi_socket" in ws_cpu else ws_cpu
        )
        for key, part in single.groupby(single["processor"].map(lambda p: key_of("cpu", p))):
            m = _ws_measurement(part)
            if m is None:
                continue
            r = row("cpu", key, str(part["processor"].iloc[0]))
            # Results in the version the score comes from; measured_n counts every version.
            r["ws_cpu_results"] = int((part["suite"] == m[4]).sum())
            r["ws_cpu_score"], r["ws_cpu_metric"] = m[0], m[1]
            if not r["measured_n"]:
                r["measured_value"], r["measured_metric"], r["measured_n"] = m[0], m[1], m[2]
                r["measured_groups"] = m[3]
    for c in cpu_rank.itertuples():
        r = row("cpu", key_of("cpu", c.processor), c.processor)
        r["on_residuals"] = True
        r["residual_pct"] = float(c.residual_pct)
        r["consensus_verdict"] = c.consensus_verdict
        r["spec_score"] = _num(getattr(c, "measured_per_socket", None))
        if r["spec_score"] is not None:
            r["measured_value"] = r["spec_score"]
            r["measured_metric"] = CPU_METRIC

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
            if "in_ranked_suites" in part.columns:
                r["spec_rows_combinable"] = int((ok["in_ranked_suites"]).sum()) if len(ok) else 0
            m = _gwpg_measurement(ok)
            if m:
                r["measured_value"], r["measured_metric"], r["measured_n"], r["measured_groups"] = m
    for c in gpu_rank.itertuples():
        r = row("gpu", key_of("gpu", c.processor), c.processor)
        r["on_residuals"] = True
        r["residual_pct"] = float(c.residual_pct)
        r["consensus_verdict"] = c.consensus_verdict
        r["spec_score"] = _num(getattr(c, "measured_score", None))
        if r["spec_score"] is not None:
            r["measured_value"] = r["spec_score"]
            r["measured_metric"] = GPU_RANKED_METRIC
            r["measured_n"] = r["spec_results"]

    # Blender rendering residuals: a separate model on a separate benchmark, never mixed with
    # the SPEC ones above.
    if blender_rank is not None and not blender_rank.empty:
        for b in blender_rank.itertuples():
            r = row(b.kind, b.model_key, b.model_key)
            r["on_blender_residuals"] = True
            r["blender_residual_pct"] = float(b.residual_pct)
            r["blender_predicted"] = _num(getattr(b, "pred_ols", None))
            r["blender_verdict"] = b.consensus_verdict

    out = pd.DataFrame(rows.values())
    if out.empty:
        return out
    status, reason = zip(*(_status(r) for r in out.to_dict("records")), strict=True)
    out["coverage_status"] = status
    out["reason"] = reason
    out["tracked"] = out["on_value_ranking"] | out["on_fair_price"]
    out["assessed_by"] = [
        " and ".join(
            s
            for s, on in (
                (SPEC_SOURCE[r["kind"]], r["on_residuals"]),
                (BLENDER_SOURCE, r["on_blender_residuals"]),
            )
            if on
        )
        for r in out.to_dict("records")
    ]
    out["assessment"] = [
        ASSESSED
        if r["on_residuals"] or r["on_blender_residuals"]
        else MEASURED_ONLY
        if r["measured_n"] or _has(r["blender_score"])
        else NO_MEASUREMENT
        for r in out.to_dict("records")
    ]
    keys = blender_spec_keys or {}
    out["blender_reason"] = [_blender_reason(r, keys) for r in out.to_dict("records")]
    out["next_source"] = [_next_source(r) for r in out.to_dict("records")]
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


SPEC_SOURCE = {"cpu": "SPEC CPU 2017", "gpu": "SPEC GWPG"}
BLENDER_SOURCE = "Blender Open Data"


def _has(v) -> bool:
    return v is not None and not (isinstance(v, float) and pd.isna(v))


def _blender_reason(r: dict, spec_keys: dict[str, set[str]]) -> str:
    """Why the Blender model does not assess a product; empty when it does."""
    if r["on_blender_residuals"]:
        return ""
    if not _has(r["blender_score"]):
        return "No Blender Open Data result with enough runs in the version used"
    if r["kind"] not in spec_keys:
        return "Blender model not validated for this kind this week"
    if r["model_key"] not in spec_keys[r["kind"]]:
        return "Blender result, but no specifications in the repository's table yet"
    return "Blender result not in the ranked set"


def _status(r: dict) -> tuple[str, str]:
    min_n = config.MIN_RESULTS_FOR_RANK if r["kind"] == "cpu" else config.GWPG_MIN_RESULTS_FOR_RANK
    n, oems = int(r["spec_results"]), int(r["spec_oems"])
    if r["on_residuals"]:
        if _limited(n, oems):
            return LIMITED, (
                f"Insufficient disclosures: assessed on {n} result{'s' if n != 1 else ''} from "
                f"{oems} OEM{'s' if oems != 1 else ''}; Measured needs 4 from 2 or more OEMs"
            )
        return MEASURED, ""
    combinable = r.get("spec_rows_combinable")
    if n > 0 and combinable == 0:
        return LIMITED, "Only in suites without a validated calibration (see results by workload)"
    if n > 0 and combinable is not None and not pd.isna(combinable) and combinable < min_n:
        # The ranking counts only results from suites whose calibration passed, so the total can
        # exceed the minimum while the usable count does not.
        return LIMITED, (
            "Insufficient disclosures in suites with a validated calibration "
            f"({int(combinable)} of {min_n} needed; {n - int(combinable)} more in other suites)"
        )
    if n > 0:
        return LIMITED, f"Insufficient disclosures ({n} of {min_n} needed)"
    if r["spec_rows_unmatched"] > 0:
        return MISSING_SPECS, "Missing specifications"
    if r["spec_rows_unparsed"] > 0:
        return PARSE_FAIL, "Parsing failure"
    if r["kind"] == "gpu" and not _is_workstation(r):
        # SPEC GWPG results are workstation submissions; consumer cards are rarely submitted.
        return NOT_FOUND, (
            "Unsupported source (consumer card: SPEC GWPG is workstation-only, and "
            "OpenBenchmarking.org blocks automated collection)"
        )
    if r["kind"] == "gpu":
        return NOT_FOUND, "Missing measurements (no SPEC GWPG result published for this card)"
    if r.get("ws_cpu_results"):
        return NOT_FOUND, (
            f"Different benchmark only: {r['ws_cpu_results']} {r['ws_cpu_metric']} "
            f"result{'s' if r['ws_cpu_results'] != 1 else ''}, no SPEC CPU 2017 result; the CPU "
            "models use SPECrate 2017 Integer, so it is shown but not assessed"
        )
    return NOT_FOUND, "Missing measurements (no SPEC CPU 2017 result published for this processor)"


def _is_workstation(r: dict) -> bool:
    return bool(_WORKSTATION.search(r["name"]) or _WORKSTATION.search(r["model_key"]))


def _next_source(r: dict) -> str:
    """Where a missing or thin measurement could come from; empty once a product is assessed."""
    if r["on_residuals"] or r["on_blender_residuals"]:
        return ""
    blender = r.get("blender_reason") or ""
    if blender.startswith("Blender result, but no specifications"):
        blender = "Blender Open Data, once its specifications are added"
    elif blender:
        blender = f"Blender Open Data ({blender[0].lower()}{blender[1:]})"
    if r["kind"] == "cpu":
        spec = "SPEC CPU 2017 and SPECworkstation rechecked weekly"
    elif _is_workstation(r):
        spec = "SPEC GWPG rechecked weekly"
    else:
        spec = ""
    return "; ".join(x for x in (blender, spec, NOT_COLLECTED) if x)


def _gwpg_measurement(ok: pd.DataFrame) -> tuple[float, str, int, int] | None:
    """A card's published composite in the suite, OS and resolution where it has most results.

    Raw composites are comparable only within one suite and version, OS and resolution, so the
    measurement names that group. Ties go to the newer suite (higher median score is not used,
    so the choice does not depend on how well the card did).
    """
    need = {"suite", "os_family", "res_tier", "composite_score"}
    if ok.empty or not need <= set(ok.columns):
        return None
    d = ok.dropna(subset=["composite_score"])
    if d.empty:
        return None
    groups = d.groupby(["suite", "os_family", "res_tier"])["composite_score"].agg(
        ["size", "median"]
    )
    groups = groups.reset_index().sort_values(["size", "suite"], ascending=[False, False])
    top = groups.iloc[0]
    metric = f"{top['suite']} composite · {top['os_family']} · {top['res_tier']}"
    return float(top["median"]), metric, int(len(d)), int(len(groups))


def _ws_measurement(part: pd.DataFrame) -> tuple[float, str, int, int, str] | None:
    """Median CPU subsystem score in the SPECworkstation version with most results.

    Versions are different benchmarks, so the median is taken within one. Ties go to the newer
    version, so the choice does not depend on the scores.
    """
    d = part.dropna(subset=["score"])
    if d.empty:
        return None
    groups = d.groupby("suite")["score"].agg(["size", "median"]).reset_index()
    top = groups.sort_values(["size", "suite"], ascending=[False, False]).iloc[0]
    metric = str(d.loc[d["suite"] == top["suite"], "metric"].iloc[0])
    return float(top["median"]), metric, int(len(d)), int(len(groups)), str(top["suite"])


def _flag(t: pd.DataFrame, col: str) -> pd.Series:
    if col in t.columns:
        return t[col].fillna(False).astype(bool)
    return pd.Series(False, index=t.index, dtype=bool)


def tracked_summary(cat: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per kind: tracked products, and how many are assessed, measured only, or unmeasured."""
    out: dict[str, dict[str, int]] = {}
    for kind in ("cpu", "gpu"):
        t = cat[(cat["kind"] == kind) & cat["tracked"]] if not cat.empty else cat
        a = t["assessment"] if len(t) else pd.Series(dtype=str)

        spec_on, bl_on = _flag(t, "on_residuals"), _flag(t, "on_blender_residuals")
        out[kind] = {
            "tracked": int(len(t)),
            "assessed": int((a == ASSESSED).sum()),
            "assessed_spec": int(spec_on.sum()),
            "assessed_blender": int(bl_on.sum()),
            "assessed_spec_only": int((spec_on & ~bl_on).sum()),
            "measured_only": int((a == MEASURED_ONLY).sum()),
            "no_measurement": int((a == NO_MEASUREMENT).sum()),
            "catalogue": int((cat["kind"] == kind).sum()) if not cat.empty else 0,
        }
    return out


def update_history(
    cat: pd.DataFrame, history: list[dict], first_seen: dict[str, str], today: str
) -> tuple[list[dict], dict[str, str]]:
    """Add today's coverage counts (replacing an earlier count from the same day) and record
    the first day each tracked product appeared. Returns the new history and first-seen map."""
    rows = [h for h in history if h.get("date") != today]
    for kind, c in tracked_summary(cat).items():
        rows.append({"date": today, "kind": kind, **c})
    rows.sort(key=lambda h: (h["date"], h["kind"]))
    seen = dict(first_seen)
    if not cat.empty:
        for r in cat[cat["tracked"]].itertuples():
            seen.setdefault(f"{r.kind}|{r.model_key}", today)
    return rows, seen
