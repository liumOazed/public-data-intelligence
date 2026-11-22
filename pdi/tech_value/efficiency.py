"""Performance per watt: Blender scores against rated power, for site/tech-value/efficiency.qmd.

Every power figure here is a *rated* one (CPU TDP or base power from ``cpu_specs.py``, GPU
board power from ``gpu_specs.py``), so every row carries ``power_kind = "rated"``. Measured
power is a later phase and will arrive as its own rows, never mixed in silently.

The metrics, for a fixed rendering job of ``JOB_SAMPLES`` Blender samples:

  perf per watt   = score / watts                (samples per minute per watt; higher is better)
  seconds per job = JOB_SAMPLES / score * 60
  energy per job  = watts * seconds per job      (joules; lower is better)
  EDP(alpha)      = energy * seconds ** alpha    (lower is better; alpha 1 = EDP, 2 = ED²P)

With a constant rated power, energy per job is just the inverse of perf per watt; EDP is what
adds speed back in, so a slow chip on a low label cannot win on energy alone.

Cost of ownership = price + electricity for the hours under load:
  kWh = watts * hours per day * load share * 365 * years / 1000
The share of time spent idle is not charged, because no idle power figure exists yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdi.tech_value.cpu_specs import CPU_SPECS
from pdi.tech_value.dashboard import display_name
from pdi.tech_value.gpu_specs import GPU_SPECS, NOT_ENTERED

JOB_SAMPLES = 10_000
# The cost calculator's starting values, agreed for the first release.
TCO_DEFAULTS = {"hours_per_day": 4.0, "load_share": 0.30, "years": 3}
# Used only until the BLS series has been collected once; the page says so when it is in use.
FALLBACK_USD_PER_KWH = 0.17
VENDORS = ("amd", "intel", "nvidia")


def rated_power() -> pd.DataFrame:
    """One row per model with a rated power figure: kind, model_key, watts, label, year."""
    rows = [
        ("cpu", key, float(spec[6]), "TDP", int(spec[7])) for key, spec in CPU_SPECS.items()
    ] + [("gpu", key, float(w), "Board power", int(y)) for key, (w, y) in GPU_SPECS.items()]
    return pd.DataFrame(rows, columns=["kind", "model_key", "watts", "power_label", "year"])


def seconds_per_job(score):
    return JOB_SAMPLES / np.asarray(score, dtype=float) * 60.0


def energy_per_job(score, watts):
    return np.asarray(watts, dtype=float) * seconds_per_job(score)


def edp(score, watts, alpha: float = 1.0):
    return energy_per_job(score, watts) * seconds_per_job(score) ** alpha


def tco(price, watts, usd_per_kwh, hours_per_day, load_share, years):
    """Purchase price plus electricity for the hours under load. Returns (total, electricity)."""
    kwh = np.asarray(watts, dtype=float) * hours_per_day * load_share * 365 * years / 1000
    power = kwh * usd_per_kwh
    return np.asarray(price, dtype=float) + power, power


def frontier(df: pd.DataFrame) -> np.ndarray:
    """True where no other model of the same kind has a higher score for no more watts."""
    out = np.zeros(len(df), dtype=bool)
    for _kind, part in df.groupby("kind"):
        s, w = part["score"].to_numpy(), part["watts"].to_numpy()
        for i, idx in enumerate(part.index):
            beaten = (s >= s[i]) & (w <= w[i]) & ((s > s[i]) | (w < w[i]))
            out[df.index.get_loc(idx)] = not beaten.any()
    return out


def build(bench: pd.DataFrame, value: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ranked, not_ranked). Ranked rows have a Blender score and a rated power figure."""
    cols = ["kind", "model_key", "median_score", "p25_score", "p75_score", "n_runs", "device_type"]
    if bench is None or bench.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["kind", "model_key", "name", "reason"])
    b = bench[cols].rename(columns={"median_score": "score", "device_type": "backend"})
    df = b.merge(rated_power(), on=["kind", "model_key"], how="left")

    missing = df[df["watts"].isna()].copy()
    missing["name"] = missing["model_key"].map(display_name)
    missing["reason"] = [
        NOT_ENTERED.get(k, "No rated power figure entered yet") for k in missing["model_key"]
    ]
    missing = missing[["kind", "model_key", "name", "reason"]].reset_index(drop=True)

    df = df[df["watts"].notna() & (df["score"] > 0)].reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df["name"] = df["model_key"].map(display_name)
    df["vendor"] = df["model_key"].str.split().str[0]
    df["power_kind"] = "rated"
    df["score_per_watt"] = df["score"] / df["watts"]
    df["seconds_per_job"] = seconds_per_job(df["score"])
    df["energy_kj_per_job"] = energy_per_job(df["score"], df["watts"]) / 1000
    df["edp"] = edp(df["score"], df["watts"], 1.0)
    df["on_frontier"] = frontier(df)
    df["pct_of_best"] = (
        100 * df["score_per_watt"] / df.groupby("kind")["score_per_watt"].transform("max")
    )
    df["rank"] = (
        df.groupby("kind")["score_per_watt"].rank(ascending=False, method="min").astype(int)
    )

    if value is not None and len(value) and "price_usd" in value.columns:
        price = (
            value[["kind", "model_key", "price_usd"]]
            .dropna()
            .drop_duplicates(["kind", "model_key"])
        )
        df = df.merge(price, on=["kind", "model_key"], how="left")
    else:
        df["price_usd"] = np.nan
    order = [
        "kind", "rank", "model_key", "name", "vendor", "year", "score", "p25_score", "p75_score",
        "n_runs", "backend", "watts", "power_label", "power_kind", "score_per_watt", "pct_of_best",
        "seconds_per_job", "energy_kj_per_job", "edp", "on_frontier", "price_usd",
    ]  # fmt: skip
    return df[order].sort_values(["kind", "rank"]).reset_index(drop=True), missing


def year_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Median and best perf per watt by launch year, per kind. Observed values, no model."""
    if df.empty:
        return pd.DataFrame(columns=["kind", "year", "n", "median", "best", "median_watts"])
    g = df.groupby(["kind", "year"])
    out = pd.DataFrame(
        {
            "n": g.size(),
            "median": g["score_per_watt"].median(),
            "best": g["score_per_watt"].max(),
            "median_watts": g["watts"].median(),
        }
    ).reset_index()
    return out


def summary(df: pd.DataFrame, rate: dict | None) -> dict:
    """Headline figures per kind, and the electricity rate the calculator starts from."""
    out: dict = {"kinds": {}}
    for kind, part in df.groupby("kind") if len(df) else []:
        best = part.sort_values("score_per_watt", ascending=False).iloc[0]
        latest = int(part["year"].max())
        recent = part[part["year"] > latest - 3]
        before = part[(part["year"] <= latest - 3) & (part["year"] > latest - 6)]
        change = None
        if len(recent) >= 3 and len(before) >= 3:
            change = 100 * (
                recent["score_per_watt"].median() / before["score_per_watt"].median() - 1
            )
        out["kinds"][kind] = {
            "n": int(len(part)),
            "n_priced": int(part["price_usd"].notna().sum()),
            "best_key": best["model_key"],
            "best_name": best["name"],
            "best_score_per_watt": float(best["score_per_watt"]),
            "best_watts": float(best["watts"]),
            "recent_years": [latest - 2, latest],
            "before_years": [latest - 5, latest - 3],
            "recent_median": float(recent["score_per_watt"].median()) if len(recent) else None,
            "before_median": float(before["score_per_watt"].median()) if len(before) else None,
            "change_pct": None if change is None else float(change),
        }
    out["electricity"] = rate or {
        "usd_per_kwh": FALLBACK_USD_PER_KWH,
        "period": None,
        "retrieved_at": None,
        "fallback": True,
    }
    out["tco_defaults"] = TCO_DEFAULTS
    out["job_samples"] = JOB_SAMPLES
    return out
