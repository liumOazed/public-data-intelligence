"""SPECpower_ssj2008: measured server power at every load level, grouped by processor.

SPECpower runs a Java business workload at target loads from 100% down to 10% and then idle,
and reads the AC power of the whole server with a calibrated analyser at each step. That makes
it the one public source of *measured* processor-platform power with idle included. A server is
not a bare CPU: memory, disks, fans and power supplies are in the reading. So results are only
used when their disclosure names the processor, they are grouped by that processor, and they
are never ranked against the desktop chips' rated power.

    python -m pdi.tech_value.specpower
"""

from __future__ import annotations

import io
import re
from datetime import date

import numpy as np
import pandas as pd

from pdi.common.fetch import fetch, latest, read_meta
from pdi.tech_value import config

CONF = "power_ssj2008"
LOADS = (100, 90, 80, 70, 60, 50, 40, 30, 20, 10)
COLUMNS = [
    "system", "hardware_vendor", "processor", "processor_key", "cpu_vendor", "nodes", "chips",
    "cores", "published", "hw_avail", "ops_per_watt", "watts_100", "watts_idle", "idle_share",
    *[f"ppw_{p}" for p in LOADS], "disclosure_url", "retrieved_at",
]  # fmt: skip

_TRADEMARKS = re.compile(r"\(r\)|\(tm\)|®|™", re.I)
_NOISE = re.compile(
    r"\b(processor|cpu|64-bit|multi-core|\d+-core)\b|@?\s*\d+(\.\d+)?\s*ghz\b|,.*$", re.I
)


def processor_name(raw: str | None) -> str | None:
    """'AMD EPYC 7742 2.25Ghz' -> 'AMD EPYC 7742'; trademarks, clocks and 'Processor' dropped."""
    if raw is None or not str(raw).strip() or str(raw).strip() in ("--", "None"):
        return None
    text = _TRADEMARKS.sub("", str(raw))
    text = _NOISE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text or None


def cpu_vendor(name: str) -> str:
    n = name.lower()
    for key, label in (("intel", "Intel"), ("xeon", "Intel"), ("amd", "AMD"), ("epyc", "AMD"),
                       ("ampere", "Ampere")):  # fmt: skip
        if key in n:
            return label
    return "Other"


def _month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.strip(), format="%b-%Y", errors="coerce")


def parse(raw_csv: bytes | str, retrieved_at: str | None = None) -> pd.DataFrame:
    text = raw_csv.decode("utf-8", errors="replace") if isinstance(raw_csv, bytes) else raw_csv
    raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    # SPEC pads many header names with tabs and spaces, and spells one "watts@10%".
    raw.columns = [
        re.sub(r"\s+", " ", c.replace("watts@", "watts @ ")).strip() for c in raw.columns
    ]

    def num(col: str) -> pd.Series:
        return pd.to_numeric(raw[col].str.replace(",", "").str.strip(), errors="coerce")

    df = pd.DataFrame(
        {
            "system": raw["System"].str.strip(),
            "hardware_vendor": raw["Hardware Vendor"].str.strip(),
            "processor": raw["Processor"].map(processor_name),
            "nodes": num("Nodes"),
            "chips": num("# Chips"),
            "cores": num("# Cores"),
            "published": _month(raw["Published"]),
            "hw_avail": _month(raw["HW Avail"]),
            "ops_per_watt": num("Result"),
            "watts_100": num("Average watts @ 100% of target load"),
            "watts_idle": num("Average watts @ active idle"),
            "disclosure_url": raw["Disclosure"].str.strip().str.replace("http://", "https://"),
        }
    )
    for p in LOADS:
        df[f"ppw_{p}"] = num(f"Performance/power @ {p}% of target load")
    keep = (
        (raw["Benchmark"].str.strip() == CONF)
        & df["processor"].notna()
        & (df["ops_per_watt"] > 0)
        & (df["watts_100"] > 0)
        & df["published"].notna()
    )
    df = df[keep].copy()
    df["processor_key"] = df["processor"].str.lower()
    df["cpu_vendor"] = df["processor"].map(cpu_vendor)
    df["idle_share"] = df["watts_idle"] / df["watts_100"]
    df["retrieved_at"] = retrieved_at
    for col in ("nodes", "chips", "cores"):
        df[col] = df[col].fillna(0).astype(int)
    return df[COLUMNS].sort_values(["published", "processor"]).reset_index(drop=True)


def by_processor(df: pd.DataFrame) -> pd.DataFrame:
    """One row per processor: medians of its measured results, newest publication first."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(["processor_key"])
    out = pd.DataFrame(
        {
            "processor": g["processor"].first(),
            "cpu_vendor": g["cpu_vendor"].first(),
            "n_results": g.size(),
            "n_vendors": g["hardware_vendor"].nunique(),
            "ops_per_watt": g["ops_per_watt"].median(),
            "best_ops_per_watt": g["ops_per_watt"].max(),
            "idle_share": g["idle_share"].median(),
            "first_published": g["published"].min(),
            "year": g["published"].min().dt.year,
            **{f"ppw_{p}": g[f"ppw_{p}"].median() for p in LOADS},
        }
    ).reset_index()
    return out.sort_values("ops_per_watt", ascending=False).reset_index(drop=True)


def year_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Best and median measured ssj_ops per watt, and median idle share, by publication year."""
    if df.empty:
        return pd.DataFrame(columns=["year", "n", "best", "median", "idle_share"])
    d = df.assign(year=df["published"].dt.year)
    g = d.groupby("year")
    return pd.DataFrame(
        {
            "n": g.size(),
            "best": g["ops_per_watt"].max(),
            "median": g["ops_per_watt"].median(),
            "idle_share": g["idle_share"].median(),
        }
    ).reset_index()


def doubling_years(trend: pd.DataFrame, last_n: int = 6) -> float | None:
    """Years for the yearly best result to double, from a log-linear fit over the last n years."""
    t = trend.dropna(subset=["best"]).tail(last_n)
    if len(t) < 4:
        return None
    slope = np.polyfit(t["year"], np.log2(t["best"]), 1)[0]
    return float(1 / slope) if slope > 0 else None


def load() -> pd.DataFrame:
    path = config.SPECPOWER_TABLE
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)


def collect(today: date | None = None):
    today = today or date.today()
    dest = config.RAW_DIR / "specpower" / f"{today.isoformat()}_{CONF}.csv"
    return fetch(config.SPEC_ENDPOINT, dest, params={"conf": CONF, "op": "dump",
                                                    "format": "csvdump"}, timeout=300)  # fmt: skip


def transform() -> pd.DataFrame:
    path = latest(config.RAW_DIR / "specpower", f"*_{CONF}.csv")
    df = parse(path.read_bytes(), read_meta(path)["retrieved_at"])
    config.SPECPOWER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.SPECPOWER_TABLE, index=False)
    return df


def main() -> None:
    collect()
    df = transform()
    print(f"SPECpower: {len(df)} results naming {df['processor_key'].nunique()} processors")


if __name__ == "__main__":
    main()
