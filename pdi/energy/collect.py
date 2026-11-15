"""Collect monthly electricity demand from Ember's monthly electricity data file.

The file has one row per area, month and "electricity source"; demand is one of those rows
and its value sits in the generation column, in terawatt-hours. Only the rows for the seven
economies are kept. The raw file is saved untouched first, as for every source.
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pdi.common.fetch import CollectError, fetch, read_meta, utc_now
from pdi.energy import config

COLUMNS = ["ISO 3 code", "Date", "Electricity source", "Generation (TWh)"]
TIDY = ["country", "period", "demand_twh"]


def parse_ember(text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (demand, check): tidy demand, and generation plus net imports beside it."""
    raw = pd.read_csv(io.StringIO(text), usecols=lambda c: c in COLUMNS, dtype=str)
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        raise CollectError(f"Ember file has no column {missing}; the layout changed")
    ours = {iso: code for code, iso in config.ISO3.items()}
    raw = raw[raw["ISO 3 code"].isin(ours)]
    wanted = (config.DEMAND_ROW, config.GENERATION_ROW, config.NET_IMPORTS_ROW)
    raw = raw[raw["Electricity source"].isin(wanted)]
    df = pd.DataFrame(
        {
            "country": raw["ISO 3 code"].map(ours),
            "period": pd.to_datetime(raw["Date"], format="%Y-%m-%d"),
            "row": raw["Electricity source"],
            "value": pd.to_numeric(raw["Generation (TWh)"], errors="coerce"),
        }
    ).dropna(subset=["value"])
    if df.duplicated(["country", "period", "row"]).any():
        raise CollectError("Ember file has two values for one country, month and row")
    if (df["period"].dt.day != 1).any():
        raise CollectError("Ember dates are not month starts; the file may no longer be monthly")
    wide = df.pivot(index=["country", "period"], columns="row", values="value").reset_index()
    for col in wanted:
        if col not in wide.columns:
            wide[col] = np.nan
    demand = (
        wide.dropna(subset=[config.DEMAND_ROW])
        .rename(columns={config.DEMAND_ROW: "demand_twh"})[TIDY]
        .sort_values(["country", "period"], ignore_index=True)
    )
    check = wide.dropna(subset=[config.DEMAND_ROW, config.GENERATION_ROW]).copy()
    return demand, check


def validate(demand: pd.DataFrame, check: pd.DataFrame) -> pd.DataFrame:
    have = set(demand["country"])
    lost = [c for c in config.REQUIRED if c not in have]
    if lost:
        raise CollectError(f"Ember file has no monthly demand for {lost}")
    bad = demand[~demand["demand_twh"].between(config.MIN_TWH, config.MAX_TWH)]
    if not bad.empty:
        r = bad.iloc[0]
        raise CollectError(
            f"{r['country']} demand of {r['demand_twh']} TWh in {r['period']:%Y-%m} is outside "
            f"{config.MIN_TWH} to {config.MAX_TWH}; probably a unit change at the source"
        )
    for country, g in demand.groupby("country"):
        span = g["period"].max().to_period("M") - g["period"].min().to_period("M")
        if span.n + 1 != len(g):
            # A gap is allowed (it stays a gap downstream) but is worth a line in the log.
            print(f"note: {country} has months missing inside its history")
    # Demand = generation + net imports, where the file gives both.
    if not check.empty:
        supply = check[config.GENERATION_ROW] + check[config.NET_IMPORTS_ROW].fillna(0.0)
        off = (check[config.DEMAND_ROW] - supply).abs() / check[config.DEMAND_ROW]
        share = float((off > config.IDENTITY_TOLERANCE).mean())
        if share > config.IDENTITY_MAX_BAD_SHARE:
            raise CollectError(
                f"demand differs from generation plus net imports by more than "
                f"{config.IDENTITY_TOLERANCE:.0%} in {share:.0%} of months; the meaning of the "
                "Ember demand row has probably changed"
            )
    return demand


def append_vintages(demand: pd.DataFrame, seen_at: str, path: Path | None = None) -> int:
    path = path or config.VINTAGES
    new = demand[TIDY].assign(first_seen=seen_at)
    if path.exists():
        old = pd.read_parquet(path)
        last = old.sort_values("first_seen").groupby(["country", "period"]).tail(1)
        m = new.merge(last, on=["country", "period"], how="left", suffixes=("", "_last"))
        changed = m["demand_twh_last"].isna() | ~np.isclose(
            m["demand_twh"], m["demand_twh_last"], rtol=1e-9, atol=0
        )
        new = m.loc[changed, [*TIDY, "first_seen"]]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return len(new)


def is_fresh(now: datetime | None = None) -> bool:
    """True when the stored file was fetched recently enough to leave the source alone."""
    if not (config.RETRIEVED.exists() and config.DEMAND.exists()):
        return False
    when = json.loads(config.RETRIEVED.read_text()).get(config.SOURCE)
    if not when:
        return False
    age = (now or utc_now()) - datetime.fromisoformat(when)
    return age < timedelta(days=config.REFETCH_AFTER_DAYS)


def collect(today: date | None = None, raw_dir: Path | None = None) -> pd.DataFrame:
    today = today or date.today()
    path = fetch(config.EMBER_URL, (raw_dir or config.RAW_DIR) / f"{today}_ember_monthly.csv",
                 timeout=300)  # fmt: skip
    demand, check = parse_ember(path.read_text(encoding="utf-8-sig"))
    demand = validate(demand, check)
    config.STORE_DIR.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(config.DEMAND, index=False)
    retrieved = read_meta(path)["retrieved_at"]
    added = append_vintages(demand, retrieved)
    config.RETRIEVED.write_text(json.dumps({config.SOURCE: retrieved}, indent=1) + "\n")
    latest = demand.groupby("country")["period"].max()
    print(
        f"stored {len(demand):,} months of demand for {sorted(latest.index)}; {added:,} new or "
        f"revised values kept as vintages; latest month ranges {latest.min():%Y-%m} to "
        f"{latest.max():%Y-%m}"
    )
    return demand


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="fetch even if fetched recently")
    args = parser.parse_args(argv)
    if not args.force and is_fresh():
        print(f"Ember file fetched less than {config.REFETCH_AFTER_DAYS} days ago; not refetched")
        return
    collect()


if __name__ == "__main__":
    main()
