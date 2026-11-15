"""Validate and store: the collected series become one tidy table, and every value is also
kept as it was first seen, so later revisions never overwrite what was known at the time."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandera.pandas as pa

from pdi.common.fetch import CollectError, utc_now
from pdi.macro import config

RETRIEVED = config.STORE_DIR / "retrieved.json"

SCHEMA = pa.DataFrameSchema(
    {
        "series_id": pa.Column(str, pa.Check.isin(list(config.catalogue()))),
        "period": pa.Column(
            "datetime64[ns]",
            [
                pa.Check.ge(pd.Timestamp("1940-01-01")),
                # A date in the future means a calendar error (Thailand's Buddhist calendar
                # runs 543 years ahead) or a column read in the wrong order.
                pa.Check.le(pd.Timestamp.today() + pd.DateOffset(months=1)),
            ],
        ),
        "value": pa.Column(float, pa.Check(np.isfinite)),
    },
    unique=["series_id", "period"],
    strict=True,
    coerce=True,
)

# A single month moving a level series by more than this (100 x log change, so about a factor
# of 2.7 either way) is a unit error: thousands read as units, or a currency redenominated.
MAX_LOG_MOVE = 100.0


def check_units(df: pd.DataFrame) -> None:
    cat = config.catalogue()
    for sid, g in df.sort_values("period").groupby("series_id"):
        s = cat[sid]
        if s.transform != "log":
            continue
        if (g["value"] <= 0).any():
            raise CollectError(f"{sid} has zero or negative values but is measured in logs")
        jump = (100 * np.log(g["value"]).diff()).abs()
        if (jump > MAX_LOG_MOVE).any():
            when = g.loc[jump.idxmax(), "period"]
            raise CollectError(
                f"{sid} moves by a factor of {np.exp(jump.max() / 100):.1f} in {when:%Y-%m}; "
                "probably a unit or currency change at the source. Not stored."
            )


def validate(tidy: pd.DataFrame) -> pd.DataFrame:
    df = SCHEMA.validate(tidy)
    n = df["series_id"].nunique()
    if n < config.MIN_SERIES_COLLECTED:
        raise CollectError(
            f"only {n} series collected (expected at least {config.MIN_SERIES_COLLECTED}); "
            "a source has probably changed format"
        )
    check_units(df)
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    cat = config.catalogue()
    return df.assign(
        country=df["series_id"].map(lambda s: cat[s].country),
        indicator=df["series_id"].map(lambda s: cat[s].indicator),
        source=df["series_id"].map(lambda s: cat[s].source),
    ).sort_values(["series_id", "period"], ignore_index=True)


def append_vintages(df: pd.DataFrame, seen_at: str, path=config.VINTAGES) -> pd.DataFrame:
    """Add one row for every (series, month) whose value is new or differs from the last one
    seen. Existing rows are never changed."""
    new = df[["series_id", "period", "value"]].assign(first_seen=seen_at)
    if path.exists():
        old = pd.read_parquet(path)
        last = old.sort_values("first_seen").groupby(["series_id", "period"]).tail(1)
        merged = new.merge(last, on=["series_id", "period"], how="left", suffixes=("", "_last"))
        changed = merged["value_last"].isna() | ~np.isclose(
            merged["value"], merged["value_last"], rtol=1e-9, atol=0
        )
        new = merged.loc[changed, ["series_id", "period", "value", "first_seen"]]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return new


def update(tidy: pd.DataFrame, retrieved: dict[str, str | None]) -> pd.DataFrame:
    df = enrich(validate(tidy))
    config.STORE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.OBSERVATIONS, index=False)
    added = append_vintages(df, utc_now().isoformat())
    RETRIEVED.write_text(json.dumps(retrieved, indent=1) + "\n")
    latest = df.groupby("series_id")["period"].max()
    print(
        f"stored {len(df):,} observations for {df['series_id'].nunique()} series; "
        f"{len(added):,} new or revised values kept as vintages; "
        f"latest month ranges {latest.min():%Y-%m} to {latest.max():%Y-%m}"
    )
    return df


def load() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(config.OBSERVATIONS)
    retrieved = json.loads(RETRIEVED.read_text()) if RETRIEVED.exists() else {}
    return df, retrieved
