"""Collect US imports by product chapter from the Census international trade API.

Two questions are asked of the API for each partner: the value of every HS chapter in every
month, and the all-commodity total in every month. The second is what Census publishes as the
partner's total, and the first must add up to it; that check is the module's first exit
criterion. Raw answers are saved untouched (the key is never written to disk).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from pdi.common.fetch import CollectError, fetch, read_meta, utc_now
from pdi.trade import config


def parse(payload: list, value: str = config.VALUE) -> pd.DataFrame:
    """The API's answer (a header row, then rows of strings) as a table with real types."""
    if not isinstance(payload, list) or len(payload) < 2:
        raise CollectError("Census answer has no rows")
    df = pd.DataFrame(payload[1:], columns=payload[0])
    df = df.loc[:, ~df.columns.duplicated()]
    for col in (value, "time"):
        if col not in df.columns:
            raise CollectError(f"Census answer has no column {col}; the layout changed")
    df["period"] = pd.to_datetime(df["time"], format="%Y-%m")
    df["value"] = pd.to_numeric(df[value], errors="raise").astype(float)
    if (df["value"] < 0).any():
        raise CollectError("Census answer has a negative import value")
    return df


def parse_chapters(payload: list, partner: str) -> tuple[pd.DataFrame, dict[str, str]]:
    df = parse(payload)
    df = df[df["I_COMMODITY"].str.fullmatch(r"\d{2}")]
    if df.duplicated(["I_COMMODITY", "period"]).any():
        raise CollectError(f"{partner}: two values for one chapter and month")
    names = dict(zip(df["I_COMMODITY"], df.get("I_COMMODITY_SDESC", df["I_COMMODITY"]),
                     strict=False))  # fmt: skip
    out = df.rename(columns={"I_COMMODITY": "chapter"}).assign(partner=partner)
    return out[["partner", "chapter", "period", "value"]], names


def parse_totals(payload: list, partner: str) -> pd.DataFrame:
    df = parse(payload)
    if df.duplicated(["period"]).any():
        raise CollectError(f"{partner}: two all-commodity totals for one month")
    return df.assign(partner=partner)[["partner", "period", "value"]]


def check_totals(imports: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Sum of chapters against the published total, for every partner and month."""
    summed = imports.groupby(["partner", "period"], as_index=False)["value"].sum()
    both = summed.merge(totals, on=["partner", "period"], how="outer", suffixes=("_sum", "_total"))
    both["gap_share"] = (both["value_sum"] - both["value_total"]).abs() / both["value_total"]
    return both.sort_values(["partner", "period"], ignore_index=True)


def validate(imports: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    have = set(imports["partner"])
    lost = [p for p in config.PARTNERS if p not in have]
    if lost:
        raise CollectError(f"no chapter data for {lost}")
    check = check_totals(imports, totals)
    missing = check[check["value_sum"].isna() | check["value_total"].isna()]
    if not missing.empty:
        r = missing.iloc[0]
        raise CollectError(
            f"{r['partner']} {r['period']:%Y-%m}: chapters or the published total are missing"
        )
    bad = check[check["gap_share"] > config.TOTALS_TOLERANCE]
    if not bad.empty:
        r = bad.sort_values("gap_share").iloc[-1]
        raise CollectError(
            f"{len(bad)} partner-months where the chapters do not add up to the published "
            f"total within {config.TOTALS_TOLERANCE:.0%}; worst is {r['partner']} "
            f"{r['period']:%Y-%m} at {r['gap_share']:.1%}. Not stored."
        )
    latest = imports.groupby("partner")["period"].max()
    if latest.nunique() != 1:
        raise CollectError(f"partners end in different months: {latest.astype(str).to_dict()}")
    return check


def is_fresh(now: datetime | None = None) -> bool:
    if not (config.RETRIEVED.exists() and config.IMPORTS.exists()):
        return False
    when = json.loads(config.RETRIEVED.read_text()).get(config.SOURCE)
    if not when:
        return False
    return (now or utc_now()) - datetime.fromisoformat(when) < timedelta(
        days=config.REFETCH_AFTER_DAYS
    )


def collect(key: str, today: date | None = None, raw_dir: Path | None = None) -> pd.DataFrame:
    today = today or date.today()
    raw_dir = raw_dir or config.RAW_DIR
    frames, totals, names, retrieved = [], [], {}, None
    for partner, (code, _name) in config.PARTNERS.items():
        common = {"time": f"from {config.HISTORY_START}", "CTY_CODE": code, "key": key}
        path = fetch(
            config.API,
            raw_dir / f"{today}_census_{partner}_hs2.json",
            params={"get": f"I_COMMODITY,I_COMMODITY_SDESC,{config.VALUE}", "COMM_LVL": "HS2",
                    **common},
            timeout=300,
        )  # fmt: skip
        df, n = parse_chapters(json.loads(path.read_text()), partner)
        frames.append(df)
        names.update(n)
        retrieved = read_meta(path)["retrieved_at"]
        path = fetch(
            config.API,
            raw_dir / f"{today}_census_{partner}_total.json",
            params={"get": config.VALUE, **common},
            timeout=300,
        )
        totals.append(parse_totals(json.loads(path.read_text()), partner))
    imports = pd.concat(frames, ignore_index=True)
    total = pd.concat(totals, ignore_index=True)
    check = validate(imports, total)
    config.STORE_DIR.mkdir(parents=True, exist_ok=True)
    imports.to_parquet(config.IMPORTS, index=False)
    total.to_parquet(config.TOTALS, index=False)
    config.CHAPTERS.write_text(json.dumps(dict(sorted(names.items())), indent=1) + "\n")
    config.RETRIEVED.write_text(json.dumps({config.SOURCE: retrieved}, indent=1) + "\n")
    print(
        f"stored {len(imports):,} chapter-months for {imports['partner'].nunique()} partners, "
        f"{imports['period'].min():%Y-%m} to {imports['period'].max():%Y-%m}; chapters match "
        f"the published totals within {check['gap_share'].max():.4%} at worst"
    )
    return imports


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="fetch even if fetched recently")
    args = parser.parse_args(argv)
    key = os.environ.get(config.KEY_ENV, "").strip()
    if not key:
        # A fork or a first clone has no key. The pages then show the last stored data.
        print(f"{config.KEY_ENV} is not set; US imports not collected")
        return
    if not args.force and is_fresh():
        print(f"Census fetched less than {config.REFETCH_AFTER_DAYS} days ago; not refetched")
        return
    collect(key)


if __name__ == "__main__":
    main()
