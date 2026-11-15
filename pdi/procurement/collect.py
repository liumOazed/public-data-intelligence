"""Collect federal contract transactions for computer hardware from USAspending.

USAspending builds a file of transactions on request. The collector asks for one file per
fiscal year, keeps only the columns named in the config (nothing about a person or an address),
and stores sums: by month, agency and product code; by supplier and fiscal year; by how the
award was competed; and by the year an award began. The transactions themselves are not
committed. The monthly sums are then compared with the figure USAspending's own search gives
for the same filters, which is the module's exit criterion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from pdi.common.fetch import RETRYABLE, USER_AGENT, CollectError, fetch, utc_now
from pdi.procurement import config, tables

MONEY = ("federal_action_obligation", "total_dollars_obligated",
         "potential_total_value_of_award", "total_outlayed_amount_for_overall_award")  # fmt: skip


def call(method: str, path: str, body: dict | None = None, retries: int = 4) -> dict:
    last = "no attempt made"
    for attempt in range(retries):
        if attempt:
            time.sleep(2**attempt)
        try:
            resp = requests.request(method, config.API + path, json=body, timeout=180,
                                    headers={"User-Agent": USER_AGENT})  # fmt: skip
        except requests.RequestException as exc:
            last = type(exc).__name__
            continue
        if resp.status_code == 200:
            return resp.json()
        last = f"HTTP {resp.status_code}: {resp.text[:200]!r}"
        if resp.status_code not in RETRYABLE:
            break
    raise CollectError(f"USAspending {path} failed: {last}")


def filters(start: date, end: date) -> dict:
    return {
        "award_type_codes": config.AWARD_TYPES,
        "psc_codes": sorted(config.PSC),
        "time_period": [{"start_date": start.isoformat(), "end_date": end.isoformat()}],
    }


def fiscal_years(today: date) -> dict[int, tuple[date, date]]:
    """Each fiscal year from the first to the current one, with its first and last day so far."""
    current = today.year + (today.month >= 10)
    return {
        fy: (date(fy - 1, 10, 1), min(date(fy, 9, 30), today))
        for fy in range(config.FIRST_FISCAL_YEAR, current + 1)
    }


def read_zip(path: Path) -> pd.DataFrame:
    """Every prime-transaction file in a USAspending download, as text columns."""
    frames = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if "PrimeTransactions" not in name:
                continue
            with z.open(name) as f:
                rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
            frames.append(pd.DataFrame(rows, dtype=str))
    if not frames:
        raise CollectError(f"{path.name} holds no prime-transaction file; the layout changed")
    return pd.concat(frames, ignore_index=True)


def parse(raw: pd.DataFrame) -> pd.DataFrame:
    """Typed transactions, only the configured columns. Stops on anything outside the slice."""
    if raw.empty:
        return pd.DataFrame(columns=config.COLUMNS)
    lost = [c for c in config.COLUMNS if c not in raw.columns]
    if lost:
        raise CollectError(f"USAspending download has no column {lost}; the layout changed")
    df = raw[config.COLUMNS].copy()
    df["action_date"] = pd.to_datetime(df["action_date"], format="%Y-%m-%d")
    for col in MONEY:
        df[col] = pd.to_numeric(df[col].replace("", None), errors="raise").astype(float)
    if df["federal_action_obligation"].isna().any():
        raise CollectError("a transaction has no obligation amount")
    stray = sorted(set(df["product_or_service_code"]) - set(config.PSC))
    if stray:
        raise CollectError(f"download holds product codes outside the slice: {stray[:5]}")
    return df.drop_duplicates("contract_transaction_unique_key", ignore_index=True)


def download_all(today: date, raw_dir: Path) -> tuple[pd.DataFrame, str]:
    years = fiscal_years(today)
    asked = {}
    for fy, (start, end) in years.items():
        body = {"filters": filters(start, end), "columns": config.COLUMNS, "file_format": "csv"}
        job = call("POST", config.DOWNLOAD, body)
        asked[fy] = job["file_name"]
        time.sleep(1)
    frames, retrieved = [], utc_now().isoformat()
    deadline = time.monotonic() + config.DOWNLOAD_WAIT_S
    for fy, name in asked.items():
        while True:
            status = call("GET", f"{config.DOWNLOAD_STATUS}?file_name={name}")
            if status.get("status") == "finished":
                break
            if status.get("status") == "failed" or time.monotonic() > deadline:
                raise CollectError(
                    f"USAspending did not build the fiscal {fy} file: "
                    f"{status.get('status')} {status.get('message')}"
                )
            time.sleep(config.DOWNLOAD_POLL_S)
        path = fetch(status["file_url"], raw_dir / f"{today}_usaspending_fy{fy}.zip", timeout=600)
        df = parse(read_zip(path))
        start, end = years[fy]
        if not df.empty and not df["action_date"].between(str(start), str(end)).all():
            raise CollectError(f"fiscal {fy} file holds actions outside {start} to {end}")
        print(f"fiscal {fy}: {len(df):,} transactions, "
              f"${df['federal_action_obligation'].sum() / 1e9:.2f}bn obligated")  # fmt: skip
        frames.append(df)
    tx = pd.concat(frames, ignore_index=True)
    if tx["contract_transaction_unique_key"].duplicated().any():
        raise CollectError("one transaction appears in two fiscal-year files")
    return tx, retrieved


def parse_published(payload: dict) -> pd.DataFrame:
    """USAspending's own monthly figure. Its months are fiscal: month 1 is October."""
    rows = []
    for r in payload.get("results", []):
        fy, m = int(r["time_period"]["fiscal_year"]), int(r["time_period"]["month"])
        year, month = (fy - 1, m + 9) if m <= 3 else (fy, m - 3)
        rows.append({"period": pd.Timestamp(year, month, 1),
                     "published": float(r["aggregated_amount"])})  # fmt: skip
    if not rows:
        raise CollectError("USAspending's spending-over-time answer has no months")
    return pd.DataFrame(rows).sort_values("period", ignore_index=True)


def check_totals(monthly: pd.DataFrame, published: pd.DataFrame, today: date) -> pd.DataFrame:
    """Our monthly sum against the published one, for every full month before retrieval."""
    ours = monthly.groupby("period", as_index=False)["obligations"].sum()
    both = ours.merge(published, on="period", how="outer").sort_values("period")
    both = both[both["period"] < pd.Timestamp(today.replace(day=1))]
    both["obligations"] = both["obligations"].fillna(0.0)
    both["gap_share"] = (both["obligations"] - both["published"]).abs() / both["published"].abs()
    both["within_limit"] = both["gap_share"] <= config.TOTALS_TOLERANCE
    return both.reset_index(drop=True)


SEARCH_FIELDS = ["Award ID", "Mod", "Transaction Amount", "Action Date"]


def search_rows(period: pd.Timestamp) -> pd.DataFrame:
    """Every transaction USAspending's search counts in one month, page by page."""
    end = (period + pd.offsets.MonthEnd(0)).date()
    body = {"filters": filters(period.date(), end), "limit": 100, "fields": SEARCH_FIELDS,
            "sort": "Transaction Amount", "order": "desc"}  # fmt: skip
    rows, page = [], 1
    while True:
        answer = call("POST", config.BY_TRANSACTION, {**body, "page": page})
        rows += answer.get("results", [])
        if not answer.get("page_metadata", {}).get("hasNext"):
            break
        page += 1
        if page > 2000:
            raise CollectError(f"{period:%Y-%m}: search listing did not end after 2,000 pages")
    return pd.DataFrame(rows, columns=SEARCH_FIELDS)


def _keys(award: pd.Series, mod: pd.Series, date: pd.Series, amount: pd.Series) -> Counter:
    return Counter(zip(award.astype(str), mod.astype(str), date.astype(str),
                       amount.astype(float).round(2), strict=True))  # fmt: skip


def reconcile(period: pd.Timestamp, month_tx: pd.DataFrame, listed: pd.DataFrame) -> dict:
    """Match the search's transactions for one month with the download's, row by row."""
    piid = month_tx["contract_award_unique_key"].str.split("_").str[2]
    ours = _keys(piid, month_tx["modification_number"],
                 month_tx["action_date"].dt.strftime("%Y-%m-%d"),
                 month_tx["federal_action_obligation"])  # fmt: skip
    theirs = _keys(listed["Award ID"], listed["Mod"], listed["Action Date"],
                   listed["Transaction Amount"])  # fmt: skip
    extra = theirs - ours
    copies = {k: v for k, v in extra.items() if k in ours}
    missing = {k: v for k, v in extra.items() if k not in ours}
    download_only = ours - theirs
    ours_sum = float(month_tx["federal_action_obligation"].sum())
    listed_sum = float(listed["Transaction Amount"].astype(float).sum())
    copies_sum = float(sum(k[3] * v for k, v in copies.items()))
    gap = abs(listed_sum - copies_sum - ours_sum) / max(abs(ours_sum), 1.0)
    return {
        "period": period.strftime("%Y-%m"),
        "download": ours_sum,
        "download_rows": int(len(month_tx)),
        "search": listed_sum,
        "search_rows": int(len(listed)),
        "extra_copies": int(sum(copies.values())),
        "extra_copies_sum": copies_sum,
        "search_rows_not_in_download": int(sum(missing.values())),
        "download_rows_not_in_search": int(sum(download_only.values())),
        "download_rows_not_in_search_sum": float(sum(k[3] * v for k, v in download_only.items())),
        "gap_after_copies_share": float(gap),
        "reconciled": bool(not missing and gap <= config.RECONCILE_TOLERANCE),
    }


def named_suppliers(sy: pd.DataFrame, names: dict[str, str]) -> dict[str, str]:
    top = (
        sy.sort_values("obligations", ascending=False)
        .groupby("fiscal_year")
        .head(config.NAMED_SUPPLIERS_PER_YEAR)
    )
    return {s: names.get(s, "") for s in sorted(set(top["supplier"]))}


def build(tx: pd.DataFrame) -> dict:
    """Every stored table from the transactions."""
    tx, names, stats = tables.with_suppliers(tx)
    sy = tables.supplier_year(tx)
    return {
        "monthly": tables.monthly(tx),
        "supplier_year": sy,
        "competition": tables.competition(tx),
        "cohorts": tables.cohorts(tx),
        "suppliers": named_suppliers(sy, names),
        "cleaning": stats,
    }


def is_fresh(now: datetime | None = None) -> bool:
    if not (config.RETRIEVED.exists() and config.MONTHLY.exists()):
        return False
    when = json.loads(config.RETRIEVED.read_text()).get(config.SOURCE)
    if not when:
        return False
    return (now or utc_now()) - datetime.fromisoformat(when) < timedelta(
        days=config.REFETCH_AFTER_DAYS
    )


def collect(today: date | None = None, raw_dir: Path | None = None) -> pd.DataFrame:
    today = today or utc_now().date()
    raw_dir = raw_dir or config.RAW_DIR
    tx, retrieved = download_all(today, raw_dir)
    built = build(tx)

    start = date(config.FIRST_FISCAL_YEAR - 1, 10, 1)
    answer = call("POST", config.OVER_TIME, {"group": "month", "filters": filters(start, today)})
    raw_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(answer).encode()
    (raw_dir / f"{today}_usaspending_over_time.json").write_bytes(body)
    (raw_dir / f"{today}_usaspending_over_time.json.meta.json").write_text(
        json.dumps({"url": config.API + config.OVER_TIME, "retrieved_at": utc_now().isoformat(),
                    "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}, indent=2)
    )  # fmt: skip
    check = check_totals(built["monthly"], parse_published(answer), today)
    month = tx["action_date"].dt.to_period("M").dt.to_timestamp()
    diagnosis = []
    for period in check.loc[~check["within_limit"], "period"]:
        row = reconcile(period, tx[month == period], search_rows(period))
        print(f"{row['period']}: {row['extra_copies']} extra copies in the search, "
              f"{row['search_rows_not_in_download']} search rows not in the download, "
              f"gap after copies {row['gap_after_copies_share']:.4%}")  # fmt: skip
        diagnosis.append(row)

    config.STORE_DIR.mkdir(parents=True, exist_ok=True)
    built["monthly"].to_parquet(config.MONTHLY, index=False)
    built["supplier_year"].to_parquet(config.SUPPLIER_YEAR, index=False)
    built["competition"].to_parquet(config.COMPETITION, index=False)
    built["cohorts"].to_parquet(config.COHORTS, index=False)
    check.to_parquet(config.CHECK, index=False)
    config.SUPPLIERS.write_text(json.dumps(built["suppliers"], indent=1) + "\n")
    config.CLEANING.write_text(json.dumps(built["cleaning"], indent=1) + "\n")
    config.DIAGNOSIS.write_text(json.dumps(diagnosis, indent=1) + "\n")
    config.RETRIEVED.write_text(json.dumps({config.SOURCE: retrieved}, indent=1) + "\n")
    print(
        f"stored {len(tx):,} transactions as sums, {check['period'].min():%Y-%m} to "
        f"{check['period'].max():%Y-%m}; {int(check['within_limit'].sum())} of {len(check)} "
        f"months within {config.TOTALS_TOLERANCE:.0%} of the published figure, worst "
        f"{check['gap_share'].max():.1%}"
    )
    return check


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="fetch even if fetched recently")
    args = parser.parse_args(argv)
    if not args.force and is_fresh():
        print(f"USAspending fetched less than {config.REFETCH_AFTER_DAYS} days ago; not refetched")
        return
    collect()


if __name__ == "__main__":
    main()
