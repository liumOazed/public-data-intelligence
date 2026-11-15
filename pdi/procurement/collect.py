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


def diagnose(period: pd.Timestamp, ours: float) -> dict:
    """List the transactions behind the published figure for one month and count repeats."""
    end = (period + pd.offsets.MonthEnd(0)).date()
    body = {"filters": filters(period.date(), end), "limit": 100, "sort": "Transaction Amount",
            "order": "desc", "fields": ["Award ID", "Mod", "Transaction Amount", "Action Date",
                                        "internal_id"]}  # fmt: skip
    rows, page = [], 1
    while True:
        answer = call("POST", config.BY_TRANSACTION, {**body, "page": page})
        rows += answer.get("results", [])
        if not answer.get("page_metadata", {}).get("hasNext") or page >= 400:
            break
        page += 1
    if not rows:
        return {"period": period.strftime("%Y-%m"), "ours": float(ours), "listed_rows": 0}
    df = pd.DataFrame(rows)
    repeat = df[df.duplicated("internal_id", keep="first")]
    once = df.drop_duplicates("internal_id")
    return {
        "period": period.strftime("%Y-%m"),
        "ours": float(ours),
        "listed_rows": int(len(df)),
        "listed_sum": float(df["Transaction Amount"].sum()),
        "repeated_rows": int(len(repeat)),
        "repeated_sum": float(repeat["Transaction Amount"].sum()),
        "listed_once_sum": float(once["Transaction Amount"].sum()),
        "gap_after_removing_repeats_share": float(
            abs(ours - once["Transaction Amount"].sum()) / abs(once["Transaction Amount"].sum())
        ),
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
    missed = check[~check["within_limit"]]
    worst = missed.assign(gap=(missed["obligations"] - missed["published"]).abs()).nlargest(
        config.DIAGNOSE_MONTHS, "gap"
    )
    diagnosis = [diagnose(r["period"], r["obligations"]) for _, r in worst.iterrows()]

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
