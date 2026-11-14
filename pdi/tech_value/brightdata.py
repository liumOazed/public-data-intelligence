"""Amazon US listings through Bright Data's Web Scraper API: a second price source.

Bright Data is a paid data provider with a free monthly allowance. One keyword search is sent
per benchmarked model and a few results are kept for each, so a run stays well inside that
allowance; the hard caps below exist because every delivered record is billed. The API key
comes from the BRIGHTDATA_API_KEY environment variable (an Actions secret) and is only ever
sent in a request header.

Runs weekly, not daily: 80 models x 5 results is 400 records a run, about 1,700 a month.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date

import pandas as pd
import requests

from pdi.common.fetch import CollectError, utc_now
from pdi.tech_value import config, retail
from pdi.tech_value.names import model_key

SOURCE = "amazon_brightdata"
_OUT_OF_STOCK = ("unavailable", "out of stock")


class MissingKey(RuntimeError):
    pass


def _price(record: dict) -> float | None:
    for field in ("final_price", "price"):
        value = record.get(field)
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "").strip()
        try:
            if value is not None and float(value) > 0:
                return float(value)
        except ValueError:
            continue
    return None


def parse(records: list[dict], wanted: set[tuple[str, str]], day: str) -> list[dict]:
    """Keep new, in-stock, US-dollar listings recognised as exactly one wanted part."""
    rows, seen = [], set()
    for r in records:
        if not isinstance(r, dict) or r.get("error") or r.get("warning"):
            continue
        name, price = r.get("title") or r.get("name"), _price(r)
        if not name or price is None:
            continue
        if str(r.get("currency") or "USD").upper() not in ("USD", "$"):
            continue
        found = model_key(name)
        if found not in wanted:
            continue
        kind, key = found
        sku = str(r.get("asin") or r.get("url") or name)
        availability = str(r.get("availability") or "").lower()
        listing = retail.Listing(
            name=name,
            price=price,
            url=r.get("url"),
            sku=sku,
            in_stock=not any(word in availability for word in _OUT_OF_STOCK),
        )
        if sku in seen or not retail.keep(listing, kind, key):
            continue
        seen.add(sku)
        rows.append(
            {
                "date": day,
                "source": SOURCE,
                "sku": sku,
                "name": name,
                "category": kind,
                "sale_price": price,
                "regular_price": None,
                "price_update_date": None,
                "online_availability": listing.in_stock,
                "url": listing.url,
                "model_key": key,
                "seller": r.get("seller_name"),
                "retrieved_at": utc_now().isoformat(),
            }
        )
    return rows


def fetch_records(keywords: list[str], key: str) -> list[dict]:
    """Start one batch job, wait for it, return its records."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    per_input = config.BRIGHTDATA_RESULTS_PER_MODEL
    resp = requests.post(
        f"{config.BRIGHTDATA_API}/trigger",
        params={
            "dataset_id": config.BRIGHTDATA_AMAZON_DATASET,
            "type": "discover_new",
            "discover_by": "keyword",
            "include_errors": "true",
            # Both caps, in both places the API reads them: records are billed.
            "limit_per_input": per_input,
            "limit_multiple_results": per_input * len(keywords),
        },
        headers=headers,
        json={"input": [{"keyword": k} for k in keywords], "limit_per_input": per_input},
        timeout=60,
    )
    if resp.status_code != 200:
        raise CollectError(f"Bright Data trigger: HTTP {resp.status_code} {resp.text[:300]}")
    snapshot = resp.json()["snapshot_id"]
    print(f"Bright Data job {snapshot} started for {len(keywords)} keywords")

    deadline = time.monotonic() + config.BRIGHTDATA_WAIT_S
    while True:
        time.sleep(15)
        status = requests.get(
            f"{config.BRIGHTDATA_API}/progress/{snapshot}", headers=headers, timeout=60
        ).json()
        if status.get("status") == "ready":
            break
        if status.get("status") == "failed":
            raise CollectError(f"Bright Data job {snapshot} failed: {status}")
        if time.monotonic() > deadline:
            raise CollectError(f"Bright Data job {snapshot} not ready in time: {status}")
    data = requests.get(
        f"{config.BRIGHTDATA_API}/snapshot/{snapshot}",
        params={"format": "json"},
        headers=headers,
        timeout=300,
    )
    if data.status_code != 200:
        raise CollectError(f"Bright Data snapshot: HTTP {data.status_code} {data.text[:300]}")
    return data.json()


def collect(today: date | None = None, limit_models: int | None = None) -> pd.DataFrame:
    key = os.environ.get(config.BRIGHTDATA_KEY_ENV, "").strip()
    if not key:
        raise MissingKey(f"{config.BRIGHTDATA_KEY_ENV} is not set")
    day = (today or date.today()).isoformat()
    wanted = retail.targets()[: limit_models or None]
    records = fetch_records([retail.search_term(k) for _, k in wanted], key)

    raw_dir = config.RAW_DIR / "brightdata" / day
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Raw records kept untouched for this run's artifact; never committed or published.
    (raw_dir / "amazon.json").write_text(json.dumps(records))

    rows = parse(records, set(wanted), day)
    models = len({r["model_key"] for r in rows})
    print(
        f"{SOURCE}: {len(records)} records delivered, {len(rows)} listings kept, "
        f"{models} of {len(wanted)} models found"
    )
    if not rows:
        sample = json.dumps(records[:2])[:1500]
        raise CollectError(f"Bright Data returned no usable listings. First records: {sample}")
    df = retail.TIDY_SCHEMA.validate(pd.DataFrame(rows))
    config.PRICES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PRICES_DIR / f"{day}_{SOURCE}.parquet", index=False)
    return df


if __name__ == "__main__":
    try:
        table = collect(limit_models=int(os.environ.get("PDI_BRIGHTDATA_LIMIT_MODELS") or 0))
    except (MissingKey, retail.NoTargets) as exc:
        print(f"::warning title=Bright Data skipped::{exc}")
        raise SystemExit(0) from None
    except (CollectError, requests.RequestException) as exc:
        # A second source going missing must not stop the benchmarks and the first source
        # from being published. The warning shows on the run page.
        print(f"::warning title=Bright Data failed::{exc}")
        raise SystemExit(0) from None
    print(table.groupby("category")["model_key"].nunique())
    cols = ["model_key", "sale_price", "seller", "name"]
    print(table.sort_values(["model_key", "sale_price"])[cols].to_string(max_colwidth=70))
