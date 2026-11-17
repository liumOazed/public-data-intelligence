"""Amazon US and Best Buy listings through Bright Data's Web Scraper API.

Bright Data is a paid data provider with a free monthly allowance. One keyword search is sent
per benchmarked model to each store and a few results are kept for each, so a run stays close
to that allowance; the hard caps below exist because every delivered record is billed. The
API key comes from the BRIGHTDATA_API_KEY environment variable (an Actions secret) and is
only ever sent in a request header.

Runs weekly, not daily: 80 models x 5 results is 400 records a run for each store, about
3,500 a month for both.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date

import pandas as pd
import requests

from pdi.common.fetch import CollectError, utc_now
from pdi.tech_value import config, retail
from pdi.tech_value.names import model_key

SOURCE = "amazon_brightdata"
BESTBUY_SOURCE = "bestbuy_brightdata"
# Best Buy files a bare part under one of these; whole PCs sit under "Gaming Desktops".
_BESTBUY_PART_CATEGORIES = ("gpus / video graphics cards", "cpus / processors")
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


# Amazon gives no condition field. These are the signals a listing is not new: it sits in the
# "Amazon Renewed" bestseller category, or it is sold by Amazon's used-goods store or by a
# seller that names itself a refurbisher.
_USED_SELLER = re.compile(
    r"renew|refurb|resale|warehouse|\bused\b|open[\s-]?box|pre[\s-]?owned", re.IGNORECASE
)


def looks_used(record: dict) -> bool:
    if "renewed" in str(record.get("root_bs_category") or "").lower():
        return True
    sellers = (record.get("seller_name"), record.get("buybox_seller"))
    return any(_USED_SELLER.search(str(s)) for s in sellers if s)


_TRUSTED_AMAZON_SELLERS = ("amazon.com", "amazon", "micro center")
_AIB_PARTNERS = frozenset(
    {
        "amd",
        "intel",
        "nvidia",
        "asus",
        "msi",
        "gigabyte",
        "pny",
        "xfx",
        "sapphire",
        "powercolor",
        "asrock",
        "zotac",
        "sparkle",
        "maxsun",
    }
)
_TRUSTED_BESTBUY_SELLERS = ("best buy", "newegg official")


def sold_new(record: dict) -> bool:
    """True only when the seller is Amazon, Micro Center, or the part's own brand/AIB partner.

    Marketplace resellers can offer a used part under a new-looking listing, and nothing in the
    data tells those apart, so they are all left out. That costs coverage and buys certainty.
    """
    seller = str(record.get("buybox_seller") or record.get("seller_name") or "").strip().lower()
    if not seller:
        return False
    if seller in _TRUSTED_AMAZON_SELLERS:
        return True
    brand = str(record.get("brand") or "").strip().lower()
    if not brand:
        return False
    s_first, b_first = seller.split()[0], brand.split()[0]
    if s_first == b_first and seller not in ("amazon germany", "amazon japan", "amazon uk"):
        return True
    return s_first in _AIB_PARTNERS and b_first in _AIB_PARTNERS


def parse(records: list[dict], wanted: set[tuple[str, str]], day: str) -> list[dict]:
    """Keep new, in-stock, US-dollar listings recognised as a wanted part."""
    rows, seen = [], set()
    for r in records:
        if not isinstance(r, dict) or r.get("error") or r.get("warning"):
            continue
        name, price = r.get("title") or r.get("name"), _price(r)
        if not name or price is None:
            continue
        if str(r.get("currency") or "USD").upper() not in ("USD", "$"):
            continue
        if looks_used(r) or not sold_new(r):
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
                "seller": r.get("seller_name") or r.get("buybox_seller"),
                "retrieved_at": utc_now().isoformat(),
            }
        )
    return rows


def _seller_name(record: dict) -> str:
    seller = record.get("seller")
    if isinstance(seller, str) and seller.strip().startswith("{"):
        try:
            seller = json.loads(seller)
        except ValueError:
            seller = None
    if isinstance(seller, dict):
        seller = seller.get("name")
    return str(seller or record.get("store_name") or "").strip()


def parse_bestbuy(records: list[dict], wanted: set[tuple[str, str]], day: str) -> list[dict]:
    """Keep new, in-stock parts sold by Best Buy (or Newegg Official) and recognised as wanted.

    Best Buy's search is loose: a search for one card returns other cards and whole PCs. A
    record counts only if Best Buy files it as a bare processor or graphics card, the seller is
    Best Buy itself or Newegg Official (not an unverified marketplace reseller), nothing in the
    title marks it refurbished, open-box or pre-owned, and its name parses to a wanted model.
    """
    rows, seen, dropped = [], set(), []
    for r in records:
        if not isinstance(r, dict) or r.get("error") or r.get("warning"):
            dropped.append(("collector error", str((r or {}).get("error"))[:60]))
            continue
        name, price = r.get("title"), _price(r)
        crumbs = r.get("breadcrumbs")
        if not r.get("product_category") and isinstance(crumbs, list):
            crumbs = ">".join(str(c.get("name")) for c in crumbs if isinstance(c, dict))
        category = str(r.get("product_category") or crumbs or "").lower()
        found = model_key(name) if name else None
        seller = _seller_name(r)
        if not name or price is None:
            reason = "no title or price"
        elif str(r.get("currency") or "USD").upper() not in ("USD", "$"):
            reason = "not US dollars"
        elif seller.lower() not in _TRUSTED_BESTBUY_SELLERS:
            reason = f"seller is {seller or 'unnamed'}"
        elif _USED_SELLER.search(name):
            reason = "not new"
        elif not category.endswith(_BESTBUY_PART_CATEGORIES):
            reason = f"category {category.rsplit('>', 1)[-1] or 'missing'}"
        elif found not in wanted:
            reason = f"not a model asked for ({found[1] if found else 'unrecognised'})"
        else:
            reason = None
        if reason:
            dropped.append((reason, str(name)[:60]))
            continue
        kind, key = found
        sku = str(r.get("sku") or r.get("product_id") or r.get("url") or name)
        listing = retail.Listing(
            name=name,
            price=price,
            url=r.get("url"),
            sku=sku,
            in_stock=str(r.get("availability_new") or "").lower() == "in_stock",
        )
        if sku in seen or not retail.keep(listing, kind, key):
            dropped.append(("duplicate, accessory or below the price floor", name[:60]))
            continue
        seen.add(sku)
        rows.append(
            {
                "date": day,
                "source": BESTBUY_SOURCE,
                "sku": sku,
                "name": name,
                "category": kind,
                "sale_price": price,
                "regular_price": None,
                "price_update_date": None,
                "online_availability": listing.in_stock,
                "url": listing.url,
                "model_key": key,
                "seller": seller or "Best Buy",
                "retrieved_at": utc_now().isoformat(),
            }
        )
    # Why each record was left out, so a thin week can be told from a broken filter.
    for reason, name in dropped:
        print(f"{BESTBUY_SOURCE} dropped: {reason} | {name}")
    return rows


base_query = retail.base_query


def search_queries(wanted: list[tuple[str, str]]) -> list[str]:
    """Deduplicated search terms for `wanted`, preserving priority order."""
    return list(dict.fromkeys(base_query(k) for _, k in wanted))


def start_job(keywords: list[str], key: str, dataset: str, by: str) -> str:
    """Start one batch job of keyword searches and return its id.

    `by` is both the discovery type and the input field name: "keyword" for Amazon,
    "keywords" for Best Buy.
    """
    per_input = config.BRIGHTDATA_RESULTS_PER_MODEL
    resp = requests.post(
        f"{config.BRIGHTDATA_API}/trigger",
        params={
            "dataset_id": dataset,
            "type": "discover_new",
            "discover_by": by,
            "include_errors": "true",
            # Both caps, in both places the API reads them: records are billed.
            "limit_per_input": per_input,
            "limit_multiple_results": per_input * len(keywords),
        },
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"input": [{by: k} for k in keywords], "limit_per_input": per_input},
        timeout=60,
    )
    if resp.status_code != 200:
        raise CollectError(f"Bright Data trigger: HTTP {resp.status_code} {resp.text[:300]}")
    snapshot = resp.json()["snapshot_id"]
    print(f"Bright Data job {snapshot} started for {len(keywords)} keywords")
    return snapshot


def wait_job(snapshot: str, key: str, wait_s: int) -> list[dict]:
    """Wait for a job to finish and return its records."""
    headers = {"Authorization": f"Bearer {key}"}
    deadline = time.monotonic() + wait_s
    while True:
        status = requests.get(
            f"{config.BRIGHTDATA_API}/progress/{snapshot}", headers=headers, timeout=60
        ).json()
        if status.get("status") == "ready":
            break
        if status.get("status") == "failed":
            raise CollectError(f"Bright Data job {snapshot} failed: {status}")
        if time.monotonic() > deadline:
            raise CollectError(f"Bright Data job {snapshot} not ready in time: {status}")
        time.sleep(15)
    data = requests.get(
        f"{config.BRIGHTDATA_API}/snapshot/{snapshot}",
        params={"format": "json"},
        headers=headers,
        timeout=300,
    )
    if data.status_code != 200:
        raise CollectError(f"Bright Data snapshot: HTTP {data.status_code} {data.text[:300]}")
    return data.json()


# name -> (source key, dataset, discovery type and input field, parser, longest wait)
STORES = {
    "amazon": (
        SOURCE,
        config.BRIGHTDATA_AMAZON_DATASET,
        "keyword",
        parse,
        config.BRIGHTDATA_WAIT_S,
    ),
    "bestbuy": (
        BESTBUY_SOURCE,
        config.BRIGHTDATA_BESTBUY_DATASET,
        "keywords",
        parse_bestbuy,
        config.BRIGHTDATA_BESTBUY_WAIT_S,
    ),
}


def _key() -> str:
    key = os.environ.get(config.BRIGHTDATA_KEY_ENV, "").strip()
    if not key:
        raise MissingKey(f"{config.BRIGHTDATA_KEY_ENV} is not set")
    return key


def start(store: str, wanted: list[tuple[str, str]]) -> str:
    _, dataset, by, _, _ = STORES[store]
    return start_job(search_queries(wanted), _key(), dataset, by)


def finish(store: str, snapshot: str, wanted: list[tuple[str, str]], day: str) -> pd.DataFrame:
    source, _, _, parser, wait_s = STORES[store]
    records = wait_job(snapshot, _key(), wait_s)

    raw_dir = config.RAW_DIR / "brightdata" / day
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Raw records kept untouched for this run's artifact; never committed or published.
    (raw_dir / f"{store}.json").write_text(json.dumps(records))

    try:
        acceptable = retail.all_targets() | set(wanted)
    except retail.NoTargets:
        acceptable = set(wanted)
    rows = parser(records, acceptable, day)
    errors = sum(1 for r in records if isinstance(r, dict) and r.get("error"))
    models = len({r["model_key"] for r in rows})
    print(
        f"{source}: {len(records)} records delivered ({errors} errors), {len(rows)} listings "
        f"kept, {models} models found ({len(wanted)} targeted)"
    )
    if not rows:
        sample = json.dumps(records[:2])[:1500]
        raise CollectError(f"{source} returned no usable listings. First records: {sample}")
    df = retail.TIDY_SCHEMA.validate(pd.DataFrame(rows))
    config.PRICES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PRICES_DIR / f"{day}_{source}.parquet", index=False)
    return df


def collect(
    today: date | None = None, limit_models: int | None = None, store: str = "amazon"
) -> pd.DataFrame:
    _key()
    day = (today or date.today()).isoformat()
    wanted = retail.targets()[: limit_models or None]
    return finish(store, start(store, wanted), wanted, day)


def main() -> None:
    """Collect both stores. Best Buy's job is slow, so it is started first and read last."""
    try:
        _key()
        wanted = retail.targets()[: int(os.environ.get("PDI_BRIGHTDATA_LIMIT_MODELS") or 0) or None]
    except (MissingKey, retail.NoTargets) as exc:
        print(f"::warning title=Bright Data skipped::{exc}")
        return
    day = date.today().isoformat()
    jobs = {}
    for store in ("bestbuy", "amazon"):
        try:
            jobs[store] = start(store, wanted)
        except (CollectError, requests.RequestException) as exc:
            print(f"::warning title=Bright Data {store} failed::{exc}")
    for store in ("amazon", "bestbuy"):
        if store not in jobs:
            continue
        try:
            table = finish(store, jobs[store], wanted, day)
        except (CollectError, requests.RequestException) as exc:
            # A second source going missing must not stop the benchmarks and the first source
            # from being published. The warning shows on the run page.
            print(f"::warning title=Bright Data {store} failed::{exc}")
            continue
        print(table.groupby("category")["model_key"].nunique())
        cols = ["model_key", "sale_price", "seller", "name"]
        print(table.sort_values(["model_key", "sale_price"])[cols].to_string(max_colwidth=70))


if __name__ == "__main__":
    main()
