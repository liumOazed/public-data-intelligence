"""eBay Browse API collector: US fixed-price Brand New listings for CPUs and graphics cards.

Uses OAuth2 Client Credentials (EBAY_CLIENT_ID = App ID, EBAY_CLIENT_SECRET = Cert ID) against
the official eBay Buy Browse API (5,000 free calls/day). Queries are deduplicated by base model
family and restricted to Brand New (conditionId 1000), Buy It Now (FIXED_PRICE), US-located
items from established sellers.
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

SOURCE = "ebay_browse"


class MissingKey(RuntimeError):
    pass


def get_token(client_id: str, client_secret: str) -> str:
    """Exchange App ID and Cert ID for an OAuth2 application access token."""
    try:
        resp = requests.post(
            config.EBAY_TOKEN_URL,
            auth=(client_id, client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise CollectError(f"eBay OAuth request failed: {exc}") from exc
    if resp.status_code != 200:
        raise CollectError(f"eBay OAuth HTTP {resp.status_code}: {resp.text[:200]}")
    token = resp.json().get("access_token")
    if not token:
        raise CollectError("eBay OAuth response had no access_token")
    return str(token)


def trusted_seller(record: dict) -> bool:
    """Require an established seller with high positive feedback."""
    seller = record.get("seller") or {}
    try:
        pct = float(seller.get("feedbackPercentage", 0.0))
        score = int(seller.get("feedbackScore", 0))
    except (TypeError, ValueError):
        return False
    return pct >= config.EBAY_MIN_FEEDBACK_PCT and score >= config.EBAY_MIN_FEEDBACK_SCORE


def total_price(record: dict) -> float | None:
    """Item price plus lowest stated shipping cost in USD."""
    price_obj = record.get("price") or {}
    if price_obj.get("currency") and price_obj.get("currency") != "USD":
        return None
    try:
        base = float(price_obj["value"])
    except (KeyError, TypeError, ValueError):
        return None
    shipping = 0.0
    options = record.get("shippingOptions") or []
    if options and isinstance(options[0], dict):
        cost = options[0].get("shippingCost") or {}
        if cost.get("currency") not in (None, "USD"):
            return None
        try:
            shipping = float(cost.get("value", 0.0))
        except (TypeError, ValueError):
            shipping = 0.0
    return round(base + max(shipping, 0.0), 2)


def parse_items(records: list[dict], wanted: set[tuple[str, str]], day: str) -> list[dict]:
    """Keep Brand New, US, fixed-price bare parts matching `wanted`."""
    rows, seen = [], set()
    for r in records:
        if not isinstance(r, dict) or r.get("error"):
            continue
        cond_id = str(r.get("conditionId") or "")
        cond = str(r.get("condition") or "").strip().lower()
        if cond_id != "1000" and cond != "new":
            continue
        buying = r.get("buyingOptions") or []
        if "FIXED_PRICE" not in buying:
            continue
        loc = r.get("itemLocation") or {}
        if loc.get("country") not in (None, "US"):
            continue
        if not trusted_seller(r):
            continue
        price = total_price(r)
        if price is None:
            continue
        sku = str(r.get("itemId") or "").strip()
        title = str(r.get("title") or "").strip()
        if not sku or sku in seen or not title:
            continue
        listing = retail.Listing(
            name=title,
            price=price,
            url=r.get("itemWebUrl"),
            sku=sku,
            in_stock=True,
        )
        matched = retail.match_wanted(listing, wanted)
        if not matched:
            continue
        kind, key = matched
        seen.add(sku)
        rows.append(
            {
                "date": day,
                "source": SOURCE,
                "sku": sku,
                "name": title,
                "category": kind,
                "sale_price": price,
                "regular_price": None,
                "price_update_date": None,
                "online_availability": True,
                "url": listing.url,
                "model_key": key,
                "seller": (r.get("seller") or {}).get("username"),
                "retrieved_at": utc_now().isoformat(),
            }
        )
    return rows


def search_pairs(wanted: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Deduplicated (kind, base_query) pairs preserving priority order."""
    return list(dict.fromkeys((kind, retail.base_query(key)) for kind, key in wanted))


def collect(today: date | None = None, pause_s: float = 0.25) -> pd.DataFrame:
    """Query eBay Browse API for active target families and write today's tidy price file."""
    client_id = os.environ.get(config.EBAY_CLIENT_ID_ENV, "").strip()
    client_secret = os.environ.get(config.EBAY_CLIENT_SECRET_ENV, "").strip()
    if not client_id or not client_secret:
        raise MissingKey(
            f"{config.EBAY_CLIENT_ID_ENV} and {config.EBAY_CLIENT_SECRET_ENV} are not set"
        )
    day = (today or date.today()).isoformat()
    wanted = retail.targets()
    acceptable = retail.all_targets() | set(wanted)
    token = get_token(client_id, client_secret)

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
            "Accept": "application/json",
        }
    )

    all_records: list[dict] = []
    for kind, query in search_pairs(wanted):
        time.sleep(pause_s)
        try:
            resp = session.get(
                config.EBAY_BROWSE_URL,
                params={
                    "q": query,
                    "category_ids": config.EBAY_CATEGORIES[kind],
                    "filter": (
                        "conditionIds:{1000},buyingOptions:{FIXED_PRICE},"
                        "itemLocationCountry:US,priceCurrency:USD"
                    ),
                    "limit": str(config.EBAY_RESULTS_PER_QUERY),
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            print(f"::warning title=eBay Browse API::{exc} for {query}")
            continue
        if resp.status_code != 200:
            print(f"::warning title=eBay Browse API::HTTP {resp.status_code} for {query}")
            if resp.status_code in (401, 403, 429):
                break
            continue
        body = resp.json()
        all_records.extend(body.get("itemSummaries") or [])

    raw_dir = config.RAW_DIR / "ebay" / day
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "items.json").write_text(json.dumps(all_records))

    rows = parse_items(all_records, acceptable, day)
    if not rows:
        raise CollectError(f"eBay Browse API returned {len(all_records)} records, 0 usable")
    df = retail.TIDY_SCHEMA.validate(pd.DataFrame(rows))
    config.PRICES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PRICES_DIR / f"{day}_ebay.parquet", index=False)
    return df


def main() -> None:
    try:
        df = collect()
    except (MissingKey, retail.NoTargets, CollectError) as exc:
        print(f"::warning title=eBay collector skipped::{exc}")
        return
    print(
        f"{SOURCE}: {len(df)} listings across {df['model_key'].nunique()} models "
        f"({df['online_availability'].sum()} in stock)"
    )


if __name__ == "__main__":
    main()
