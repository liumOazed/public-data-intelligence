"""Best Buy Products API: the daily US price snapshot for CPUs and graphics cards.

Daily prices cannot be backfilled, so this collector matters more than its size suggests. It
writes one small Parquet file per day into data/store/, which the workflow commits. The API
key comes from the BESTBUY_API_KEY environment variable (an Actions secret) and is never
written to disk: the saved URL has it redacted.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from pdi.common.fetch import CollectError, fetch, read_meta
from pdi.tech_value import config
from pdi.tech_value.names import model_key

TIDY_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column(str, pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")),
        "sku": pa.Column(str),
        "name": pa.Column(str),
        "category": pa.Column(str, pa.Check.isin(list(config.BESTBUY_CATEGORIES))),
        # A price outside this range for a CPU or GPU is a unit or parsing error, not a deal.
        "sale_price": pa.Column(float, pa.Check.in_range(5, 20000)),
        "regular_price": pa.Column(float, pa.Check.in_range(5, 20000), nullable=True),
        "price_update_date": pa.Column(str, nullable=True),
        "online_availability": pa.Column(bool),
        "model_key": pa.Column(str, nullable=True),
    },
    strict=False,
    coerce=True,
)


class MissingKey(RuntimeError):
    pass


def collect(today: date | None = None, pause_s: float = 0.3) -> list[Path]:
    """Save every page of both categories, untouched. Returns the raw page paths."""
    key = os.environ.get(config.BESTBUY_KEY_ENV, "").strip()
    if not key:
        raise MissingKey(f"{config.BESTBUY_KEY_ENV} is not set")
    today = today or date.today()
    out_dir = config.RAW_DIR / "bestbuy" / today.isoformat()
    paths: list[Path] = []
    for category, category_id in config.BESTBUY_CATEGORIES.items():
        page, total_pages = 1, 1
        while page <= total_pages:
            dest = out_dir / f"{category}_{page:03d}.json"
            fetch(
                f"{config.BESTBUY_ENDPOINT}(categoryPath.id={category_id})",
                dest,
                params={
                    "apiKey": key,
                    "format": "json",
                    "show": config.BESTBUY_FIELDS,
                    "pageSize": 100,
                    "page": page,
                },
            )
            body = json.loads(dest.read_bytes())
            if "products" not in body:
                raise CollectError(f"Best Buy page has no 'products': keys {list(body)}")
            total_pages = int(body.get("totalPages", 1))
            paths.append(dest)
            page += 1
            time.sleep(pause_s)
    return paths


def parse_page(body: dict, category: str, day: str) -> list[dict]:
    rows = []
    for p in body["products"]:
        if str(p.get("condition", "New")).lower() != "new":
            continue  # refurbished and open-box prices are a different measurement
        if p.get("salePrice") is None:
            continue
        key = model_key(p.get("name"))
        rows.append(
            {
                "date": day,
                "source": "bestbuy_products",
                "sku": str(p["sku"]),
                "name": p.get("name"),
                "manufacturer": p.get("manufacturer"),
                "model_number": p.get("modelNumber"),
                "category": category,
                "regular_price": p.get("regularPrice"),
                "sale_price": p.get("salePrice"),
                "on_sale": bool(p.get("onSale", False)),
                # Best Buy's own date for when this price last changed.
                "price_update_date": (p.get("priceUpdateDate") or "")[:10] or None,
                "online_availability": bool(p.get("onlineAvailability", False)),
                "url": p.get("url"),
                # A listing in the CPU category only keeps a CPU key, and likewise for GPUs.
                "model_key": key[1] if key and key[0] == category else None,
            }
        )
    return rows


def transform(day: str | None = None) -> pd.DataFrame:
    """Raw pages for one day -> data/store/tech_value/prices/<day>.parquet."""
    root = config.RAW_DIR / "bestbuy"
    day = day or sorted(p.name for p in root.iterdir() if p.is_dir())[-1]
    rows, retrieved = [], []
    for path in sorted((root / day).glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        category = path.name.split("_")[0]
        rows += parse_page(json.loads(path.read_bytes()), category, day)
        retrieved.append(read_meta(path)["retrieved_at"])
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"no usable Best Buy products for {day}")
    df = df.drop_duplicates(subset=["sku"], keep="last")
    for category, part in df.groupby("category"):
        share = part["model_key"].notna().mean()
        if share < config.BESTBUY_MIN_RECOGNISED:
            raise ValueError(
                f"only {share:.0%} of Best Buy '{category}' products look like a {category}; "
                f"category id {config.BESTBUY_CATEGORIES[category]} is probably wrong"
            )
    df["retrieved_at"] = max(retrieved)
    df = TIDY_SCHEMA.validate(df)
    config.PRICES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PRICES_DIR / f"{day}.parquet", index=False)
    return df


if __name__ == "__main__":
    try:
        pages = collect()
    except MissingKey as exc:
        # Not a failure: until the secret exists the rest of the module still runs, and the
        # site says that no price has been collected yet.
        print(f"::warning title=Price collector skipped::{exc}. No prices collected today.")
    else:
        print(f"collected {len(pages)} pages; tidy rows: {len(transform())}")
