"""Fetch the two inputs of the Demand Radar experiment once: pageviews and retail sales.

Run as `python -m pdi.radar.collect`. Needs CENSUS_API_KEY.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd

from pdi.common.fetch import CollectError, fetch, utc_now
from pdi.common.paths import RAW
from pdi.radar import config

RAW_DIR = RAW / "radar"
PACE_S = 4  # the pageview API answers 429 to a fast caller


def parse_pageviews(payload: dict, article: str) -> pd.DataFrame:
    items = payload.get("items", [])
    df = pd.DataFrame(
        {
            "article": article,
            "period": pd.to_datetime([i["timestamp"][:6] for i in items], format="%Y%m"),
            "views": [int(i["views"]) for i in items],
        }
    )
    return df


def parse_sales(payload: list, category: str) -> pd.DataFrame:
    head, rows = payload[0], payload[1:]
    df = pd.DataFrame(rows, columns=head)
    df = df[(df["category_code"] == category) & (df["data_type_code"] == config.SALES_DATA_TYPE)]
    df = df[df["seasonally_adj"] == config.SALES_ADJUSTED]
    out = pd.DataFrame(
        {
            "category": category,
            "period": pd.to_datetime(df["time"], format="%Y-%m"),
            "sales_musd": pd.to_numeric(df["cell_value"], errors="coerce"),
        }
    )
    return out.dropna().drop_duplicates(["category", "period"]).sort_values("period")


def validate(views: pd.DataFrame, sales: pd.DataFrame) -> None:
    for code, (_name, articles) in config.PAIRS.items():
        got = set(views["article"])
        missing = [a for a in articles if a not in got]
        if missing:
            raise CollectError(f"no pageviews for {', '.join(missing)}")
        s = sales[sales["category"] == code]
        if len(s) < 100:
            raise CollectError(f"Census category {code}: only {len(s)} months of sales")
        if (s["sales_musd"] <= 0).any():
            raise CollectError(f"Census category {code}: a month with no sales")


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    key = os.environ.get(config.CENSUS_KEY_ENV)
    if not key:
        raise CollectError(f"{config.CENSUS_KEY_ENV} is not set")
    end = utc_now().strftime("%Y%m01") + "00"
    frames = []
    for _code, (_name, articles) in config.PAIRS.items():
        for article in articles:
            url = config.PAGEVIEWS_API.format(
                article=article, start=config.PAGEVIEWS_START, end=end
            )
            path = fetch(url, RAW_DIR / f"pageviews_{article}.json", retries=6)
            frames.append(parse_pageviews(json.loads(path.read_text()), article))
            time.sleep(PACE_S)
    views = pd.concat(frames, ignore_index=True)

    sales = []
    for code in config.PAIRS:
        params = {
            "get": "cell_value,category_code,data_type_code,seasonally_adj",
            "time": "from 2014-01",
            "category_code": code,
            "key": key,
        }
        path = fetch(config.CENSUS_API, RAW_DIR / f"marts_{code}.json", params=params)
        sales.append(parse_sales(json.loads(path.read_text()), code))
    sales = pd.concat(sales, ignore_index=True)
    validate(views, sales)
    return views, sales


def main() -> None:
    views, sales = collect()
    config.STORE_DIR.mkdir(parents=True, exist_ok=True)
    views.to_parquet(config.PAGEVIEWS, index=False)
    sales.to_parquet(config.SALES, index=False)
    (config.STORE_DIR / "retrieved.json").write_text(
        json.dumps({"wikimedia_pageviews": utc_now().isoformat(),
                    "census_marts": utc_now().isoformat()}, indent=1) + "\n"
    )  # fmt: skip
    print(
        f"stored {len(views):,} article-months for {views['article'].nunique()} articles and "
        f"{len(sales):,} category-months for {sales['category'].nunique()} kinds of business"
    )


if __name__ == "__main__":
    main()
