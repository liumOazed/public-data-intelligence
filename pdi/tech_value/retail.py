"""Retail price crawler: today's listing prices for benchmarked CPUs and graphics cards.

For each benchmarked model, search each retailer once and keep the listings whose name is
recognised as exactly that model. One Parquet file per day goes into data/store/, because
daily prices cannot be backfilled.

Rules the crawler keeps:
- robots.txt is read on every run and a disallowed page is not fetched;
- it identifies itself in the User-Agent and waits between requests;
- it does nothing to get past a bot wall. A retailer that blocks it is not a source.
"""

from __future__ import annotations

import re
import time
import urllib.robotparser
from dataclasses import dataclass
from datetime import date
from typing import Callable
from urllib.parse import quote_plus, urljoin, urlsplit

import pandas as pd
import pandera.pandas as pa
import requests
from bs4 import BeautifulSoup

from pdi.common.fetch import CollectError, utc_now
from pdi.tech_value import config, value
from pdi.tech_value.names import model_key

TIDY_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column(str, pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")),
        "source": pa.Column(str),
        "sku": pa.Column(str),
        "name": pa.Column(str),
        "category": pa.Column(str, pa.Check.isin(["cpu", "gpu"])),
        # Outside this range the number is a parsing error, not a price.
        "sale_price": pa.Column(float, pa.Check.in_range(40, 20000)),
        "online_availability": pa.Column(bool),
        "model_key": pa.Column(str),
        "url": pa.Column(str, nullable=True),
    },
    strict=False,
    coerce=True,
)

# Listings that name a part but are not the part.
_ACCESSORY = re.compile(
    r"\b(water\s?block|backplate|bracket|riser|cable|adapter|cooler for|heatsink for|"
    r"compatible with|for use with|support bracket|refurbished|open box|used)\b",
    re.I,
)


# A whole computer names its processor too. These words never appear on a bare CPU or graphics card.
_SYSTEM = re.compile(
    r"\b(mini\s?pc|workstation|desktop\s+(?:computer|pc)|gaming\s+(?:pc|desktop)|barebones?|"
    r"all-in-one|nuc|tower|motherboard|combo|bundle|windows\s+1[01]|ssd|hdd|nvme|"
    r"\d+\s?gb\s+(?:ram|ddr[345]|memory))\b",
    re.I,
)


class NoTargets(RuntimeError):
    """There is nothing to look up yet."""


@dataclass(frozen=True)
class Listing:
    name: str
    price: float
    url: str | None
    sku: str
    in_stock: bool


def parse_magento_search(html: str, base_url: str) -> list[Listing]:
    """Search-results page of a Magento 2 storefront (Central Computers).

    Each result is an element with class "product-item" holding a "product-item-link" anchor
    and a price element carrying data-price-type="finalPrice" and data-price-amount.
    """
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for item in soup.select("li.product-item, div.product-item"):
        link = item.select_one("a.product-item-link")
        price = item.select_one('[data-price-type="finalPrice"][data-price-amount]')
        if not link or not price:
            continue
        try:
            amount = float(price["data-price-amount"])
        except (KeyError, ValueError):
            continue
        box = item.select_one("[data-product-id]")
        href = link.get("href")
        out.append(
            Listing(
                name=" ".join(link.get_text(" ", strip=True).split()),
                price=amount,
                url=urljoin(base_url, href) if href else None,
                sku=str(box["data-product-id"]) if box else (href or link.get_text()),
                in_stock=item.select_one(".stock.unavailable") is None,
            )
        )
    return out


@dataclass(frozen=True)
class Retailer:
    key: str  # also the id in sources/registry.yaml
    search_url: str  # "{q}" is replaced by the URL-encoded query
    parse: Callable[[str, str], list[Listing]]


RETAILERS = (
    Retailer(
        key="centralcomputer",
        search_url="https://www.centralcomputer.com/catalogsearch/result/?q={q}",
        parse=parse_magento_search,
    ),
)


def search_term(key: str) -> str:
    """ "amd ryzen 7 9800x3d" -> "ryzen 7 9800x3d"; "nvidia rtx 4070 ti" -> "rtx 4070 ti"."""
    return key.split(" ", 1)[1]


def keep(listing: Listing, kind: str, key: str) -> bool:
    """A listing counts only if its name is recognised as exactly the model searched for."""
    if _ACCESSORY.search(listing.name) or _SYSTEM.search(listing.name):
        return False
    if listing.price < config.MIN_PART_PRICE[kind]:
        return False
    return model_key(listing.name) == (kind, key)


def targets() -> list[tuple[str, str]]:
    """(kind, model_key) for the most-benchmarked models: only what can be scored is priced."""
    if not config.BLENDER_TABLE.exists():
        raise NoTargets("no benchmark table yet; the weekly pipeline has to run first")
    bench, _ = value.benchmark_by_model(pd.read_parquet(config.BLENDER_TABLE))
    out = []
    for kind in ("cpu", "gpu"):
        part = bench[bench["kind"] == kind].sort_values("n_runs", ascending=False)
        out += [(kind, k) for k in part["model_key"].head(config.CRAWLER_MODELS_PER_KIND)]
    return out


class Robots:
    """robots.txt per host, fetched once per run. No readable robots.txt means no crawling."""

    def __init__(self, session: requests.Session):
        self.session = session
        self.cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self.cache:
            parser = None
            try:
                resp = self.session.get(root + "/robots.txt", timeout=30)
                if resp.status_code == 200:
                    parser = urllib.robotparser.RobotFileParser()
                    parser.parse(resp.text.splitlines())
            except requests.RequestException:
                pass
            self.cache[root] = parser
        parser = self.cache[root]
        return bool(parser and parser.can_fetch(config.CRAWLER_USER_AGENT, url))


def collect(today: date | None = None) -> pd.DataFrame:
    """Crawl every retailer for every target model and write today's tidy price file."""
    day = (today or date.today()).isoformat()
    session = requests.Session()
    session.headers.update({"User-Agent": config.CRAWLER_USER_AGENT, "Accept": "text/html"})
    robots = Robots(session)
    wanted = targets()
    rows, problems = [], []

    for retailer in RETAILERS:
        raw_dir = config.RAW_DIR / "retail" / retailer.key / day
        raw_dir.mkdir(parents=True, exist_ok=True)
        seen, found_models, fetched = set(), 0, 0
        for kind, key in wanted:
            url = retailer.search_url.format(q=quote_plus(search_term(key)))
            if not robots.allowed(url):
                problems.append(f"{retailer.key}: robots.txt does not allow {url}")
                break
            time.sleep(config.CRAWLER_PAUSE_S)
            try:
                resp = session.get(url, timeout=60)
            except requests.RequestException as exc:
                problems.append(f"{retailer.key}: {type(exc).__name__} for {key}")
                continue
            if resp.status_code != 200:
                problems.append(f"{retailer.key}: HTTP {resp.status_code} for {key}")
                if resp.status_code in (403, 429):
                    break  # told to go away: stop, do not retry around it
                continue
            fetched += 1
            # Raw page kept untouched for this run's artifact; never committed or published.
            (raw_dir / f"{key.replace(' ', '_')}.html").write_bytes(resp.content)
            listings = retailer.parse(resp.text, url)
            kept = [x for x in listings if keep(x, kind, key)]
            found_models += bool(kept)
            for x in kept:
                if x.sku in seen:
                    continue
                seen.add(x.sku)
                rows.append(
                    {
                        "date": day,
                        "source": retailer.key,
                        "sku": x.sku,
                        "name": x.name,
                        "category": kind,
                        "sale_price": x.price,
                        "regular_price": None,
                        "price_update_date": None,
                        "online_availability": x.in_stock,
                        "url": x.url,
                        "model_key": key,
                        "retrieved_at": utc_now().isoformat(),
                    }
                )
        print(
            f"{retailer.key}: {fetched} pages fetched, {found_models} of {len(wanted)} models "
            f"found, {len(seen)} listings kept"
        )
        if fetched and found_models < config.CRAWLER_MIN_MODELS_FOUND:
            problems.append(
                f"{retailer.key}: only {found_models} models found in {fetched} pages; "
                "the page format has probably changed"
            )

    for line in problems[:20]:
        print(f"::warning title=Retail crawler::{line}")
    if not rows:
        raise CollectError("retail crawler collected no prices: " + "; ".join(problems[:5]))
    df = TIDY_SCHEMA.validate(pd.DataFrame(rows))
    config.PRICES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PRICES_DIR / f"{day}.parquet", index=False)
    return df


if __name__ == "__main__":
    try:
        table = collect()
    except NoTargets as exc:
        # Not a failure: on a fresh repository the daily run can come before the first
        # weekly one.
        print(f"::warning title=Retail crawler skipped::{exc}")
        raise SystemExit(0) from None
    print(table.groupby(["source", "category"])["model_key"].nunique())
    print(table.sort_values("sale_price")[["source", "model_key", "sale_price", "name"]].head(30))
