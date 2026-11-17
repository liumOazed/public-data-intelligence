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

import os
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
from pdi.tech_value.names import active_keys, model_key, newest_keys

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
    r"\b(water\s?block|riser|cooler for|heatsink for|"
    r"compatible with|for use with|support bracket|gpu bracket|vertical bracket|"
    r"replacement\s+backplate|with\s+backplate|power\s+cable|extension\s+cable|adapter\s+cable|"
    r"no\s+(?:cpu|gpu|processor|graphics\s+card|video\s+card)|"
    r"box\s+only|empty\s+box|fan\s+only|cooler\s+only|heatsink\s+only|"
    r"refurbished|renewed|open(?:ed)?[\s-]?box|pre[\s-]?owned|\bused\b)\b",
    re.I,
)


# A whole computer names its processor too. "Gaming Desktop Processor" is Intel's box wording for
# a bare CPU, not a desktop PC; "Workstation Edition" / "Workstation Graphics" is a workstation GPU.
_SYSTEM = re.compile(
    r"\b(mini\s?pc|(?<!for\s)workstation(?!\s+(?:edition|graphics|gpu|video))|"
    r"desktop\s+(?:computer|pc)|"
    r"(?<!for\s)(?<!for\scomputer\s)gaming\s+(?:pc\b|desktop(?!\s+processor))|barebones?|"
    r"egpu|external\s+gpu|"
    r"all-in-one|nuc|tower|motherboard|combo|bundle|windows\s+1[01]|ssd|hdd|nvme|"
    r"\d+\s?gb\s+(?:ram|ddr[345]|memory))\b|"
    r"\+\s*(?:amd|ryzen|intel|core|nvidia|geforce|radeon|rtx|rx)\b",
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
    """Search-results or category page of a Magento 2 storefront (Central Computers).

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


def parse_magento_next_page(html: str, base_url: str) -> str | None:
    """Next-page link on a Magento 2 category or search listing page, if any."""
    soup = BeautifulSoup(html, "html.parser")
    nxt = soup.select_one("li.pages-item-next a.action.next[href], a.action.next[href]")
    if not nxt:
        return None
    href = (nxt.get("href") or "").strip()
    if not href or href.startswith("#"):
        return None
    return urljoin(base_url, href)


MAX_CATEGORY_PAGES = 5


@dataclass(frozen=True)
class Retailer:
    key: str  # also the id in sources/registry.yaml
    search_url: str  # "{q}" is replaced by the URL-encoded query
    parse: Callable[[str, str], list[Listing]]
    category_urls: tuple[str, ...] = ()
    next_page: Callable[[str, str], str | None] | None = None


RETAILERS = (
    Retailer(
        key="centralcomputer",
        search_url="https://www.centralcomputer.com/catalogsearch/result/?q={q}",
        parse=parse_magento_search,
        category_urls=(
            "https://www.centralcomputer.com/all-products/hardware/video-cards.html?product_list_limit=96",
            "https://www.centralcomputer.com/all-products/ai-components/ai-gpus.html?product_list_limit=96",
            "https://www.centralcomputer.com/all-products/hardware/cpus.html?product_list_limit=96",
            "https://www.centralcomputer.com/all-products/ai-components/cpus.html?product_list_limit=96",
        ),
        next_page=parse_magento_next_page,
    ),
)


def search_term(key: str) -> str:
    """ "amd ryzen 7 9800x3d" -> "ryzen 7 9800x3d"; "nvidia rtx 4070 ti" -> "rtx 4070 ti"."""
    return key.split(" ", 1)[1]


def base_query(key: str) -> str:
    """Collapse closely related SKU variants into one search query.

    A search for "rtx 5070" returns both RTX 5070 and RTX 5070 Ti; a search for "core i7-14700k"
    returns 14700K, 14700KF and 14700F.
    """
    term = search_term(key)
    term = re.sub(r"\s+blackwell$", "", term)
    term = re.sub(r"\s+(?:ti|super|xtx|xt|gre)(?:\s+(?:ti|super))?$", "", term)
    term = re.sub(r"^((?:ryzen|threadripper)\s+(?:\d\s+)?\d{4})[a-z0-9]*$", r"\1", term)
    term = re.sub(r"^(core\s+i[3579]-\d{4,5}k)f$", r"\1", term)
    term = re.sub(r"^(core\s+ultra\s+[3579]\s+\d{3}k)f$", r"\1", term)
    return term


def keep(listing: Listing, kind: str, key: str) -> bool:
    """A listing counts only if its name is recognised as exactly the model searched for."""
    if _ACCESSORY.search(listing.name) or _SYSTEM.search(listing.name):
        return False
    if not (config.MIN_PART_PRICE[kind] <= listing.price <= config.MAX_PART_PRICE):
        return False
    return model_key(listing.name) == (kind, key)


def match_wanted(listing: Listing, wanted: set[tuple[str, str]]) -> tuple[str, str] | None:
    """Return (kind, model_key) when a listing is a bare part in `wanted`."""
    found = model_key(listing.name)
    if found not in wanted:
        return None
    kind, key = found
    return found if keep(listing, kind, key) else None


def all_targets() -> set[tuple[str, str]]:
    """Every (kind, model_key) with a valid benchmark score."""
    if not config.BLENDER_TABLE.exists():
        raise NoTargets("no benchmark table yet; the weekly pipeline has to run first")
    bench, _ = value.benchmark_by_model(pd.read_parquet(config.BLENDER_TABLE))
    return set(zip(bench["kind"], bench["model_key"], strict=False))


def recent_market_models() -> set[tuple[str, str]]:
    """(kind, model_key) seen in stock in the recent price window, if any price files exist."""
    prices = value.load_prices()
    if prices.empty:
        return set()
    latest = pd.to_datetime(prices["date"]).max()
    cutoff = (latest - pd.Timedelta(days=config.PRICE_MAX_AGE_DAYS)).strftime("%Y-%m-%d")
    active_rows = prices[(prices["date"] >= cutoff) & prices["online_availability"].astype(bool)]
    return set(zip(active_rows["category"], active_rows["model_key"], strict=False))


def targets() -> list[tuple[str, str]]:
    """(kind, model_key) dynamically chosen from latest benchmark generations and retail stock."""
    if not config.BLENDER_TABLE.exists():
        raise NoTargets("no benchmark table yet; the weekly pipeline has to run first")
    bench, _ = value.benchmark_by_model(pd.read_parquet(config.BLENDER_TABLE))
    all_pairs = list(zip(bench["kind"], bench["model_key"], strict=False))
    active = active_keys(all_pairs)
    newest = newest_keys(all_pairs)
    in_market = recent_market_models()

    out = []
    for kind in ("cpu", "gpu"):
        part = bench[bench["kind"] == kind].copy()
        eligible = part[part["model_key"].map(lambda k, kd=kind: (kd, k) in active)].copy()
        chosen = eligible if len(eligible) else part.copy()
        # Boost models in the #1 newest generation (3x) and models confirmed in recent retail
        # stock (2x) so newly launched chips enter the top-N immediately and dead SKUs sink.
        chosen["priority_score"] = [
            runs
            * (3 if (kind, k) in newest else 1)
            * (2 if (not in_market or (kind, k) in in_market) else 1)
            for k, runs in zip(chosen["model_key"], chosen["n_runs"], strict=False)
        ]
        chosen = chosen.sort_values(["priority_score", "n_runs"], ascending=[False, False])
        out += [(kind, k) for k in chosen["model_key"].head(config.CRAWLER_MODELS_PER_KIND)]
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


def _ingest_listings(
    listings: list[Listing],
    acceptable: set[tuple[str, str]],
    day: str,
    source: str,
    seen: set[str],
    found_models: set[tuple[str, str]],
    rows: list[dict],
) -> None:
    for x in listings:
        matched = match_wanted(x, acceptable)
        if not matched or x.sku in seen:
            continue
        m_kind, m_key = matched
        seen.add(x.sku)
        found_models.add((m_kind, m_key))
        rows.append(
            {
                "date": day,
                "source": source,
                "sku": x.sku,
                "name": x.name,
                "category": m_kind,
                "sale_price": x.price,
                "regular_price": None,
                "price_update_date": None,
                "online_availability": x.in_stock,
                "url": x.url,
                "model_key": m_key,
                "retrieved_at": utc_now().isoformat(),
            }
        )


def collect(today: date | None = None) -> pd.DataFrame:
    """Crawl every retailer's category pages and fallback searches to write today's price file."""
    day = (today or date.today()).isoformat()
    session = requests.Session()
    session.headers.update({"User-Agent": config.CRAWLER_USER_AGENT, "Accept": "text/html"})
    robots = Robots(session)
    wanted = targets()
    acceptable = all_targets() | set(wanted)
    rows, problems = [], []

    for retailer in RETAILERS:
        raw_dir = config.RAW_DIR / "retail" / retailer.key / day
        raw_dir.mkdir(parents=True, exist_ok=True)
        seen, found_models, fetched, blocked = set(), set(), 0, False
        visited_urls: set[str] = set()

        # 1. Crawl full category listing pages first (with pagination) to capture all in-stock SKUs
        #    in a few requests instead of querying every model one by one.
        for cat_idx, start_url in enumerate(retailer.category_urls, start=1):
            url: str | None = start_url
            page_no = 1
            while url and url not in visited_urls and page_no <= MAX_CATEGORY_PAGES:
                visited_urls.add(url)
                if not robots.allowed(url):
                    problems.append(f"{retailer.key}: robots.txt does not allow {url}")
                    break
                time.sleep(config.CRAWLER_PAUSE_S)
                try:
                    resp = session.get(url, timeout=60)
                except requests.RequestException as exc:
                    problems.append(f"{retailer.key}: {type(exc).__name__} for category {cat_idx}")
                    break
                if resp.status_code != 200:
                    problems.append(
                        f"{retailer.key}: HTTP {resp.status_code} for category {cat_idx}"
                    )
                    if resp.status_code in (403, 429):
                        blocked = True
                    break
                fetched += 1
                (raw_dir / f"category_{cat_idx}_p{page_no}.html").write_bytes(resp.content)
                _ingest_listings(
                    retailer.parse(resp.text, url),
                    acceptable,
                    day,
                    retailer.key,
                    seen,
                    found_models,
                    rows,
                )
                url = retailer.next_page(resp.text, url) if retailer.next_page else None
                page_no += 1
            if blocked:
                break

        # 2. Fallback keyword search only for target models not found on category pages,
        #    deduplicated by base_query so variant families share a single search request.
        queried_terms: set[str] = set()
        for kind, key in wanted:
            if blocked or (kind, key) in found_models:
                continue
            q_term = base_query(key)
            if q_term in queried_terms:
                continue
            queried_terms.add(q_term)
            url = retailer.search_url.format(q=quote_plus(q_term))
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
            (raw_dir / f"{q_term.replace(' ', '_')}.html").write_bytes(resp.content)
            _ingest_listings(
                retailer.parse(resp.text, url),
                acceptable,
                day,
                retailer.key,
                seen,
                found_models,
                rows,
            )

        n_found = len(found_models)
        print(
            f"{retailer.key}: {fetched} pages fetched, {n_found} models "
            f"found ({len(wanted)} targets), {len(seen)} listings kept"
        )
        if fetched and n_found < config.CRAWLER_MIN_MODELS_FOUND:
            problems.append(
                f"{retailer.key}: only {n_found} models found in {fetched} pages; "
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
    if os.environ.get("PDI_FIRE_DRILL") == "1":
        # A deliberate failure, started by hand, to prove a broken collector stops the
        # publish and opens an Issue. Nothing is fetched.
        raise SystemExit("Fire drill: the retail crawler was told to fail.")
    try:
        table = collect()
    except NoTargets as exc:
        # Not a failure: on a fresh repository the daily run can come before the first
        # weekly one.
        print(f"::warning title=Retail crawler skipped::{exc}")
        raise SystemExit(0) from None
    print(table.groupby(["source", "category"])["model_key"].nunique())
    print(table.sort_values("sale_price")[["source", "model_key", "sale_price", "name"]].head(30))
