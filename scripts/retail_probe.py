"""One-off probe: which US retailers answer a plainly identified crawler from this machine?

Prints, per site: what robots.txt says about the search path, the HTTP status of one search
page, and what price-bearing structure the page has. Nothing is stored.
"""

from __future__ import annotations

import json
import re
import urllib.robotparser
from urllib.parse import urlsplit

import requests

UA = "pdi-research-bot/0.1 (+https://github.com/liumOazed/public-data-intelligence)"
SITES = {
    "shopblt": "https://www.shopblt.com/cgi-bin/shop/shop.cgi?action=thispage&thispage=01100500U001_BMP3497P.shtml&order_id=!ORDERID!",
    "shopblt_search": "https://www.shopblt.com/search/order_id=!ORDERID!&s_max=25&t_all=1&s_all=rtx+4070",
    "provantage": "https://www.provantage.com/service/searchsvcs?QUERY=rtx+4070",
    "antonline": "https://www.antonline.com/search?q=rtx+4070",
    "superbiiz": "https://www.superbiiz.com/query.php?s=rtx+4070",
    "connection": "https://www.connection.com/product/searchpage?SearchType=1&term=rtx+4070",
    "cdw": "https://www.cdw.com/search/?key=rtx%204070",
    "zones": "https://www.zones.com/site/locate/refine.html?txt_search=rtx+4070",
    "pcnation": "https://www.pcnation.com/web/search?q=rtx+4070",
    "neobits": "https://www.neobits.com/search?keywords=rtx+4070",
    "directdial": "https://www.directdial.com/us/search/rtx%204070",
    "gamestop": "https://www.gamestop.com/search/?q=rtx+4070",
    "target": "https://www.target.com/s?searchTerm=rtx+4070",
    "officedepot": "https://www.officedepot.com/a/search/?q=rtx+4070",
    "staples": "https://www.staples.com/rtx+4070/directory_rtx%204070",
    "dell": "https://www.dell.com/en-us/search/rtx%204070",
    "amd_shop": "https://shop-us-en.amd.com/",
    "centralcomputer_gpu": "https://www.centralcomputer.com/catalogsearch/result/?q=ryzen+9800x3d",
}
LD = re.compile(r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>", re.S | re.I)
PRICE = re.compile(r"\$\s?\d[\d,]*\.\d\d")


def probe(name: str, url: str) -> None:
    print(f"\n===== {name} =====")
    parts = urlsplit(url)
    root = f"{parts.scheme}://{parts.netloc}"
    try:
        r = requests.get(root + "/robots.txt", headers={"User-Agent": UA}, timeout=20)
        print(f"robots.txt: HTTP {r.status_code}, {len(r.text)} chars")
        if r.status_code == 200:
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(r.text.splitlines())
            print(f"robots allows search url for our UA: {rp.can_fetch(UA, url)}")
            print(f"robots allows search url for '*': {rp.can_fetch('*', url)}")
            print(f"crawl-delay: {rp.crawl_delay('*')}")
    except requests.RequestException as exc:
        print(f"robots.txt: {type(exc).__name__}")
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept": "text/html"}, timeout=30)
    except requests.RequestException as exc:
        print(f"search page: {type(exc).__name__}: {str(exc)[:150]}")
        return
    text = r.text
    title = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
    print(f"search page: HTTP {r.status_code}, {len(text)} chars, final url {r.url[:120]}")
    print(f"title: {title.group(1).strip()[:120] if title else None}")
    print(f"prices found: {len(PRICE.findall(text))}, '4070' mentions: {text.count('4070')}")
    for marker in (
        "__NEXT_DATA__",
        "__initialState__",
        "__INITIAL_STATE__",
        "captcha",
        "Access Denied",
    ):
        print(f"  marker {marker!r}: {marker.lower() in text.lower()}")
    blocks = LD.findall(text)
    print(f"json-ld blocks: {len(blocks)}")
    for block in blocks[:4]:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            print("  (unparseable json-ld)")
            continue
        kind = data.get("@type") if isinstance(data, dict) else [d.get("@type") for d in data][:5]
        print(f"  type {kind}: {json.dumps(data)[:700]}")
    m = re.search(r"4070", text)
    p = PRICE.search(text, m.start() if m else 0)
    if m and p:
        start = max(0, p.start() - 500)
        print(f"context before first price after a '4070':\n{text[start : p.end() + 100]!r}")


if __name__ == "__main__":
    for site, target in SITES.items():
        probe(site, target)
