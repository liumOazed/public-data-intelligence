"""Temporary: find more retailers that allow a polite crawler. Removed before merge."""

import re
import time
import urllib.robotparser

import requests
from bs4 import BeautifulSoup

UA = "pdi-research-bot/0.1 (+https://github.com/liumOazed/public-data-intelligence)"
Q = "rtx+5070"
URLS = [
    f"https://www.superbiiz.com/query.php?s={Q}",
    f"https://www.outletpc.com/search?q={Q}",
    f"https://www.mwave.com/mwave/search?q={Q}",
    f"https://www.wiredzone.com/search?query={Q}",
    f"https://www.sabrepc.com/catalogsearch/result/?q={Q}",
    f"https://www.performance-pcs.com/catalogsearch/result/?q={Q}",
    f"https://www.zotacstore.com/us/catalogsearch/result/?q={Q}",
    f"https://mitxpc.com/search?q={Q}",
    f"https://www.abt.com/resources/pages/search.php?keywords={Q}",
    f"https://shop.asus.com/us/catalogsearch/result/?q={Q}",
    f"https://us-store.msi.com/index.php?route=product/search&search={Q}",
    f"https://www.pny.com/search?q={Q}",
    f"https://www.neweggbusiness.com/product/productlist.aspx?Submit=ENE&Description={Q}",
    f"https://www.compsource.com/search/?q={Q}",
    f"https://www.insight.com/en_US/search.html?q={Q}",
    f"https://www.shi.com/search?q={Q}",
    f"https://www.buydig.com/shop/search/?q={Q}",
    f"https://www.beachcamera.com/shop/search?q={Q}",
    f"https://www.focuscamera.com/catalogsearch/result/?q={Q}",
    f"https://www.electronicexpress.com/search?q={Q}",
    f"https://www.serversupply.com/search.asp?keyword={Q}",
    f"https://www.pcrichard.com/search?q={Q}",
    f"https://www.nfm.com/search?q={Q}",
    f"https://www.bjs.com/search/{Q}",
    f"https://www.costco.com/CatalogSearch?keyword={Q}",
    f"https://www.samsclub.com/s/{Q}",
    f"https://www.tigerdirect.com/applications/SearchTools/search.asp?keywords={Q}",
    f"https://www.ebuyer.com/search?q={Q}",
    f"https://www.cyberpowerpc.com/search?q={Q}",
    f"https://www.govconnection.com/IPA/Shop/Product/Search?term={Q}",
]


def path(el, depth=4):
    out = []
    while el is not None and el.name and len(out) < depth:
        cls = ".".join(el.get("class", [])[:3])
        out.append(f"{el.name}{'.' + cls if cls else ''}")
        el = el.parent
    return " < ".join(out)


def digest(soup, nodes, n=3):
    seen = {}
    for node in nodes:
        p = path(node)
        seen[p] = seen.get(p, 0) + 1
    return sorted(seen.items(), key=lambda x: -x[1])[:n]


for url in URLS:
    host = url.split("/")[2]
    try:
        rp = urllib.robotparser.RobotFileParser()
        r = requests.get(f"https://{host}/robots.txt", headers={"User-Agent": UA}, timeout=15)
        if r.status_code != 200:
            print(f"{host}: robots {r.status_code}")
            continue
        rp.parse(r.text.splitlines())
        if not rp.can_fetch(UA, url):
            print(f"{host}: robots disallow")
            continue
        time.sleep(1)
        r = requests.get(url, headers={"User-Agent": UA}, timeout=25)
    except Exception as e:  # noqa: BLE001
        print(f"{host}: error {type(e).__name__}")
        continue
    html = r.text
    prices = len(re.findall(r"\$\s?\d[\d,]*\.\d\d", html))
    hits = len(re.findall(r"5070", html))
    hints = [
        h
        for h in (
            "product-item-link",
            "cdn.shopify",
            "woocommerce",
            "bigcommerce",
            "__NEXT_DATA__",
            "ld+json",
        )
        if h in html
    ]
    print(
        f"{host}: status {r.status_code} bytes {len(html)} prices {prices} hits5070 {hits} hints {hints} final {r.url[:90]}"
    )
    if r.status_code == 200 and prices >= 5 and hits >= 5:
        soup = BeautifulSoup(html, "html.parser")
        print(
            "   PRICE",
            digest(
                soup, [n.parent for n in soup.find_all(string=re.compile(r"\$\s?\d[\d,]*\.\d\d"))]
            ),
        )
        print("   LINK", digest(soup, [a for a in soup.find_all("a") if "5070" in a.get_text()]))
        a = next((a for a in soup.find_all("a") if "5070" in a.get_text()), None)
        if a:
            box = a
            for _ in range(4):
                box = box.parent or box
            for t in box.select("svg, script, style, img"):
                t.decompose()
            print("   BLOCK", re.sub(r"\s+", " ", str(box))[:1800])
