"""Temporary: show how two retailers build their search pages. Removed before merge."""

import re
import time
import urllib.robotparser

import requests
from bs4 import BeautifulSoup

UA = "pdi-research-bot/0.1 (+https://github.com/liumOazed/public-data-intelligence)"
URLS = [
    "https://www.zones.com/site/locate/refine.html?txt_search=rtx+5070",
    "https://www.pcnation.com/web/search?q=rtx+5070",
    "https://www.pcnation.com/?q=rtx%205070",
]


def path(el, depth=5):
    out = []
    while el is not None and el.name and len(out) < depth:
        cls = ".".join(el.get("class", [])[:3])
        out.append(f"{el.name}{'.' + cls if cls else ''}")
        el = el.parent
    return " < ".join(out)


for url in URLS:
    host = url.split("/")[2]
    rp = urllib.robotparser.RobotFileParser()
    try:
        r = requests.get(f"https://{host}/robots.txt", headers={"User-Agent": UA}, timeout=30)
        rp.parse(r.text.splitlines())
        print(f"\n===== {url}\nrobots {r.status_code} allowed={rp.can_fetch(UA, url)}")
        if not rp.can_fetch(UA, url):
            continue
        time.sleep(2)
        r = requests.get(url, headers={"User-Agent": UA}, timeout=40)
    except Exception as e:  # noqa: BLE001
        print("error", e)
        continue
    html = r.text
    print("status", r.status_code, "bytes", len(html), "final", r.url)
    soup = BeautifulSoup(html, "html.parser")
    print("title", soup.title.string if soup.title else None)
    print("ld+json blocks", len(soup.select('script[type="application/ld+json"]')))
    seen = {}
    for node in soup.find_all(string=re.compile(r"\$\s?\d[\d,]*\.\d\d")):
        p = path(node.parent)
        seen[p] = seen.get(p, 0) + 1
    for p, n in sorted(seen.items(), key=lambda x: -x[1])[:8]:
        print("PRICE", n, p)
    seen = {}
    for a in soup.find_all("a", string=re.compile(r"5070", re.I)):
        p = path(a) + " href=" + str(a.get("href"))[:60]
        seen[path(a)] = (seen.get(path(a), (0, ""))[0] + 1, p)
    for _, (n, p) in sorted(seen.items(), key=lambda x: -x[1][0])[:6]:
        print("LINK", n, p)
    m = re.search(r"\$\s?\d[\d,]*\.\d\d", html)
    if m:
        s = max(0, m.start() - 3500)
        print("SNIPPET", re.sub(r"\s+", " ", html[s : m.start() + 1500]))
    for pat in ("__NEXT_DATA__", "application/json", "api/", "algolia", "searchspring", "klevu"):
        print("has", pat, pat in html)
    time.sleep(2)
