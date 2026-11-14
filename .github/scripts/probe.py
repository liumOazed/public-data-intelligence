"""Temporary: show how two retailers build their search pages. Removed before merge."""

import re
import time
import urllib.robotparser

import requests
from bs4 import BeautifulSoup

UA = "pdi-research-bot/0.1 (+https://github.com/liumOazed/public-data-intelligence)"


def get(url):
    host = url.split("/")[2]
    rp = urllib.robotparser.RobotFileParser()
    r = requests.get(f"https://{host}/robots.txt", headers={"User-Agent": UA}, timeout=30)
    rp.parse(r.text.splitlines())
    print(f"\n===== {url}\nallowed={rp.can_fetch(UA, url)}")
    if not rp.can_fetch(UA, url):
        return None
    time.sleep(2)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=40)
    print("status", r.status_code, "bytes", len(r.text), "final", r.url)
    return r.text


def squash(s):
    return re.sub(r"\s+", " ", s)


for q in ("rtx+5070", "ryzen+7+9800x3d"):
    for page in ("refine", "search"):
        html = get(f"https://www.zones.com/site/locate/{page}.html?txt_search={q}")
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        prods = soup.select(".js-product")
        print(
            "js-product blocks",
            len(prods),
            "h1",
            [h.get_text(" ", strip=True) for h in soup.select("h1")][:3],
        )
        for p in prods[:2]:
            for t in p.select("svg, script, style, img"):
                t.decompose()
            print("BLOCK", squash(str(p))[:3500])
        for p in prods[:25]:
            a = p.select_one("a[href]")
            pr = p.select_one(".product-price")
            print(
                "ROW",
                squash(p.get_text(" ", strip=True))[:110],
                "|",
                pr.get_text(" ", strip=True) if pr else None,
                "|",
                a.get("href")[:70] if a else None,
            )

html = get("https://www.pcnation.com/?q=rtx%205070") or ""
for m in list(re.finditer(r"api/", html))[:12]:
    print("API", squash(html[max(0, m.start() - 150) : m.start() + 150]))
print(
    "SCRIPTS", [s.get("src") for s in BeautifulSoup(html, "html.parser").select("script[src]")][:25]
)
r = requests.get("https://www.pcnation.com/robots.txt", headers={"User-Agent": UA}, timeout=30)
print("PCN ROBOTS", squash(r.text)[:1500])
r = requests.get("https://www.zones.com/robots.txt", headers={"User-Agent": UA}, timeout=30)
print("ZONES ROBOTS", squash(r.text)[:1500])
