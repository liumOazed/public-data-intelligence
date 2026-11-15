"""Temporary: check energy sources for the sector decision."""
import csv, io, re, collections, requests, zipfile
H = {"User-Agent": "public-data-intelligence probe"}
AREAS = ["United States of America", "United States", "Indonesia", "Malaysia", "Philippines", "Singapore", "Thailand", "Viet Nam", "Vietnam"]
r = requests.get("https://files.ember-energy.org/public-downloads/generation/outputs/release_generation_monthly_global.csv", headers=H, timeout=300)
print("ember", r.status_code, len(r.content))
rows = list(csv.DictReader(io.StringIO(r.text)))
print("columns", list(rows[0].keys()))
g = collections.defaultdict(list)
for x in rows:
    if x.get("Area") in AREAS and x.get("Variable") in ("Demand", "Total Generation") and x.get("Unit") == "TWh":
        g[(x["Area"], x["Variable"])].append(x["Date"])
for k, v in sorted(g.items()):
    v.sort(); print("  ", k, "n", len(v), "first", v[0], "last", v[-1])
print("  sample", [x for x in rows if x.get("Area") == "Thailand" and x.get("Variable") == "Demand"][-1:])
for url in ["https://www.jodidata.org/oil/database/data-downloads.aspx", "https://www.jodidata.org/gas/database/data-downloads.aspx"]:
    p = requests.get(url, headers=H, timeout=60)
    print("jodi page", url, p.status_code, len(p.text))
    links = sorted(set(re.findall(r'href="([^"]+\.(?:zip|csv)[^"]*)"', p.text, re.I)))
    for l in links[:20]: print("   link", l)
    for m in re.finditer(r"(?i)[^.<>]{0,200}(terms|licen|cite|citation|free of charge|copyright)[^.<>]{0,300}", p.text):
        print("   terms?", re.sub(r"\s+", " ", m.group(0))[:400]); break
