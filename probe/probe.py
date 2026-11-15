"""Temporary: check energy sources for the sector decision."""
import csv, io, re, collections, requests
H = {"User-Agent": "public-data-intelligence probe"}
ISO = ["USA", "IDN", "MYS", "PHL", "SGP", "THA", "VNM"]
r = requests.get("https://files.ember-energy.org/public-downloads/generation/outputs/release_generation_monthly_global.csv", headers=H, timeout=300)
rows = list(csv.DictReader(io.StringIO(r.text)))
print("sources", sorted({x["Electricity source"] for x in rows})[:40])
g = collections.defaultdict(list)
for x in rows:
    if x["ISO 3 code"] in ISO and x["Electricity source"] in ("Demand", "Total Generation"):
        g[(x["ISO 3 code"], x["Electricity source"])].append((x["Date"], x["Generation (TWh)"]))
for k, v in sorted(g.items()):
    v.sort(); print("  ", k, "n", len(v), "first", v[0], "last", v[-2:])
p = requests.get("https://www.jodidata.org/oil/database/data-downloads.aspx", headers=H, timeout=60)
links = sorted(set(re.findall(r'href="([^"]+\.(?:zip|csv)[^"]*)"', p.text, re.I)))
print("jodi links", links[-12:])
txt = re.sub(r"<[^>]+>", " ", p.text); txt = re.sub(r"\s+", " ", txt)
for kw in ["free", "cite", "citation", "terms", "conditions", "reproduc"]:
    for m in re.finditer(rf"(?i).{{0,200}}{kw}.{{0,250}}", txt):
        print("   ", kw, "|", m.group(0)[:450]); break
for l in [l for l in links if "2026" in l or "2025" in l][:3]:
    u = "https://www.jodidata.org" + l if l.startswith("/") else l
    d = requests.get(u, headers=H, timeout=300)
    print("file", u, d.status_code, len(d.content))
    lines = d.text.splitlines()
    print("  header", lines[0][:300])
    hits = [ln for ln in lines if any(ln.startswith(c) or f",{c}," in ln[:20] for c in ["TH", "VN", "PH", "MY", "ID", "SG", "US"])]
    print("  sea/us rows", len(hits)); print("  ", hits[:3])
    months = collections.defaultdict(set)
    for ln in hits:
        parts = ln.split(",")
        months[parts[0]].add(parts[1])
    print("  months by area", {k: (min(v), max(v), len(v)) for k, v in months.items()})
    flows = collections.Counter(ln.split(",")[3] + "/" + ln.split(",")[2] for ln in hits if len(ln.split(",")) > 4)
    print("  flows", flows.most_common(15))
