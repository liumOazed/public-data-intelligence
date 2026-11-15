"""Temporary: pin IMF series for all six SE Asian economies; read IMF terms."""
import csv, io, collections, re, requests
H = {"User-Agent": "public-data-intelligence probe"}
B = "https://api.imf.org/external/sdmx/2.1"
CSV = {"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}
C = "IDN+MYS+PHL+SGP+THA+VNM"
def get(url, headers=None, **kw):
    try:
        r = requests.get(url, headers={**H, **(headers or {})}, timeout=180, **kw)
        return r.status_code, r.text
    except Exception as e:
        return "ERR", repr(e)

for flow, key in [("CPI", f"{C}.CPI._T.IX.M"), ("ITG", f"{C}.XG.FOB_USD.M"), ("ITG", f"{C}.MG.CIF_USD.M"),
                  ("IL", f"{C}.TRGMV_REVS.USD.M"), ("IL", f"{C}.RXF11_REVS.USD.M"), ("IL", f"{C}.RGOLDNV_REVS.USD.M"),
                  ("PI", f"{C}.IND.IX.M"), ("PI", f"{C}..IX.M"), ("LS", f"{C}.U.PT.M")]:
    st, txt = get(f"{B}/data/IMF.STA,{flow}/{key}", headers=CSV, params={"startPeriod": "1990-01"})
    print("$$", flow, key, st, len(txt))
    if st != 200:
        print("   ", txt[:200]); continue
    rows = list(csv.DictReader(io.StringIO(txt)))
    g = collections.defaultdict(list)
    for r in rows:
        k = (r["COUNTRY"], r.get("INDICATOR") or r.get("PRODUCTION_INDEX") or r.get("COICOP_1999"), r.get("TYPE_OF_TRANSFORMATION"), r.get("S_ADJUSTMENT", ""))
        g[k].append((r["TIME_PERIOD"], r["OBS_VALUE"]))
    for k, v in sorted(g.items()):
        v.sort()
        r0 = next(r for r in rows if (r["COUNTRY"], r.get("INDICATOR") or r.get("PRODUCTION_INDEX") or r.get("COICOP_1999"), r.get("TYPE_OF_TRANSFORMATION"), r.get("S_ADJUSTMENT","")) == k)
        print("   ", k, "n", len(v), "first", v[0], "last", v[-3:], "| SCALE", r0.get("SCALE"), "| NAME", (r0.get("SERIES_NAME") or "")[:80], "| UNIT", r0.get("UNIT"), "| REF", r0.get("REFERENCE_PERIOD"), "| SRC", r0.get("SHORT_SOURCE_CITATION"))
    print("   columns:", list(rows[0].keys()) if rows else None)
    if flow == "IL" and rows:
        print("   FULL_DESCRIPTION:", rows[0].get("FULL_DESCRIPTION", "")[:400])

# Code names for IL indicators
for path in ["codelist/IMF.STA/CL_IL_INDICATOR", "codelist/IMF.STA/CL_INDICATOR_IL", "codelist/all/all/latest"]:
    st, txt = get(f"{B}/{path}")
    print("@@", path, st, len(txt))
    for code in ["TRGMV_REVS", "RXF11_REVS", "RGOLDNV_REVS", "TRGNV_REVS"]:
        m = re.search(rf'id="{code}".{{0,400}}?<com:Name xml:lang="en">([^<]+)<', txt, re.S)
        if m: print("    ", code, "|", m.group(1))
    if st == 200 and len(txt) > 1000 and "all" in path:
        ids = re.findall(r'<str:Codelist [^>]*id="([^"]+)"', txt)
        print("    codelists with IL:", [i for i in ids if "IL" in i or "INDICATOR" in i][:40])

# Terms of use
for url in ["https://www.imf.org/en/About/copyright-and-terms", "https://data.imf.org/en/terms-of-use", "https://www.imf.org/external/terms.htm"]:
    st, txt = get(url)
    text = re.sub(r"<[^>]+>", " ", txt); text = re.sub(r"\s+", " ", text)
    print("TT", url, st, len(text))
    for m in re.finditer(r"(?i)[^.]{0,300}\b(data|dataset|database)s?\b[^.]{0,400}\.", text):
        s = m.group(0)
        if re.search(r"(?i)free|permi|commercial|attribut|reuse|redistribut|license|licence", s):
            print("   ", s.strip()[:700])
