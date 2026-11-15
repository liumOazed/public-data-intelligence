"""Temporary: pin IMF series for TH, VN, PH."""
import re, requests
H = {"User-Agent": "public-data-intelligence probe"}
B = "https://api.imf.org/external/sdmx/2.1"
def get(url, headers=None, **kw):
    try:
        r = requests.get(url, headers={**H, **(headers or {})}, timeout=120, **kw)
        return r.status_code, r.text
    except Exception as e:
        return "ERR", repr(e)

# 1. Structure: dimension order and indicator names
for flow in ["CPI", "PI", "IL", "ITG", "IMTS", "ER", "LS", "PPI"]:
    st, txt = get(f"{B}/dataflow/IMF.STA/{flow}/+", params={"references": "all"})
    dims = re.findall(r'<str:Dimension [^>]*id="([^"]+)"[^>]*position="(\d+)"', txt) or re.findall(r'<str:Dimension [^>]*position="(\d+)"[^>]*id="([^"]+)"', txt)
    print("##", flow, st, len(txt), "dims", dims)
    want = {"IL": ["RXF11_REVS","TRGMV_REVS","TRGNV_REVS","RGOLDNV_REVS","RXF11FX_REVS","TRRPIMF_REVS"],
            "CPI": [], "PI": [], "ITG": ["XG","MG"], "LS": ["U"], "PPI": ["PPI"]}.get(flow, [])
    for code, name in re.findall(r'<str:Code id="([^"]+)"[^>]*>\s*<com:Name xml:lang="en">([^<]+)<', txt):
        if code in want or (flow in ("CPI", "PI") and len(code) < 30 and not re.match(r'^[A-Z]{3}$', code)):
            print("   code", code, "|", name[:150])
    # license / terms annotations
    for m in re.findall(r'(?i)(licen[cs]e[^<]{0,200}|terms[^<]{0,200}|copyright[^<]{0,200})', txt)[:5]:
        print("   terms?", m)

# 2. Data in CSV form for candidate keys, latest periods
CSV = {"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}
cands = [
    ("CPI", "THA+VNM+PHL...M"), ("CPI", "THA+VNM+PHL....M"), ("CPI", "THA+VNM+PHL..M"),
    ("PI", "THA+VNM+PHL...M"), ("PI", "THA+VNM+PHL..M"),
    ("IL", "THA+VNM+PHL.RXF11_REVS.USD.M"), ("IL", "THA+VNM+PHL.TRGMV_REVS.USD.M"),
    ("ITG", "THA+VNM+PHL.XG.FOB_USD.M"), ("ITG", "THA+VNM+PHL.MG.CIF_USD.M"),
    ("ER", "THA+VNM+PHL.XDC_USD.PA_RT.M"), ("LS", "THA+VNM+PHL.U.PT.M"),
    ("PPI", "THA+VNM+PHL.PPI.IX.M"),
]
for flow, key in cands:
    st, txt = get(f"{B}/data/IMF.STA,{flow}/{key}", headers=CSV, params={"startPeriod": "2024-01"})
    print("$$", flow, key, st, len(txt))
    lines = txt.splitlines()
    print("   header:", lines[0][:300] if lines else "")
    if len(lines) > 1:
        # last period per series
        import csv, io, collections
        rows = list(csv.DictReader(io.StringIO(txt)))
        last = collections.defaultdict(str); n = collections.Counter()
        for r in rows:
            k = tuple((c, r[c]) for c in r if c not in ("TIME_PERIOD", "OBS_VALUE") and c.isupper() and len(r[c]) < 40 and c not in ("STRUCTURE","STRUCTURE_ID","ACTION","STRUCTURE_NAME"))
            n[k] += 1
            if r.get("TIME_PERIOD","") > last[k]: last[k] = r["TIME_PERIOD"]
        for k in list(last)[:30]:
            print("   ", dict(k), "n", n[k], "last", last[k])
        print("   sample row:", lines[1][:400])
    else:
        print("   body:", txt[:300])
