"""Temporary: find IMF monthly series for TH, VN, PH."""
import re, requests, json
H = {"User-Agent": "public-data-intelligence probe"}
def get(url, **kw):
    try:
        r = requests.get(url, headers=H, timeout=90, **kw)
        return r.status_code, r.text
    except Exception as e:
        return "ERR", repr(e)

for base in ["https://api.imf.org/external/sdmx/2.1", "https://api.imf.org/external/sdmx/3.0"]:
    path = "/dataflow" if base.endswith("2.1") else "/structure/dataflow"
    st, txt = get(base + path)
    print("==", base + path, st, len(txt))
    ids = re.findall(r'<str:Dataflow[^>]*id="([^"]+)"[^>]*agencyID="([^"]+)"', txt) or re.findall(r'id="([^"]+)"[^>]*agencyID="([^"]+)"', txt)
    names = re.findall(r'<str:Dataflow[^>]*id="([^"]+)".*?<com:Name[^>]*>([^<]+)<', txt, re.S)
    for i, n in names[:400]:
        print("  FLOW", i, "|", n)
    if not names: print(txt[:1500])

B = "https://api.imf.org/external/sdmx/2.1"
for flow in ["CPI", "IMTS", "ER", "MFS_IR", "IRFCL", "ITG", "PPI", "IL", "MFS_CBS", "PI", "IIP", "LS", "QGDP", "ANEA", "MFS_MA", "MFS_FMP", "PCPS", "NSDP", "QNEA", "STA"]:
    st, txt = get(f"{B}/dataflow/IMF.STA/{flow}/+/latest", params={"references": "datastructure"})
    print("##", flow, st, len(txt))
    if st == 200:
        dims = re.findall(r'<str:Dimension id="([^"]+)"[^>]*position="(\d+)"', txt)
        print("   dims", dims)
        cl = re.findall(r'<str:Codelist[^>]*id="([^"]+)"', txt)
        print("   codelists", cl[:20])
    for key in ["THA+VNM+PHL..M", "THA+VNM+PHL...M", "THA+VNM+PHL....M", "THA+VNM+PHL.....M", "THA+VNM+PHL"]:
        st2, t2 = get(f"{B}/data/IMF.STA,{flow}/{key}", params={"startPeriod": "2025-06", "detail": "serieskeysonly"}, )
        if st2 == 200 and "Series" in t2:
            keys = re.findall(r'<(?:generic:)?SeriesKey>(.*?)</(?:generic:)?SeriesKey>', t2, re.S)
            ser = re.findall(r'<Series ([^>]*)>', t2)
            print(f"   data {key}: {st2} {len(t2)} series={len(ser) or len(keys)}")
            for s in (ser or keys)[:80]:
                print("     ", re.sub(r'\s+', ' ', s)[:300])
            break
        else:
            print(f"   data {key}: {st2} {t2[:200]!r}")
