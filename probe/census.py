import json, os, time
import requests

KEY = os.environ.get("CENSUS_API_KEY", "").strip()
print("key present:", bool(KEY), "length", len(KEY))
BASE = "https://api.census.gov/data/timeseries/intltrade/imports/hs"


def q(label, params, show=4):
    t = time.time()
    try:
        r = requests.get(BASE, params={**params, "key": KEY}, timeout=180)
    except Exception as e:
        print(label, "EXC", type(e).__name__)
        return None
    print(f"== {label}: {r.status_code} {len(r.content):,} bytes {time.time()-t:.1f}s")
    if r.status_code != 200:
        print("   ", r.text[:300].replace(KEY, "KEY") if KEY else r.text[:300])
        return None
    try:
        rows = r.json()
    except Exception:
        print("    not json:", r.text[:200])
        return None
    print("    rows", len(rows) - 1, "header", rows[0])
    for row in rows[1 : 1 + show]:
        print("   ", row)
    return rows


C = {"CN": "5700", "ID": "5600", "MY": "5570", "PH": "5650", "SG": "5590", "TH": "5490", "VN": "5520"}
G = "CTY_CODE,CTY_NAME,I_COMMODITY,I_COMMODITY_SDESC,GEN_VAL_MO,CON_VAL_MO"
# 1 one country one month hs2
rows = q("VN 2025-06 HS2", {"get": G, "time": "2025-06", "COMM_LVL": "HS2", "CTY_CODE": "5520"})
if rows:
    h = rows[0]; gi = h.index("GEN_VAL_MO")
    print("    n chapters", len(rows) - 1, "sum GEN", sum(int(r[gi]) for r in rows[1:]))
    print("    chapters", sorted({r[h.index('I_COMMODITY')] for r in rows[1:]})[:8], "...")
# 2 names check for all codes
for k, code in C.items():
    q(f"{k} total 2025-06", {"get": "CTY_CODE,CTY_NAME,GEN_VAL_MO,CON_VAL_MO", "time": "2025-06", "CTY_CODE": code}, 2)
# 3 world total
q("world total", {"get": "CTY_CODE,CTY_NAME,GEN_VAL_MO", "time": "2025-06", "CTY_CODE": "-"}, 3)
rows = q("world HS2", {"get": G, "time": "2025-06", "COMM_LVL": "HS2", "CTY_CODE": "-"}, 3)
if rows:
    gi = rows[0].index("GEN_VAL_MO")
    print("    world sum of chapters", sum(int(r[gi]) for r in rows[1:]), "n", len(rows) - 1)
# 4 multiple countries in one call
q("multi cty", {"get": G, "time": "2025-06", "COMM_LVL": "HS2", "CTY_CODE": ["5520", "5700"]}, 2)
# 5 range of time
rows = q("VN from 2013-01 HS2", {"get": "CTY_CODE,I_COMMODITY,GEN_VAL_MO", "time": "from 2013-01", "COMM_LVL": "HS2", "CTY_CODE": "5520"}, 2)
if rows:
    ti = rows[0].index("time")
    ts = sorted({r[ti] for r in rows[1:]})
    print("    months", len(ts), ts[0], ts[-1])
rows = q("world from 2013-01 HS2", {"get": "CTY_CODE,I_COMMODITY,GEN_VAL_MO", "time": "from 2013-01", "COMM_LVL": "HS2", "CTY_CODE": "-"}, 2)
if rows:
    ti = rows[0].index("time")
    ts = sorted({r[ti] for r in rows[1:]})
    print("    months", len(ts), ts[0], ts[-1])
# 6 year-to-date / annual columns and latest months
q("VN total recent", {"get": "CTY_CODE,GEN_VAL_MO,GEN_VAL_YR,CON_VAL_MO", "time": "from 2026-01", "CTY_CODE": "5520"}, 12)
q("CN total recent", {"get": "CTY_CODE,GEN_VAL_MO,GEN_VAL_YR", "time": "from 2026-01", "CTY_CODE": "5700"}, 12)
# 7 total-all-commodities row inside HS2 query?
rows = q("VN 2025-06 no COMM_LVL", {"get": "CTY_CODE,I_COMMODITY,COMM_LVL,GEN_VAL_MO", "time": "2025-06", "CTY_CODE": "5520", "I_COMMODITY": "-"}, 3)
# 8 exports endpoint not needed; variables list size
try:
    v = requests.get(BASE + "/variables.json", timeout=60).json()["variables"]
    print("variables", sorted(v)[:80])
except Exception as e:
    print("variables EXC", e)
# 9 terms page reachable
for u in ("https://www.census.gov/data/developers/about/terms-of-service.html",):
    try:
        r = requests.get(u, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        print(u, r.status_code, len(r.content))
    except Exception as e:
        print(u, "EXC", type(e).__name__)
