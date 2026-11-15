import json
import time

import requests

B = "https://api.usaspending.gov"
H = {"User-Agent": "public-data-intelligence (+https://github.com/liumOazed/public-data-intelligence)"}


def call(method, path, body=None):
    r = requests.request(method, B + path, json=body, headers=H, timeout=180)
    try:
        return r.json()
    except Exception:
        print("   ", r.status_code, r.text[:300])
        return {}


seen = {}
for t in ["7A", "7B", "7C", "7D", "7E", "7F", "7G", "7H", "7J", "7K", "IT AND TELECOM"]:
    j = call("POST", "/api/v2/autocomplete/psc/", {"search_text": t, "limit": 100})
    for x in j.get("results", []):
        seen[x["product_or_service_code"]] = x["psc_description"]
for k in sorted(seen):
    print("  AC", k, seen[k])

CONTRACTS = ["A", "B", "C", "D"]
cand = sorted({f"7{a}{n}" for a in "ABCDEFGHJK" for n in ("20", "21", "22", "23", "24", "25")} | {"7010", "7020", "7021", "7022", "7025", "7030", "7035", "7040", "7042", "7045", "7050"})
for fy in (2020, 2021, 2023, 2025, 2026):
    F = {"award_type_codes": CONTRACTS, "time_period": [{"start_date": f"{fy-1}-10-01", "end_date": f"{fy}-09-30"}], "psc_codes": cand}
    j = call("POST", "/api/v2/search/spending_by_category/psc/", {"filters": F, "limit": 100, "page": 1})
    print(f"FY{fy}")
    for x in j.get("results", []):
        print(f"    {x['code']} {x['amount']/1e6:10.0f}m {x['name']}")

HW = ["7010", "7020", "7021", "7022", "7025", "7035", "7042", "7050", "7B20", "7B21", "7B22", "7C20", "7C21", "7E20", "7E21", "7G20", "7G21", "7G22", "7K20"]
Fh = {"award_type_codes": CONTRACTS, "time_period": [{"start_date": "2016-10-01", "end_date": "2026-10-31"}], "psc_codes": HW}
j = call("POST", "/api/v2/search/spending_over_time/", {"group": "fiscal_year", "filters": Fh})
print("fy totals bn", [(r["time_period"]["fiscal_year"], round(r["aggregated_amount"] / 1e9, 2)) for r in j["results"]])
j = call("POST", "/api/v2/search/spending_over_time/", {"group": "month", "filters": Fh})
print("last months", [(r["time_period"]["fiscal_year"], r["time_period"]["month"], round(r["aggregated_amount"] / 1e6)) for r in j["results"]][-16:])
for fy in range(2017, 2027):
    j = call("POST", "/api/v2/download/count/", {"filters": dict(Fh, time_period=[{"start_date": f"{fy-1}-10-01", "end_date": f"{fy}-09-30"}])})
    print("count", fy, j.get("calculated_transaction_count"))
j = call("POST", "/api/v2/search/spending_by_category/recipient/", {"filters": dict(Fh, time_period=[{"start_date": "2024-10-01", "end_date": "2026-09-30"}]), "limit": 25, "page": 1})
for x in j["results"]:
    print(f"   {x['amount']/1e6:9.0f}m {x['name']} {x['uei']}")
j = call("POST", "/api/v2/search/spending_by_category/awarding_agency/", {"filters": dict(Fh, time_period=[{"start_date": "2024-10-01", "end_date": "2026-09-30"}]), "limit": 12, "page": 1})
for x in j["results"]:
    print(f"   {x['amount']/1e6:9.0f}m {x['name']}")

# timing of a one-fiscal-year download
t = time.time()
d = call("POST", "/api/v2/download/transactions/", {"filters": dict(Fh, time_period=[{"start_date": "2024-10-01", "end_date": "2025-09-30"}]), "columns": ["contract_transaction_unique_key", "action_date", "federal_action_obligation", "last_modified_date"], "file_format": "csv"})
s = {}
for i in range(60):
    time.sleep(10)
    s = call("GET", f"/api/v2/download/status?file_name={d['file_name']}")
    if s.get("status") in ("finished", "failed"):
        break
print("status", s, round(time.time() - t))
if s.get("status") == "finished":
    import csv, io, zipfile
    from collections import defaultdict
    r = requests.get(s["file_url"], headers=H, timeout=300)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    by = defaultdict(float)
    n = 0
    for name in z.namelist():
        print("  member", name, z.getinfo(name).file_size)
        if "PrimeTransactions" in name:
            for row in csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8")):
                by[row["action_date"][:7]] += float(row["federal_action_obligation"] or 0)
                n += 1
    print("  rows", n, {k: round(v) for k, v in sorted(by.items())})
    j = call("POST", "/api/v2/search/spending_over_time/", {"group": "month", "filters": dict(Fh, time_period=[{"start_date": "2024-10-01", "end_date": "2025-09-30"}])})
    print("  api", [(r["time_period"]["fiscal_year"], r["time_period"]["month"], round(r["aggregated_amount"])) for r in j["results"]])
