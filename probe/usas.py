import json
import time

import requests

B = "https://api.usaspending.gov"
H = {"User-Agent": "public-data-intelligence (+https://github.com/liumOazed/public-data-intelligence)"}


def call(label, method, path, body=None, show=1500, quiet=False):
    t = time.time()
    r = requests.request(method, B + path, json=body, headers=H, timeout=180)
    if not quiet:
        print(f"== {label}: {r.status_code} {len(r.content):,} bytes {time.time()-t:.1f}s")
    try:
        j = r.json()
    except Exception:
        print("   ", r.text[:300])
        return None
    if not quiet:
        print("   ", json.dumps(j)[:show])
    return j


top = call("Product", "GET", "/api/v2/references/filter_tree/psc/Product/", quiet=True)
ids = [x["id"] for x in top["results"]]
print("product groups", ids)
codes = {}
for g in [i for i in ids if i.startswith("7")]:
    j = call(g, "GET", f"/api/v2/references/filter_tree/psc/Product/{g}/", quiet=True)
    for x in j["results"]:
        codes[x["id"]] = x["description"]
        print("  ", g, x["id"], x["description"])

CONTRACTS = ["A", "B", "C", "D"]
allc = [c for c in codes if c[:2] in ("70", "7A", "7B", "7C", "7D", "7E", "7F", "7G", "7H", "7J", "7K")]
F = {"award_type_codes": CONTRACTS, "time_period": [{"start_date": "2018-10-01", "end_date": "2026-09-30"}], "psc_codes": allc}
j = call("by psc all IT product codes FY19-26", "POST", "/api/v2/search/spending_by_category/psc/", {"filters": F, "limit": 100, "page": 1}, quiet=True)
for x in j["results"]:
    print(f"   {x['code']} {x['amount']/1e9:8.2f}bn {x['name']}")
for fy in (2019, 2022, 2025, 2026):
    Fy = dict(F, time_period=[{"start_date": f"{fy-1}-10-01", "end_date": f"{fy}-09-30"}])
    j = call(f"by psc FY{fy}", "POST", "/api/v2/search/spending_by_category/psc/", {"filters": Fy, "limit": 100, "page": 1}, quiet=True)
    print(f"FY{fy}", [(x["code"], round(x["amount"] / 1e6)) for x in j["results"]])

HW = [c for c in allc if c in ("7010", "7020", "7021", "7022", "7025", "7035", "7042", "7050", "7B20", "7B21", "7B22", "7C20", "7E20", "7E21", "7G20", "7G21", "7G22", "7J20", "7K20")]
print("HW candidate", HW)
Fh = dict(F, psc_codes=HW)
j = call("over time HW", "POST", "/api/v2/search/spending_over_time/", {"group": "month", "filters": Fh}, quiet=True)
print("months", len(j["results"]), [(r["time_period"]["fiscal_year"], r["time_period"]["month"], round(r["aggregated_amount"] / 1e6)) for r in j["results"]][-16:])
print("fy totals", {fy: round(sum(r["aggregated_amount"] for r in j["results"] if r["time_period"]["fiscal_year"] == fy) / 1e9, 2) for fy in sorted({r["time_period"]["fiscal_year"] for r in j["results"]})})
call("download count HW FY19-26", "POST", "/api/v2/download/count/", {"filters": Fh}, 300)
call("top recipients HW", "POST", "/api/v2/search/spending_by_category/recipient/", {"filters": Fh, "limit": 15, "page": 1}, 3000)
cols = ["contract_transaction_unique_key", "contract_award_unique_key", "award_id_piid", "modification_number", "federal_action_obligation", "total_dollars_obligated", "total_outlayed_amount_for_overall_award", "base_and_all_options_value", "potential_total_value_of_award", "action_date", "awarding_agency_name", "recipient_uei", "recipient_name", "recipient_parent_uei", "recipient_parent_name", "product_or_service_code", "extent_competed", "number_of_offers_received", "contracting_officers_determination_of_business_size", "action_type", "award_type", "last_modified_date", "usaspending_permalink"]
d = call("download with columns", "POST", "/api/v2/download/transactions/", {"filters": dict(Fh, time_period=[{"start_date": "2025-10-01", "end_date": "2026-09-30"}]), "columns": cols, "file_format": "csv"}, 500)
if d and d.get("file_name"):
    s = None
    for i in range(30):
        time.sleep(10)
        s = call("status", "GET", f"/api/v2/download/status?file_name={d['file_name']}", quiet=True)
        if s and s.get("status") in ("finished", "failed"):
            break
    print("status", s)
    if s and s.get("status") == "finished":
        import csv, io, zipfile
        r = requests.get(s["file_url"], headers=H, timeout=300)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        for n in z.namelist():
            print("  member", n, z.getinfo(n).file_size)
            if "PrimeTransactions" in n:
                rows = list(csv.reader(io.TextIOWrapper(z.open(n), encoding="utf-8")))
                hd = rows[0]
                print("  rows", len(rows) - 1, "header", hd)
                ob = sum(float(r[hd.index("federal_action_obligation")] or 0) for r in rows[1:])
                print("  sum obligations FY26", ob)
                from collections import Counter
                for col in ("extent_competed", "action_type", "award_type", "contracting_officers_determination_of_business_size"):
                    if col in hd:
                        print("  ", col, Counter(r[hd.index(col)] for r in rows[1:]).most_common(12))
                if "number_of_offers_received" in hd:
                    print("  offers", Counter(r[hd.index("number_of_offers_received")] for r in rows[1:]).most_common(8))
                names = [r[hd.index("recipient_name")] for r in rows[1:]]
                print("  names with comma and no corp suffix (possible individuals)", [n for n in set(names) if "," in n and not any(k in n.upper() for k in ("INC", "LLC", "CORP", "LTD", "CO", "L.P", "LP", "PLC", "GROUP"))][:20])
                for row in rows[1:4]:
                    print("   ", row)
j = call("over time HW FY26 check", "POST", "/api/v2/search/spending_over_time/", {"group": "fiscal_year", "filters": dict(Fh, time_period=[{"start_date": "2025-10-01", "end_date": "2026-09-30"}])}, 600)
