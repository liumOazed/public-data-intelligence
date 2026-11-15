import csv
import io
import time
import zipfile
from collections import Counter

import requests

B = "https://api.usaspending.gov"
H = {"User-Agent": "public-data-intelligence (+https://github.com/liumOazed/public-data-intelligence)"}


def call(method, path, body=None):
    for a in range(3):
        r = requests.request(method, B + path, json=body, headers=H, timeout=180)
        if r.status_code == 200:
            return r.json()
        print("   ", r.status_code, r.text[:300])
        time.sleep(3)
    return {}


HW = ["7010", "7020", "7021", "7022", "7025", "7035", "7042", "7050", "7B20", "7B21", "7B22", "7C20", "7C21", "7E20", "7E21", "7G20", "7G21", "7G22", "7K20"]
F = {"award_type_codes": ["A", "B", "C", "D"], "time_period": [{"start_date": "2025-06-01", "end_date": "2025-06-30"}], "psc_codes": HW}
print("count", call("POST", "/api/v2/download/count/", {"filters": F}).get("calculated_transaction_count"))
j = call("POST", "/api/v2/search/spending_over_time/", {"group": "month", "filters": F})
print("api", [(r["time_period"], r["aggregated_amount"]) for r in j["results"]])

d = call("POST", "/api/v2/download/transactions/", {"filters": F, "columns": [], "file_format": "csv"})
print("download request", {k: v for k, v in d.get("download_request", {}).items() if k != "columns"})

fields = ["Award ID", "Mod", "Recipient Name", "Action Date", "Transaction Amount", "Awarding Agency", "Award Type", "product_or_service_code", "internal_id", "generated_internal_id"]
tx = []
page = 1
while True:
    j = call("POST", "/api/v2/search/spending_by_transaction/", {"filters": F, "fields": fields, "limit": 100, "page": page, "sort": "Transaction Amount", "order": "desc"})
    tx += j.get("results", [])
    if not j.get("page_metadata", {}).get("hasNext"):
        break
    page += 1
print("search rows", len(tx), "sum", sum(x["Transaction Amount"] or 0 for x in tx), "pages", page)
print("search sample", tx[:2])

s = {}
for i in range(60):
    time.sleep(10)
    s = call("GET", f"/api/v2/download/status?file_name={d['file_name']}")
    if s.get("status") in ("finished", "failed"):
        break
print("status", s)
r = requests.get(s["file_url"], headers=H, timeout=300)
z = zipfile.ZipFile(io.BytesIO(r.content))
rows = []
for name in z.namelist():
    if "PrimeTransactions" in name:
        rows += list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8")))
print("download rows", len(rows), "sum", sum(float(x["federal_action_obligation"] or 0) for x in rows))
print("download date range", min(x["action_date"] for x in rows), max(x["action_date"] for x in rows))
print("download award types", Counter(x["award_type"] for x in rows).most_common())
print("search award types", Counter(x["Award Type"] for x in tx).most_common())
print("download psc", Counter(x["product_or_service_code"] for x in rows).most_common(30))
print("search psc", Counter(str(x["product_or_service_code"]) for x in tx).most_common(30))
dk = Counter((x["award_id_piid"], x["modification_number"], round(float(x["federal_action_obligation"] or 0), 2)) for x in rows)
sk = Counter((x["Award ID"], x["Mod"], round(x["Transaction Amount"] or 0, 2)) for x in tx)
only_s = sk - dk
only_d = dk - sk
print("only in search", sum(only_s.values()), "sum", sum(k[2] * v for k, v in only_s.items()))
print("only in download", sum(only_d.values()), "sum", sum(k[2] * v for k, v in only_d.items()))
miss = [x for x in tx if (x["Award ID"], x["Mod"], round(x["Transaction Amount"] or 0, 2)) in only_s]
for x in sorted(miss, key=lambda x: -abs(x["Transaction Amount"] or 0))[:25]:
    print("   S", x)
print("missing by agency", Counter(x["Awarding Agency"] for x in miss).most_common(8))
print("missing by psc", Counter(str(x["product_or_service_code"]) for x in miss).most_common(8))
print("missing by type", Counter(x["Award Type"] for x in miss).most_common(8))
for k, v in list(only_d.items())[:10]:
    print("   D", k, v)
if rows:
    print("download columns with key/date", [c for c in rows[0] if "date" in c or "key" in c or "fiscal" in c])
