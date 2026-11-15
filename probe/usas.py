import json
import time

import requests

B = "https://api.usaspending.gov"
H = {"User-Agent": "public-data-intelligence (+https://github.com/liumOazed/public-data-intelligence)"}


def call(label, method, path, body=None, show=1500):
    t = time.time()
    try:
        r = requests.request(method, B + path, json=body, headers=H, timeout=120)
    except Exception as e:
        print(f"== {label}: EXC {type(e).__name__}")
        return None
    print(f"== {label}: {r.status_code} {len(r.content):,} bytes {time.time()-t:.1f}s")
    try:
        j = r.json()
    except Exception:
        print("   ", r.text[:300])
        return None
    print("   ", json.dumps(j)[:show])
    return j


# PSC tree: find IT hardware codes
call("psc tree top", "GET", "/api/v2/references/filter_tree/psc/", show=900)
call("psc tree Product", "GET", "/api/v2/references/filter_tree/psc/Product/", show=3000)
j = call("psc tree Product/70", "GET", "/api/v2/references/filter_tree/psc/Product/70/", show=3500)
call("psc tree Product/7B", "GET", "/api/v2/references/filter_tree/psc/Product/7B/", show=2500)
call("psc search hardware", "GET", "/api/v2/references/filter_tree/psc/?filter=hardware&depth=3", show=6000)

CONTRACTS = ["A", "B", "C", "D"]
F = {
    "award_type_codes": CONTRACTS,
    "time_period": [{"start_date": "2023-10-01", "end_date": "2026-09-30"}],
    "psc_codes": {"require": [["Product", "70"]]},
}
call("over time by month, PSC 70 tree", "POST", "/api/v2/search/spending_over_time/", {"group": "month", "filters": F}, 2500)
F2 = dict(F, psc_codes=["7B22", "7B21", "7E20", "7010", "7021"])
call("over time, flat codes", "POST", "/api/v2/search/spending_over_time/", {"group": "month", "filters": F2}, 1800)
call("top recipients", "POST", "/api/v2/search/spending_by_category/recipient/", {"filters": F, "limit": 10, "page": 1}, 2500)
call("by awarding agency", "POST", "/api/v2/search/spending_by_category/awarding_agency/", {"filters": F, "limit": 5, "page": 1}, 1200)
call("by psc", "POST", "/api/v2/search/spending_by_category/psc/", {"filters": F, "limit": 30, "page": 1}, 3500)
call("count", "POST", "/api/v2/search/spending_by_award_count/", {"filters": F}, 400)
call(
    "awards",
    "POST",
    "/api/v2/search/spending_by_award/",
    {
        "filters": F,
        "fields": ["Award ID", "Recipient Name", "Award Amount", "Total Outlays", "Start Date", "End Date",
                   "Awarding Agency", "Contract Award Type", "recipient_id", "Recipient UEI", "PSC", "Base Obligation Date", "Last Modified Date"],
        "limit": 3, "page": 1, "sort": "Award Amount", "order": "desc",
    },
    3000,
)
call("transactions", "POST", "/api/v2/search/spending_by_transaction/",
     {"filters": F, "fields": ["Award ID", "Recipient Name", "Action Date", "Transaction Amount", "Awarding Agency", "Mod", "Award Type", "recipient_id", "Recipient UEI", "product_or_service_code"],
      "limit": 3, "page": 1, "sort": "Transaction Amount", "order": "desc"}, 2500)
call("transaction count", "POST", "/api/v2/search/transaction_spending_summary/", {"filters": F}, 400)
d = call("download count", "POST", "/api/v2/download/count/", {"filters": F}, 400)
d = call("download transactions", "POST", "/api/v2/download/transactions/",
         {"filters": dict(F, time_period=[{"start_date": "2026-07-01", "end_date": "2026-09-30"}]), "columns": [], "file_format": "csv"}, 900)
if d and d.get("file_name"):
    for i in range(20):
        time.sleep(15)
        s = call(f"status {i}", "GET", f"/api/v2/download/status?file_name={d['file_name']}", show=500)
        if s and s.get("status") in ("finished", "failed"):
            break
    if s and s.get("status") == "finished":
        import io, zipfile, csv
        r = requests.get(s["file_url"], headers=H, timeout=300)
        print("zip", r.status_code, len(r.content))
        z = zipfile.ZipFile(io.BytesIO(r.content))
        for n in z.namelist():
            print("  member", n, z.getinfo(n).file_size)
            if n.endswith(".csv"):
                rows = list(csv.reader(io.TextIOWrapper(z.open(n), encoding="utf-8")))
                print("  rows", len(rows) - 1, "columns", len(rows[0]))
                print("  header", rows[0])
                want = ["action_date", "federal_action_obligation", "total_dollars_obligated", "base_and_exercised_options_value", "base_and_all_options_value", "current_total_value_of_award", "potential_total_value_of_award", "total_outlayed_amount_for_overall_award", "recipient_name", "recipient_name_raw", "recipient_uei", "recipient_parent_name", "recipient_parent_uei", "product_or_service_code", "extent_competed", "number_of_offers_received", "awarding_agency_name", "award_id_piid", "modification_number", "contracting_officers_determination_of_business_size", "type_of_set_aside", "action_type"]
                idx = [rows[0].index(w) for w in want if w in rows[0]]
                print("  missing", [w for w in want if w not in rows[0]])
                for row in rows[1:4]:
                    print("   ", [row[i] for i in idx])
