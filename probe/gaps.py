import io, time, zipfile
from collections import Counter
import pandas as pd, requests
from pdi.procurement import collect, config
from pdi.common.fetch import USER_AGENT

F = ["Award ID", "Mod", "Transaction Amount", "Action Date", "internal_id", "Awarding Agency", "Award Type", "generated_internal_id", "product_or_service_code"]
MONTHS = ["2026-01", "2025-01", "2022-02", "2023-08", "2022-06", "2019-11", "2023-05", "2017-03"]


def listed(p):
    end = (p + pd.offsets.MonthEnd(0)).date()
    rows, page = [], 1
    while True:
        for a in range(5):
            try:
                ans = collect.call("POST", config.BY_TRANSACTION, {"filters": collect.filters(p.date(), end), "limit": 100, "sort": "Transaction Amount", "order": "desc", "fields": F, "page": page})
                break
            except Exception as e:
                print("retry", e); time.sleep(20)
        rows += ans.get("results", [])
        if not ans.get("page_metadata", {}).get("hasNext"):
            break
        page += 1
    return pd.DataFrame(rows)


def download(p):
    end = (p + pd.offsets.MonthEnd(0)).date()
    cols = config.COLUMNS + ["award_id_piid", "action_type"]
    job = collect.call("POST", config.DOWNLOAD, {"filters": collect.filters(p.date(), end), "columns": cols, "file_format": "csv"})
    for _ in range(100):
        s = collect.call("GET", f"{config.DOWNLOAD_STATUS}?file_name={job['file_name']}")
        if s.get("status") in ("finished", "failed"): break
        time.sleep(10)
    z = zipfile.ZipFile(io.BytesIO(requests.get(s["file_url"], headers={"User-Agent": USER_AGENT}, timeout=600).content))
    return pd.concat([pd.read_csv(z.open(n), dtype=str, keep_default_na=False) for n in z.namelist() if "PrimeTransactions" in n])


pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
summary = []
for m in MONTHS:
    p = pd.Timestamp(m + "-01")
    s = listed(p); d = download(p)
    d["amt"] = d["federal_action_obligation"].astype(float).round(2)
    s["amt"] = s["Transaction Amount"].astype(float).round(2)
    dkey = Counter(zip(d["award_id_piid"], d["modification_number"], d["amt"], d["action_date"]))
    skey = Counter(zip(s["Award ID"], s["Mod"].astype(str), s["amt"], s["Action Date"]))
    extra_s = skey - dkey  # rows in the search beyond what the download has
    extra_d = dkey - skey
    # Of the search's extra rows, how many are copies of a row the download also has (same key)?
    copies = {k: v for k, v in extra_s.items() if k in dkey}
    absent = {k: v for k, v in extra_s.items() if k not in dkey}
    row = dict(month=m, search=s["amt"].sum(), download=d["amt"].sum(), s_rows=len(s), d_rows=len(d),
               search_extra_copies=sum(copies.values()), copies_sum=sum(k[2] * v for k, v in copies.items()),
               search_only_rows=sum(absent.values()), search_only_sum=sum(k[2] * v for k, v in absent.items()),
               download_only_rows=sum(extra_d.values()), download_only_sum=sum(k[2] * v for k, v in extra_d.items()),
               internal_id_unique=s["internal_id"].nunique(), gen_id_unique=s["generated_internal_id"].nunique())
    row["search_minus_copies_gap"] = abs(row["search"] - row["copies_sum"] - row["download"]) / abs(row["download"])
    summary.append(row)
    print(row, flush=True)
    print("  search-only examples:", sorted(absent.items(), key=lambda kv: -abs(kv[0][2]))[:6])
    print("  download-only examples:", sorted(extra_d.items(), key=lambda kv: -abs(kv[0][2]))[:6])
    if absent:
        a = s[[k in absent for k in zip(s["Award ID"], s["Mod"].astype(str), s["amt"], s["Action Date"])]]
        print("  search-only psc", Counter(a["product_or_service_code"]).most_common(6), "agency", Counter(a["Awarding Agency"]).most_common(3))
        # Is the award in the download at all (other mods)?
        print("  search-only awards present in download:", a["Award ID"].isin(set(d["award_id_piid"])).mean())
    if extra_d:
        dd = d[[k in extra_d for k in zip(d["award_id_piid"], d["modification_number"], d["amt"], d["action_date"])]]
        print(dd.sort_values("amt", key=abs, ascending=False)[["award_id_piid", "modification_number", "amt", "action_date", "product_or_service_code", "action_type", "awarding_agency_name"]].head(8).to_string())
o = pd.DataFrame(summary)
print(o.round(4).to_string())
