import io, time, zipfile, csv, json
from collections import Counter
import pandas as pd, requests
from pdi.procurement import collect, config
from pdi.common.fetch import USER_AGENT

check = pd.read_parquet(config.CHECK)
bad = check[~check["within_limit"]].copy()
print("failing months", len(bad))
F = ["Award ID", "Mod", "Transaction Amount", "Action Date", "internal_id", "Awarding Agency", "Award Type", "generated_internal_id"]

def listed(period):
    end = (period + pd.offsets.MonthEnd(0)).date()
    rows, page = [], 1
    while True:
        a = collect.call("POST", config.BY_TRANSACTION, {"filters": collect.filters(period.date(), end), "limit": 100, "sort": "Transaction Amount", "order": "desc", "fields": F, "page": page})
        rows += a.get("results", [])
        if not a.get("page_metadata", {}).get("hasNext"):
            break
        page += 1
    return pd.DataFrame(rows)

out = []
keep = {}
for _, r in bad.iterrows():
    df = listed(r["period"])
    rep = df[df.duplicated("internal_id", keep="first")]
    once = df.drop_duplicates("internal_id")
    row = dict(period=r["period"].strftime("%Y-%m"), ours=r["obligations"], published=r["published"], listed=df["Transaction Amount"].sum(), rows=len(df), repeats=len(rep), repeat_sum=rep["Transaction Amount"].sum(), once=once["Transaction Amount"].sum())
    row["gap_before"] = abs(row["ours"] - row["published"]) / abs(row["published"])
    row["gap_after"] = abs(row["ours"] - row["once"]) / abs(row["once"])
    row["rep_agency"] = dict(Counter(rep["Awarding Agency"]).most_common(3))
    row["max_copies"] = int(df["internal_id"].value_counts().max())
    out.append(row); keep[row["period"]] = df
    print(row, flush=True)
o = pd.DataFrame(out)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(o[["period", "ours", "published", "once", "repeats", "gap_before", "gap_after"]].round(4).to_string())
print("within 1% after counting once:", int((o["gap_after"] <= 0.01).sum()), "of", len(o))
print("repeats total", int(o["repeats"].sum()), "of rows", int(o["rows"].sum()), "sum", o["repeat_sum"].sum())

# The months still apart after counting once: download and compare row by row.
still = o[o["gap_after"] > 0.01].sort_values("gap_after", ascending=False).head(4)
for _, r in still.iterrows():
    p = pd.Timestamp(r["period"] + "-01"); end = (p + pd.offsets.MonthEnd(0)).date()
    job = collect.call("POST", config.DOWNLOAD, {"filters": collect.filters(p.date(), end), "columns": config.COLUMNS + ["award_id_piid", "action_type", "last_modified_date", "initial_report_date"], "file_format": "csv"})
    for _ in range(80):
        s = collect.call("GET", f"{config.DOWNLOAD_STATUS}?file_name={job['file_name']}")
        if s.get("status") in ("finished", "failed"): break
        time.sleep(15)
    z = zipfile.ZipFile(io.BytesIO(requests.get(s["file_url"], headers={"User-Agent": USER_AGENT}, timeout=600).content))
    d = pd.concat([pd.read_csv(z.open(n), dtype=str, keep_default_na=False) for n in z.namelist() if "PrimeTransactions" in n])
    d["amt"] = d["federal_action_obligation"].astype(float).round(2)
    s_df = keep[r["period"]].drop_duplicates("internal_id")
    dk = Counter(zip(d["award_id_piid"], d["modification_number"], d["amt"]))
    sk = Counter(zip(s_df["Award ID"], s_df["Mod"], s_df["Transaction Amount"].round(2)))
    od, os_ = dk - sk, sk - dk
    print("=====", r["period"], "download rows", len(d), "unique keys", d["contract_transaction_unique_key"].nunique(), "search unique", len(s_df))
    print("  only in download:", sum(od.values()), "sum", sum(k[2] * v for k, v in od.items()))
    print("  only in search:", sum(os_.values()), "sum", sum(k[2] * v for k, v in os_.items()))
    dd = d[[k in od for k in zip(d["award_id_piid"], d["modification_number"], d["amt"])]]
    print(dd.sort_values("amt", key=abs, ascending=False)[["award_id_piid", "modification_number", "amt", "action_date", "awarding_agency_name", "action_type", "last_modified_date", "initial_report_date", "product_or_service_code"]].head(15).to_string())
    print("  download dup piid/mod/amt groups:", sum(1 for v in dk.values() if v > 1), "extra rows", sum(v - 1 for v in dk.values() if v > 1), "extra sum", sum(k[2] * (v - 1) for k, v in dk.items() if v > 1))
    ss = s_df[[k in os_ for k in zip(s_df["Award ID"], s_df["Mod"], s_df["Transaction Amount"].round(2))]]
    print(ss.sort_values("Transaction Amount", key=abs, ascending=False).head(10).to_string())
