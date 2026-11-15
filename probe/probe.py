import sys, requests
UA = {"User-Agent": "public-data-intelligence (+https://github.com/liumOazed/public-data-intelligence)"}
SEA = "ID+MY+PH+SG+TH+VN+US"
urls = [
 "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL,UNRATE",
 "https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&file_type=json",
 f"https://stats.bis.org/api/v1/data/WS_CBPOL/M.{SEA}/all?format=csv&lastNObservations=2",
 f"https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/M.{SEA}?format=csv&lastNObservations=2",
 f"https://stats.bis.org/api/v1/data/WS_LONG_CPI/M.{SEA}./all?format=csv&lastNObservations=2",
 f"https://stats.bis.org/api/v1/data/WS_XRU/M.{SEA}../all?format=csv&lastNObservations=2",
 f"https://stats.bis.org/api/v1/data/WS_EER/M.R.B.{SEA}/all?format=csv&lastNObservations=2",
 f"https://stats.bis.org/api/v1/data/WS_CBPOL/M.{SEA}/all?format=csv&startPeriod=1990-01&detail=dataonly",
 "https://api.data.gov.my/opendosm?id=cpi_headline&limit=3",
 "https://api.data.gov.my/opendosm?id=cpi_headline&limit=3&sort=-date",
 "https://api.data.gov.my/opendosm?id=ipi&limit=3&sort=-date",
 "https://api.data.gov.my/opendosm?id=trade_headline&limit=3&sort=-date",
 "https://api.data.gov.my/opendosm?id=lfs_month&limit=3&sort=-date",
 "https://api.data.gov.my/opendosm?id=lfs_month_sa&limit=3&sort=-date",
 "https://api.data.gov.my/opendosm?id=ppi&limit=3&sort=-date",
 "https://storage.dosm.gov.my/cpi/cpi_2d.parquet",
 "https://tablebuilder.singstat.gov.sg/api/table/tabledata/M212881?limit=3",
 "https://tablebuilder.singstat.gov.sg/api/table/resourceid?keyword=consumer%20price%20index&searchoption=all",
 "https://webapi.bps.go.id/v1/api/list/model/data/lang/eng/domain/0000/var/1/key/x",
]
for u in urls:
    print("=" * 100); print(u)
    try:
        r = requests.get(u, headers=UA, timeout=60)
        print(r.status_code, r.headers.get("Content-Type"), len(r.content))
        txt = r.text
        if "detail=dataonly" in u:
            lines = txt.splitlines(); print(lines[0]); 
            import collections; c = collections.Counter(l.split(",")[2] if len(l.split(","))>2 else "" for l in lines[1:]); print(c)
            print("\n".join(lines[1:3])); print("\n".join(lines[-3:]))
        else:
            print(txt[:1500])
    except Exception as e:
        print("ERR", type(e).__name__, e)
