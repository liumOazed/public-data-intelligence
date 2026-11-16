"""Names only: which Census MARTS categories exist, and whether candidate articles have
monthly pageviews since July 2015. Prints no figures."""
import json, os, urllib.error, urllib.request, urllib.parse

UA = {"User-Agent": "public-data-intelligence research pipeline (github.com/liumOazed/public-data-intelligence)"}

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.loads(r.read())

key = os.environ.get("CENSUS_API_KEY", "")
v = get("https://api.census.gov/data/timeseries/eits/marts/variables.json")
print("variables:", sorted(v["variables"]))
for var in ("category_code", "data_type_code"):
    try:
        d = get(f"https://api.census.gov/data/timeseries/eits/marts/variables/{var}.json")
        print(var, json.dumps(d.get("values", d), indent=0)[:4000])
    except Exception as e:
        print(var, "error", e)
for q in ({"get": "category_code,data_type_code,seasonally_adj", "time": "2015-01"},
          {"get": "category_code,data_type_code,seasonally_adj", "time": "2015-01", "for": "us:*"}):
    if key:
        q["key"] = key
    try:
        rows = get("https://api.census.gov/data/timeseries/eits/marts?" + urllib.parse.urlencode(q))
        cats = sorted({(r[0], r[1], r[2]) for r in rows[1:]})
        print("query", sorted(k for k in q if k != "key"), "ok; categories in 2015-01:", cats)
        break
    except urllib.error.HTTPError as e:
        print("query", sorted(k for k in q if k != "key"), "failed", e.code, e.read()[:300])
ARTICLES = ["Laptop", "Television", "Smartphone", "Video_game_console", "Headphones",
            "Refrigerator", "Washing_machine", "Furniture", "Couch", "Mattress", "Bed",
            "Table_(furniture)", "Clothing", "Jeans", "Dress", "Sneakers", "Coat_(clothing)",
            "Bicycle", "Golf", "Camping", "Fishing_tackle", "Guitar", "Book", "Board_game",
            "Jigsaw_puzzle", "Treadmill"]
for a in ARTICLES:
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
           f"all-access/user/{a}/monthly/2015070100/2026103100")
    try:
        n = len(get(url)["items"])
        print(a, "months:", n)
    except Exception as e:
        print(a, "error", e)
