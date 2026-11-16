"""The Demand Radar experiment, fixed on 16 November 2026 before any figure was fetched.

Only the names below were checked beforehand: that each Census category is published in
2015 and in 2026, and that each article has monthly pageviews from July 2015. Changing any of
this after seeing a result makes it a new test, with its own row in the failures log.
"""

from __future__ import annotations

from pdi.common.paths import STORE

MODULE = "radar"
STORE_DIR = STORE / "radar"
PAGEVIEWS = STORE_DIR / "pageviews.parquet"  # article, period, views
SALES = STORE_DIR / "retail_sales.parquet"  # category, period, sales_musd
RESULT = STORE_DIR / "lead_test.json"

CENSUS_API = "https://api.census.gov/data/timeseries/eits/marts"
CENSUS_KEY_ENV = "CENSUS_API_KEY"
SALES_DATA_TYPE = "SM"  # sales, monthly, millions of dollars
SALES_ADJUSTED = "no"  # not seasonally adjusted; both series are compared on a year earlier

PAGEVIEWS_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
    "all-access/user/{article}/monthly/{start}/{end}"
)
PAGEVIEWS_START = "2015070100"  # the first month the API separates people from bots

# Census kind of business -> (name, English Wikipedia articles)
PAIRS = {
    "443": (
        "Electronics and appliance stores",
        ["Laptop", "Television", "Smartphone", "Video_game_console", "Headphones",
         "Refrigerator", "Washing_machine"],
    ),
    "442": (
        "Furniture and home furnishings stores",
        ["Furniture", "Couch", "Mattress", "Bed", "Table_(furniture)"],
    ),
    "448": (
        "Clothing and clothing accessories stores",
        ["Clothing", "Jeans", "Dress", "Sneakers", "Coat_(clothing)"],
    ),
    "451": (
        "Sporting goods, hobby, musical instrument and book stores",
        ["Bicycle", "Golf", "Camping", "Fishing_tackle", "Guitar", "Book", "Board_game",
         "Jigsaw_puzzle", "Treadmill"],
    ),
}  # fmt: skip

TRAIN_END = "2021-12"
MIN_TRAIN_MONTHS = 48
MIN_TEST_MONTHS = 24
SE_MULTIPLE = 2.0
