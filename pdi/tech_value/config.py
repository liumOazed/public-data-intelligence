"""Tech Value settings. Anything that could be tuned after seeing results is fixed here first."""

from __future__ import annotations

from pdi.common.paths import RAW, STORE

MODULE = "tech-value"
RAW_DIR = RAW / "tech_value"
STORE_DIR = STORE / "tech_value"

SPEC_TABLE = STORE_DIR / "spec_results.parquet"
GWPG_TABLE = STORE_DIR / "spec_gwpg_results.parquet"
BLENDER_TABLE = STORE_DIR / "blender_devices.parquet"
PRICES_DIR = STORE_DIR / "prices"
AUTOML_LEADERBOARD = STORE_DIR / "automl_leaderboard.parquet"
AUTOML_SUMMARY = STORE_DIR / "automl_summary.json"

# --- Benchmarks -------------------------------------------------------------------------------
# One SPEC suite, one metric. Suites and base/peak are different measurements and are never mixed.
SPEC_ENDPOINT = "https://www.spec.org/cgi-bin/osgresults"
SPEC_CONF = "rint2017"  # SPECrate 2017 Integer
SPEC_BENCHMARK = "CINT2017rate"
SPEC_METRIC = "base_score"

# SPEC Graphics & Workstation Performance Group (GWPG) official result summary pages.
GWPG_ENDPOINTS = {
    "SPECviewperf 15": "https://www.spec.org/gwpg/gpc.data/viewperf15/summary.html",
    "SPECviewperf 2020 v3": "https://www.spec.org/gwpg/gpc.data/viewperf2020v3/summary.html",
    "SPECviewperf 2020 v3 Linux": "https://www.spec.org/gwpg/gpc.data/viewperf2020v3linux/summary.html",
    "SPECviewperf 2020 v2": "https://www.spec.org/gwpg/gpc.data/viewperf2020v2/summary.html",
    "SPECviewperf 2020 v1": "https://www.spec.org/gwpg/gpc.data/viewperf2020/summary.html",
    "SPECviewperf 13": "https://www.spec.org/gwpg/gpc.data/vp13/summary.html",
    "SPECviewperf 13 Linux": "https://www.spec.org/gwpg/gpc.data/vp13linux/summary.html",
    "SPECviewperf 12.1": "https://www.spec.org/gwpg/gpc.data/vp12.1/summary.html",
    "SPECworkstation 4.0": "https://www.spec.org/gwpg/wpc.data/specworkstation4_summary.html",
    "SPECworkstation 3.1": "https://www.spec.org/gwpg/wpc.data/specworkstation31_summary.html",
    "SPECworkstation 3.0": "https://www.spec.org/gwpg/wpc.data/specworkstation3_summary.html",
    "SPECapc Maya 2024": "https://www.spec.org/gwpg/apc.data/specapc_maya2024_summary.html",
    "SPECapc Solidworks 2022": "https://www.spec.org/gwpg/apc.data/specapc_sw2022_summary.html",
    "SPECapc Creo 9": "https://www.spec.org/gwpg/apc.data/specapc_creo9_summary.html",
    "SPECapc 3ds Max 2020": "https://www.spec.org/gwpg/apc.data/specapc_3dsm2020_summary.html",
}
GWPG_MIN_RESULTS_FOR_RANK = 2

BLENDER_SNAPSHOT_URL = "https://opendata.blender.org/snapshots/opendata-latest.zip"
BLENDER_SCENES = ("monster", "junkshop", "classroom")
# A device needs this many complete runs on one Blender version before its median is shown.
BLENDER_MIN_RUNS = 5

# --- Prices -----------------------------------------------------------------------------------
BESTBUY_ENDPOINT = "https://api.bestbuy.com/v1/products"
BESTBUY_KEY_ENV = "BESTBUY_API_KEY"
# Best Buy category ids. From memory, not yet checked against the live category API: the
# collector fails loudly if a category returns products that do not look like CPUs or GPUs.
BESTBUY_CATEGORIES = {"cpu": "abcat0507010", "gpu": "abcat0507002"}
BESTBUY_FIELDS = (
    "sku,name,manufacturer,modelNumber,regularPrice,salePrice,onSale,priceUpdateDate,"
    "onlineAvailability,condition,url"
)
# Below this share of recognisable product names, assume the category id is wrong and stop.
BESTBUY_MIN_RECOGNISED = 0.2

EBAY_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_BROWSE_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
EBAY_CLIENT_ID_ENV = "EBAY_CLIENT_ID"
EBAY_CLIENT_SECRET_ENV = "EBAY_CLIENT_SECRET"
# eBay US leaf categories: 164 = CPUs/Processors, 27386 = Computer Graphics Cards.
EBAY_CATEGORIES = {"cpu": "164", "gpu": "27386"}
EBAY_RESULTS_PER_QUERY = 10
EBAY_MIN_FEEDBACK_PCT = 98.0
EBAY_MIN_FEEDBACK_SCORE = 100


# --- Retail crawler -----------------------------------------------------------------------------
# Retailers are crawled only where robots.txt allows the page and the site answers a plainly
# identified client. Nothing here tries to get past a bot wall.
CRAWLER_USER_AGENT = "pdi-research-bot/0.1 (+https://github.com/liumOazed/public-data-intelligence)"
CRAWLER_PAUSE_S = 2.0
# How many benchmarked models of each kind to look up per day, most-benchmarked first.
CRAWLER_MODELS_PER_KIND = 40
# Below these a "CPU" or "GPU" listing is an accessory, not the part; above MAX_PART_PRICE is a
# server/lot/placeholder listing outside TIDY_SCHEMA's range.
MIN_PART_PRICE = {"cpu": 40.0, "gpu": 80.0}
MAX_PART_PRICE = 20000.0
# If fewer than this many searched models come back with any listing at all, treat the
# retailer's page format as changed and stop.
CRAWLER_MIN_MODELS_FOUND = 3

# --- Model evaluation, written before any result was seen ---------------------------------------
# Processors are assigned to a period by the first hardware-availability date they appear with.
# Train on older generations, pick on the next ones, test once on the newest.
VALID_START = "2023-01-01"
HOLDOUT_START = "2024-06-01"
# Expanding-window folds inside the pre-holdout period, used to measure fold-to-fold spread.
N_TIME_FOLDS = 4
# Group folds for the out-of-fold residuals that the ranking is built on.
N_RESIDUAL_FOLDS = 5
# A processor needs this many published results before its residual is ranked.
MIN_RESULTS_FOR_RANK = 3
VINTAGE_ORIGIN = "2017-01-01"

AUTOML_TIME_LIMIT_S = 900
AUTOML_PRESET = "medium_quality"

# A listing priced under this share of the model's median listing is ignored as a likely mislabel.
MIN_SHARE_OF_MEDIAN_PRICE = 0.5

# Bright Data Web Scraper API (Amazon US and Best Buy search by keyword). Every delivered
# record is billed against a free monthly allowance of 5,000, so results per query are capped
# and variant SKUs share one base search query.
BRIGHTDATA_API = "https://api.brightdata.com/datasets/v3"
BRIGHTDATA_KEY_ENV = "BRIGHTDATA_API_KEY"
BRIGHTDATA_AMAZON_DATASET = "gd_l7q7dkf244hwjntr0"
BRIGHTDATA_RESULTS_PER_MODEL = 4
BRIGHTDATA_WAIT_S = 20 * 60
BRIGHTDATA_BESTBUY_DATASET = "gd_ltre1jqe1jfr7cccf"
# Best Buy pages are slow to collect: four searches took over 20 minutes in the first test.
BRIGHTDATA_BESTBUY_WAIT_S = 60 * 60

# A source that is collected weekly still counts on the days between its runs.
PRICE_MAX_AGE_DAYS = 8

# Fair price model: predeclared, like the benchmark model's settings.
FAIR_PRICE_MIN_LISTINGS = 2  # parts with fewer listings are not used for fitting
FAIR_PRICE_MIN_PARTS = 10  # fewer fitted parts of one kind than this: no fair price
FAIR_PRICE_GAP_PCT = 15.0  # further than this from fair price is called below or above
