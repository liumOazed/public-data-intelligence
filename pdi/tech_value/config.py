"""Tech Value settings. Anything that could be tuned after seeing results is fixed here first."""

from __future__ import annotations

from pdi.common.paths import RAW, STORE

MODULE = "tech-value"
RAW_DIR = RAW / "tech_value"
STORE_DIR = STORE / "tech_value"

SPEC_TABLE = STORE_DIR / "spec_results.parquet"
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

# --- Retail crawler -----------------------------------------------------------------------------
# Retailers are crawled only where robots.txt allows the page and the site answers a plainly
# identified client. Nothing here tries to get past a bot wall.
CRAWLER_USER_AGENT = "pdi-research-bot/0.1 (+https://github.com/liumOazed/public-data-intelligence)"
CRAWLER_PAUSE_S = 2.0
# How many benchmarked models of each kind to look up per day, most-benchmarked first.
CRAWLER_MODELS_PER_KIND = 40
# Below these a "CPU" or "GPU" listing is an accessory, not the part.
MIN_PART_PRICE = {"cpu": 40.0, "gpu": 80.0}
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
