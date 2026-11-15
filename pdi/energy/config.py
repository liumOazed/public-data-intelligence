"""Energy Demand settings. Written on 15 November 2026, before any number was computed.

One measure only: monthly electricity demand, for the United States and the Southeast Asian
economies the source covers, from one source that puts every country on one definition. The
"unusual month" rule is the Macro Monitor's (same window, same threshold), not a new one.
"""

from __future__ import annotations

from pdi.common.paths import RAW, STORE
from pdi.macro import config as macro

MODULE = "energy"
SOURCE = "ember_monthly"
RAW_DIR = RAW / "energy"
STORE_DIR = STORE / "energy"
DEMAND = STORE_DIR / "demand.parquet"
# Every value as first seen, appended only: the source revises recent months.
VINTAGES = STORE_DIR / "vintages.parquet"
RETRIEVED = STORE_DIR / "retrieved.json"

EMBER_URL = (
    "https://files.ember-energy.org/public-downloads/generation/outputs/"
    "release_generation_monthly_global.csv"
)
# The file is about 30 MB and the publisher updates it twice a month, so it is fetched at most
# once in this many days however often the pipeline runs.
REFETCH_AFTER_DAYS = 6

# Our code -> the source's ISO3 code.
ISO3 = {"US": "USA", "ID": "IDN", "MY": "MYS", "PH": "PHL", "SG": "SGP", "TH": "THA", "VN": "VNM"}
COUNTRIES = macro.COUNTRIES
# These must be in the file or the run stops. Indonesia had no monthly demand in the file on
# 15 November 2026; it is picked up without a code change if the source adds it.
REQUIRED = ("US", "MY", "PH", "SG", "TH", "VN")

# Rows of the source file that are used. "Demand" is the measure; the other two only check it.
DEMAND_ROW = "Demand"
GENERATION_ROW = "Total generation"
NET_IMPORTS_ROW = "Net imports"
# Demand should equal generation plus net imports. More than this share of months off by more
# than the tolerance means the file's meaning changed.
IDENTITY_TOLERANCE = 0.02
IDENTITY_MAX_BAD_SHARE = 0.05

# Plausible monthly demand in TWh for any of these countries. Singapore uses about 5 a month and
# the United States about 400; a value outside the range is a unit error.
MIN_TWH, MAX_TWH = 0.5, 1000.0

# The first full year every required country has. Demand is shown as an index on this year so
# that a country using 5 TWh a month and one using 400 fit on one chart.
INDEX_BASE_YEAR = 2019

# The unusual-month rule, shared with the Macro Monitor.
FLAG_Z = macro.FLAG_Z
