"""Trade and Supply Chain settings.

Every date, threshold and test rule below was written on 15 November 2026, before any trade
figure was collected. They are not to be changed after results are seen; a change is a new
test and gets its own row in the failures log.
"""

from __future__ import annotations

from pdi.common.paths import RAW, STORE

MODULE = "trade"
SOURCE = "census_trade"
RAW_DIR = RAW / "trade"
STORE_DIR = STORE / "trade"
IMPORTS = STORE_DIR / "us_imports_hs2.parquet"  # partner, chapter, period, value
TOTALS = STORE_DIR / "us_imports_totals.parquet"  # partner, period, value (all commodities)
CHAPTERS = STORE_DIR / "chapters.json"  # chapter -> the source's short description
RETRIEVED = STORE_DIR / "retrieved.json"
LEAD_TEST = STORE_DIR / "lead_test.json"

API = "https://api.census.gov/data/timeseries/intltrade/imports/hs"
KEY_ENV = "CENSUS_API_KEY"
# General imports, customs value, in US dollars: the headline measure in the Census releases.
VALUE = "GEN_VAL_MO"
HISTORY_START = "2013-01"
# Census publishes once a month; the collector asks at most once in this many days.
REFETCH_AFTER_DAYS = 6

# Our code -> Census country code. "WLD" is every country together (the API's "-").
PARTNERS = {
    "CN": ("5700", "China"),
    "ID": ("5600", "Indonesia"),
    "MY": ("5570", "Malaysia"),
    "PH": ("5650", "Philippines"),
    "SG": ("5590", "Singapore"),
    "TH": ("5490", "Thailand"),
    "VN": ("5520", "Vietnam"),
    "WLD": ("-", "All countries"),
}
SEA = ("ID", "MY", "PH", "SG", "TH", "VN")
# Chapters 01 to 97 are goods. 98 and 99 are special provisions (returned goods, low-value
# shipments); they stay in the totals but are not shown as products.
SPECIAL_CHAPTERS = ("98", "99")

# --- Exit criterion 1: the figures match what Census publishes --------------------------------
# For every partner and month, the sum of the chapters must be within this share of the
# all-commodity total that Census publishes for that partner, asked for in a separate call.
TOTALS_TOLERANCE = 0.01

# --- Share shift -----------------------------------------------------------------------------
# A partner's share of US imports of a chapter, over twelve months (which removes the seasons).
# The shift is that share minus the share in the base year. 2017 is the last calendar year
# before the first US Section 301 tariffs on China (July 2018).
BASE_YEAR = 2017
# A chapter is listed in the "who gained where China lost" table only if the US imported at
# least this much of it from all countries in the latest twelve months. Smaller chapters are
# noisy; they stay in the full table.
MIN_CHAPTER_USD = 5e9

# --- Exit criterion 2: the lead-time test ------------------------------------------------------
# Claim under test: US-reported imports from an economy tell you about that economy's own
# reported total exports before its own figure is out. Census publishes month t about five
# weeks after it ends; the IMF series for these economies arrive one to five months later.
#
# For each economy, the target is the year-on-year growth (100 x log change on twelve months
# earlier) of its total goods exports in month t (IMF, in the Macro Monitor store).
#   Baseline: the same growth one month earlier (persistence), the best a reader has without
#             US data if the economy's own figure is one month behind.
#   Model:    a straight line on that lagged growth and on the year-on-year growth of US
#             imports from the economy in month t, fitted by least squares.
# Fitted on months up to LEAD_TRAIN_END, scored once on every later month. The claim is shown
# for an economy only if the model's mean absolute error beats the baseline's by more than
# LEAD_SE_MULTIPLE standard errors of the month-by-month gain. Otherwise it is dropped and the
# result goes in the failures log.
LEAD_TRAIN_END = "2022-12"
LEAD_MIN_TRAIN_MONTHS = 60
LEAD_MIN_TEST_MONTHS = 24
LEAD_SE_MULTIPLE = 2.0
# Reported beside the test as an observation, not as the test: the correlation between the two
# growth rates with US imports shifted by this many months either way.
LEAD_CORRELATION_SHIFTS = (-3, -2, -1, 0, 1, 2, 3)
