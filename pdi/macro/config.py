"""Macro Monitor settings. Every threshold and date that could be tuned after seeing results is
fixed here first, and the README records when it was written.

The catalogue below is the whole module: one row per series, with how its monthly move is
measured. A series is added here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from pdi.common.paths import RAW, STORE

MODULE = "macro"
RAW_DIR = RAW / "macro"
STORE_DIR = STORE / "macro"

OBSERVATIONS = STORE_DIR / "observations.parquet"
# Every value as it was when this pipeline first saw it. Macro figures get revised, and a
# backtest on revised figures uses information nobody had at the time. Rows are only ever
# appended: one row per series, month and distinct value, with the date it was first seen.
VINTAGES = STORE_DIR / "vintages.parquet"
FLAGS = STORE_DIR / "flags.parquet"
AUTOML_SUMMARY = STORE_DIR / "automl_summary.json"
AUTOML_LEADERBOARD = STORE_DIR / "automl_leaderboard.parquet"
AUTOML_HOLDOUT = STORE_DIR / "automl_holdout.parquet"
AUTOML_FORECASTS = STORE_DIR / "automl_forecasts.parquet"

COUNTRIES = {
    "US": "United States",
    "ID": "Indonesia",
    "MY": "Malaysia",
    "PH": "Philippines",
    "SG": "Singapore",
    "TH": "Thailand",
    "VN": "Vietnam",
}
SEA = ("ID", "MY", "PH", "SG", "TH", "VN")


@dataclass(frozen=True)
class Series:
    id: str  # our id: <country>.<indicator>
    country: str
    indicator: str  # shared across countries when the definition is the same
    name: str
    units: str
    source: str  # registry key
    source_code: str  # the source's own id for the series
    # How the monthly move is measured. "log": 100 x change in log level (about a percent
    # change). "diff": change in the value itself (percentage points for rates).
    transform: str
    # "sa": seasonally adjusted by the publisher, or not seasonal (rates, prices of money).
    # "nsa": seasonal; the move is compared with the same calendar month in earlier years.
    seasonal: str
    # Which direction is "up" in plain words, for the flag sentence.
    up_means: str = "rose"
    down_means: str = "fell"
    daily: bool = False  # published daily or weekly; averaged to months


def _fred(code, indicator, name, units, transform="log", seasonal="sa", **kw) -> Series:
    return Series(f"US.{indicator}", "US", indicator, name, units, "fred", code, transform,
                  seasonal, **kw)  # fmt: skip


# --- United States: about 25 monthly series from FRED -----------------------------------------
# Only series whose original publisher is a US federal agency (BLS, BEA, Census, the Federal
# Reserve Board, the Labor Department), which are public domain. FRED also carries series that
# third parties own under copyright (University of Michigan sentiment, S&P/Case-Shiller, ICE
# BofA, Freddie Mac mortgage rates); those are left out because republishing them needs the
# owner's permission. With a FRED API key the collector also reads each series' notes and stops
# if one says "Copyright".
US_SERIES = [
    _fred("CPIAUCSL", "cpi", "Consumer prices (CPI, all items)", "index 1982-84=100"),
    _fred(
        "CPILFESL",
        "core_cpi",
        "Core consumer prices (CPI less food and energy)",
        "index 1982-84=100",
    ),
    _fred("PCEPI", "pce_prices", "PCE price index", "index 2017=100"),
    _fred("PCEPILFE", "core_pce_prices", "Core PCE price index", "index 2017=100"),
    _fred("PPIFIS", "ppi", "Producer prices, final demand", "index Nov 2009=100"),
    _fred("PAYEMS", "payrolls", "Nonfarm payroll employment", "thousands of jobs"),
    _fred(
        "UNRATE",
        "unemployment",
        "Unemployment rate",
        "percent",
        "diff",
        up_means="rose (worse)",
        down_means="fell (better)",
    ),
    _fred("CIVPART", "participation", "Labour force participation rate", "percent", "diff"),
    _fred(
        "CES0500000003", "hourly_earnings", "Average hourly earnings, private", "dollars an hour"
    ),
    _fred("JTSJOL", "job_openings", "Job openings (JOLTS)", "thousands"),
    _fred("JTSQUR", "quits_rate", "Quits rate (JOLTS)", "percent", "diff"),
    _fred("RSAFS", "retail_sales", "Retail and food services sales", "millions of dollars"),
    _fred(
        "PCE",
        "consumer_spending",
        "Personal consumption expenditures",
        "billions of dollars, annual rate",
    ),
    _fred(
        "DSPIC96",
        "real_income",
        "Real disposable personal income",
        "billions of 2017 dollars, annual rate",
    ),
    _fred("INDPRO", "industrial_output", "Industrial production", "index 2017=100"),
    _fred("TCU", "capacity_use", "Capacity utilisation, total industry", "percent", "diff"),
    _fred("DGORDER", "durable_orders", "New orders for durable goods", "millions of dollars"),
    _fred("HOUST", "housing_starts", "Housing starts", "thousands of units, annual rate"),
    _fred("PERMIT", "building_permits", "Building permits", "thousands of units, annual rate"),
    _fred(
        "TTLCONS",
        "construction_spending",
        "Construction spending",
        "millions of dollars, annual rate",
    ),
    _fred("M2SL", "money_supply", "M2 money stock", "billions of dollars"),
    _fred("FEDFUNDS", "policy_rate", "Federal funds effective rate", "percent", "diff"),
    _fred("GS2", "yield_2y", "2-year Treasury yield", "percent", "diff"),
    _fred("GS10", "yield_10y", "10-year Treasury yield", "percent", "diff"),
    _fred(
        "DTWEXBGS",
        "dollar_broad",
        "Broad trade-weighted dollar index",
        "index Jan 2006=100",
        daily=True,
        up_means="rose (stronger dollar)",
        down_means="fell (weaker dollar)",
    ),
]

INDEED_SERIES = [
    Series("US.job_postings", "US", "job_postings", "Indeed job postings index",
           "index 1 Feb 2020=100", "indeed_postings", "aggregate_job_postings_US",
           "log", "sa", daily=True),
]  # fmt: skip

# --- Southeast Asia: the same definitions for every country, from the BIS ---------------------
# The BIS compiles these from each central bank and statistics office into one definition, so
# a move in one country can be set against the same measure in its neighbours. Vietnam has
# only its exchange rate in these BIS tables, and Singapore has no policy rate (the MAS steers
# the exchange rate instead).
BIS_INDICATORS = {
    # indicator: (dataflow, key template, name, units, transform, seasonal, up, down)
    "cpi": ("WS_LONG_CPI", "M.{c}.628", "Consumer prices (CPI)", "index 2010=100", "log", "nsa",
            "rose", "fell"),
    "policy_rate": ("WS_CBPOL", "M.{c}", "Central bank policy rate", "percent", "diff", "sa",
                    "rose", "fell"),
    "fx_usd": ("WS_XRU", "M.{c}..", "Exchange rate, local currency per US dollar",
               "local currency per USD", "log", "sa",
               "rose (local currency weaker)", "fell (local currency stronger)"),
    "reer": ("WS_EER", "M.R.B.{c}", "Real effective exchange rate, broad basket",
             "index 2020=100", "log", "sa",
             "rose (currency dearer)", "fell (currency cheaper)"),
}  # fmt: skip

# --- Malaysia in more depth, from OpenDOSM -----------------------------------------------------
# Malaysia's statistics office runs the best open statistics API in the region.
DOSM_SERIES = [
    # (dataset id, filter, value column, indicator, name, units, transform, seasonal)
    ("ipi", {"series": "abs"}, "index_sa", "industrial_output", "Industrial production",
     "index 2015=100", "log", "sa"),
    ("trade_headline", {"series": "abs"}, "exports", "exports", "Exports of goods",
     "ringgit", "log", "nsa"),
    ("trade_headline", {"series": "abs"}, "imports", "imports", "Imports of goods",
     "ringgit", "log", "nsa"),
    ("lfs_month_sa", {}, "u_rate", "unemployment", "Unemployment rate", "percent", "diff", "sa"),
    ("ppi", {"series": "abs"}, "index", "ppi", "Producer prices", "index 2010=100", "log",
     "nsa"),
]  # fmt: skip


# --- Southeast Asia from the IMF ----------------------------------------------------------------
# Trade and reserves on one definition for all six economies, compiled by the IMF from each
# country's authorities, so these too can be set against the neighbours. Vietnam's consumer
# prices come from here as well, because the BIS table has no Vietnam. Checked on GitHub's
# runners on 15 November 2026: every series below answered, with the latest month between
# June and October 2026. Industrial production was looked at and left out: the IMF's table
# stops in 2025 or earlier for every one of the six.
ISO3 = {"ID": "IDN", "MY": "MYS", "PH": "PHL", "SG": "SGP", "TH": "THA", "VN": "VNM"}
IMF_INDICATORS = {
    # indicator: (dataflow, key template, countries, name, units, transform, seasonal, up, down)
    "exports_usd": ("ITG", "{c}.XG.FOB_USD.M", SEA, "Exports of goods, free on board",
                    "US dollars", "log", "nsa", "rose", "fell"),
    "imports_usd": ("ITG", "{c}.MG.CIF_USD.M", SEA, "Imports of goods, cost insurance freight",
                    "US dollars", "log", "nsa", "rose", "fell"),
    # Excluding gold, so a move in the gold price is not read as a move in reserves.
    "reserves": ("IL", "{c}.RXF11_REVS.USD.M", SEA, "Official reserves excluding gold",
                 "US dollars", "log", "sa", "rose", "fell"),
    "cpi": ("CPI", "{c}.CPI._T.IX.M", ("VN",), "Consumer prices (CPI)", "index", "log", "nsa",
            "rose", "fell"),
}  # fmt: skip


def imf_series() -> list[Series]:
    out = []
    for ind, (flow, _key, countries, name, units, tr, seas, up, down) in IMF_INDICATORS.items():
        for c in countries:
            out.append(Series(f"{c}.{ind}", c, ind, name, units, "imf", flow, tr, seas, up, down))
    return out


# Checked on the first live run: the BIS publishes no such series. Vietnam's consumer prices
# come from the IMF instead (see IMF_INDICATORS).
BIS_NOT_PUBLISHED = {"VN.cpi", "VN.policy_rate", "VN.reer", "SG.policy_rate"}


def bis_series() -> list[Series]:
    out = []
    for ind, (flow, _key, name, units, tr, seas, up, down) in BIS_INDICATORS.items():
        for c in SEA:
            if f"{c}.{ind}" in BIS_NOT_PUBLISHED:
                continue
            out.append(Series(f"{c}.{ind}", c, ind, name, units, "bis", flow, tr, seas, up, down))
    return out


def dosm_series() -> list[Series]:
    return [
        Series(f"MY.{ind}", "MY", ind, name, units, "opendosm", f"{ds}:{col}", tr, seas)
        for ds, _f, col, ind, name, units, tr, seas in DOSM_SERIES
    ]


def catalogue() -> dict[str, Series]:
    rows = US_SERIES + INDEED_SERIES + bis_series() + dosm_series() + imf_series()
    out = {s.id: s for s in rows}
    if len(out) != len(rows):
        raise ValueError("duplicate series id in the Macro Monitor catalogue")
    return out


# Indicators compared across Southeast Asia: only those compiled on one definition for every
# country, by the BIS or the IMF. Consumer prices mix the two (Vietnam's from the IMF, the
# others from the BIS); both are the national all-items index.
NEIGHBOUR_INDICATORS = tuple(BIS_INDICATORS) + ("exports_usd", "imports_usd", "reserves")

# --- Collection --------------------------------------------------------------------------------
FRED_KEY_ENV = "FRED_API_KEY"
FRED_API = "https://api.stlouisfed.org/fred"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BIS_API = "https://stats.bis.org/api/v1/data"
DOSM_API = "https://api.data.gov.my/opendosm"
IMF_API = "https://api.imf.org/external/sdmx/2.1/data"
# Trade for every country starts in 2006 except Singapore, whose earlier months are on another
# basis (a 3.1x jump in January 2004 stopped the unit check on the first try). One start date
# for every IMF series keeps them comparable, and still leaves 20 years of history.
IMF_HISTORY_START = "2006-01"
IMF_CSV_ACCEPT = "application/vnd.sdmx.data+csv;version=1.0.0"
INDEED_URL = (
    "https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/master/US/"
    "aggregate_job_postings_US.csv"
)
HISTORY_START = "1990-01"
# Below this many series collected, assume a source changed format and stop the run.
MIN_SERIES_COLLECTED = 60

# --- Flags, written on 15 November 2026 before any flag was computed ---------------------------
# A move is measured as the month's change (see Series.transform). Its expected value is the
# median move over the previous ten years, or for seasonal series the median move in the same
# calendar month over those years. The surprise is the difference, divided by a robust spread
# (1.4826 x median absolute deviation, which equals the standard deviation for normal data).
HISTORY_MONTHS = 120
MIN_HISTORY_MONTHS = 60
MIN_SAME_MONTH_YEARS = 5
FLAG_Z = 2.5
# Against neighbours: the country's surprise minus the median surprise of the other Southeast
# Asian countries that month, scored against that gap's own ten-year history. Needs at least
# this many other countries reporting the same month.
MIN_NEIGHBOURS = 3
# Backtest window and the rule for whether a flag mattered, fixed before the backtest ran.
BACKTEST_MONTHS = 24
# A flag "held" if over the next three months the series did not give back more than half of
# the flagged surprise; otherwise it "reversed". Fewer than three later months: "too early".
FOLLOW_MONTHS = 3
REVERSAL_SHARE = 0.5

# --- Forecast test (AutoGluon TimeSeries), written before any result was seen ------------------
# Target: the monthly move of each series, as defined above, one month ahead. Models are fitted
# on data before VALID_START, picked on one-step forecasts for the validation months, and the
# pick is scored once on one-step forecasts for the holdout months. The baseline it must beat
# is whichever of the four rule-of-thumb models did best on validation: Naive (next month's
# move equals this month's), SeasonalNaive (equals the move twelve months earlier), Average
# (equals the series' average move, which is a random walk with drift in the level) and
# SeasonalAverage (the average move for that calendar month). Beating only Naive would be too
# easy: monthly moves are close to noise, so repeating the last one is a poor forecast.
# Error is MASE-style: absolute error divided by the series' own mean absolute month-to-month
# change in the training period, so series with different units can be averaged.
VALID_START = "2023-11-01"
HOLDOUT_START = "2024-11-01"
# The pick must beat the naive baseline on the holdout by more than the spread (standard
# deviation) of its improvement across the four quarterly folds of the validation year.
N_VALID_FOLDS = 4
AUTOML_TIME_LIMIT_S = 1200
AUTOML_MODELS = {
    "Naive": {},
    "SeasonalNaive": {},
    "Average": {},
    "SeasonalAverage": {},
    "ETS": {},
    "AutoARIMA": {},
    "Theta": {},
    "NPTS": {},
    "RecursiveTabular": {},
    "DirectTabular": {},
}
NAIVE_MODELS = ("Naive", "SeasonalNaive", "Average", "SeasonalAverage")
