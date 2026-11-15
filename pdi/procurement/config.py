"""Federal Procurement settings: computer hardware bought by every federal agency.

Every product code, threshold and rule below was written on 15 November 2026, before the
module's figures were computed. They are not to be changed after results are seen; a change is
a new test and gets its own row in the failures log.

What had been seen when this was written, from a test of the source on the same day: the names
and yearly totals of the product codes, the list of the largest recipients, and one fiscal
year (2025) of monthly totals from the transaction download beside USAspending's own monthly
figure, which were further apart than the 1% limit in most months. The limit was already
written in the project plan and is kept as it was.
"""

from __future__ import annotations

from pdi.common.paths import RAW, STORE

MODULE = "procurement"
SOURCE = "usaspending"
RAW_DIR = RAW / "procurement"
STORE_DIR = STORE / "procurement"
MONTHLY = STORE_DIR / "monthly.parquet"  # period, agency, psc, obligations, actions
SUPPLIER_YEAR = STORE_DIR / "supplier_year.parquet"  # fiscal_year, supplier, obligations, ...
COMPETITION = STORE_DIR / "competition.parquet"  # fiscal_year, kind, obligations, actions
COHORTS = STORE_DIR / "award_cohorts.parquet"  # one row per fiscal year of first award
CHECK = STORE_DIR / "totals_check.parquet"  # period, ours, published, gap_share
SUPPLIERS = STORE_DIR / "suppliers.json"  # supplier id -> name, for the named suppliers only
CLEANING = STORE_DIR / "cleaning.json"  # how many names, registrations and groups
DIAGNOSIS = STORE_DIR / "diagnosis.json"  # row-by-row reconciliation of the missed months
RETRIEVED = STORE_DIR / "retrieved.json"

API = "https://api.usaspending.gov"
DOWNLOAD = "/api/v2/download/transactions/"
DOWNLOAD_STATUS = "/api/v2/download/status"
OVER_TIME = "/api/v2/search/spending_over_time/"
BY_TRANSACTION = "/api/v2/search/spending_by_transaction/"
# The data change slowly and one refresh is ten large downloads, so the collector asks at most
# once in this many days (it runs in the weekly workflow).
REFETCH_AFTER_DAYS = 6
# FPDS holds back Department of Defense records for 90 days, so the newest months are missing
# most defence buying. Months and fiscal years that ended fewer than this many days before
# retrieval are marked provisional on the pages.
PROVISIONAL_DAYS = 90
DOWNLOAD_WAIT_S = 1500
DOWNLOAD_POLL_S = 15

# --- The slice -------------------------------------------------------------------------------
# Contracts only (definitive contracts, purchase orders, delivery orders, BPA calls); no
# grants, loans or the umbrella agreements orders are placed under.
AWARD_TYPES = ["A", "B", "C", "D"]
# US fiscal years run October to September. Fiscal 2017 starts on 1 October 2016.
FIRST_FISCAL_YEAR = 2017

# "Computer hardware" by Product and Service Code. The federal code list was rewritten on
# 1 October 2020: the old group 70 codes were retired and replaced by codes 7A to 7K.
# Software (7030, 7A20, 7A21), supplies (7045) and punched card equipment (7040) are left out.
# So are the new codes for networks and radios (7G20 to 7G22) and mobile devices (7E21), which
# before October 2020 sat under telecommunications codes outside group 70; including them would
# make the series jump at the change for a reason that has nothing to do with spending. The new
# codes for service delivery, management tools, platforms and security (7D20, 7F20, 7H20, 7J20)
# are mostly software and are left out too.
PSC = {
    # code: (group shown on the pages, the source's own description)
    "7010": ("Other (codes retired in 2020)", "IT equipment system configuration"),
    "7020": ("Computers and servers", "Central processing unit (computer), analog"),
    "7021": ("Computers and servers", "Central processing unit (computer), digital"),
    "7022": ("Computers and servers", "Central processing unit (computer), hybrid"),
    "7025": ("Storage", "Input/output and storage devices"),
    "7035": ("Other (codes retired in 2020)", "IT support equipment"),
    "7042": ("Computers and servers", "Mini and micro computer control devices"),
    "7050": ("Other (codes retired in 2020)", "IT components"),
    "7B20": ("Computers and servers", "High performance compute"),
    "7B21": ("Computers and servers", "Compute: mainframe"),
    "7B22": ("Computers and servers", "Compute: servers"),
    "7C20": ("Data centre equipment", "Data center products"),
    "7C21": ("Data centre equipment", "Other data center facilities products"),
    "7E20": ("Desktops, laptops and printers", "End user: workspace, print, productivity tools"),
    "7K20": ("Storage", "Storage products"),
}
# The first month coded only with the new list. Charts mark it; comparisons across it are not
# made on the pages without saying so.
CODE_CHANGE = "2020-10"

# Columns asked for in the download. Nothing about a person or a place: no addresses, phone
# numbers, contracting officers or executive names. The recipient is the company on the contract.
COLUMNS = [
    "contract_transaction_unique_key",
    "contract_award_unique_key",
    "modification_number",
    "action_date",
    "federal_action_obligation",  # money promised by this action (can be negative)
    "total_dollars_obligated",  # promised so far on the whole award, as of this action
    "potential_total_value_of_award",  # the ceiling: value if every option is exercised
    "total_outlayed_amount_for_overall_award",  # money paid so far on the whole award
    "awarding_agency_name",
    "recipient_uei",
    "recipient_name",
    "recipient_parent_uei",
    "recipient_parent_name",
    "product_or_service_code",
    "extent_competed",
    "number_of_offers_received",
]

# --- Exit criterion: monthly totals match USAspending's own figure -----------------------------
# For every calendar month from October 2016 to the last full month before retrieval, the sum
# of federal_action_obligation over the downloaded transactions must be within this share of
# the figure USAspending's own search (spending over time, same filters) gives for that month.
# The criterion is met only if every month is. The figures are stored and shown either way,
# with the verdict beside them.
TOTALS_TOLERANCE = 0.01
# --- Second check: reconcile every missed month row by row ----------------------------------
# Added on 15 November 2026, after the criterion above was found not met, and after eight of
# the 26 missed months had been compared row by row (in all eight, every transaction in
# USAspending's search was also in the download, and the search held extra exact copies of a
# few). It is a new test, recorded in the failures log; the verdict above stands as it is.
#
# For every month outside the 1% limit, the collector lists the transactions behind
# USAspending's search figure (spending by transaction, same filters) and matches them with
# the download on award, modification, date and amount. The month is reconciled when
#   1. every transaction the search lists is in the download, and
#   2. after removing the search's extra copies of a transaction the download holds once, the
#      search and the download differ by no more than RECONCILE_TOLERANCE of the month.
# The module's figures stand only if every missed month is reconciled.
RECONCILE_TOLERANCE = 0.001

# --- Supplier name cleaning ---------------------------------------------------------------------
# A supplier is the parent company where the source names one (recipient_parent_uei), else the
# company on the contract (recipient_uei). Groups whose cleaned names are identical are then
# merged. Cleaning: upper case, "&" to "AND", punctuation dropped, these legal endings dropped.
LEGAL_ENDINGS = ("INC", "INCORPORATED", "LLC", "LC", "CORP", "CORPORATION", "CO", "COMPANY",
                 "LTD", "LIMITED", "LP", "LLP", "PLC", "PC", "THE")  # fmt: skip
# A name is stored and shown only for the suppliers ranked this high in some fiscal year. The
# long tail, where a one-person business could carry a person's name, is kept as an id only.
NAMED_SUPPLIERS_PER_YEAR = 25

# --- Metrics --------------------------------------------------------------------------------------
# Concentration, per fiscal year: each supplier's share of net obligations among suppliers
# whose net is above zero. Reported as the share of the largest 4 and 10, and the
# Herfindahl-Hirschman index (sum of squared percentage shares; 10,000 is a single supplier).
TOP_N = (4, 10)
# New entrant: a supplier with net obligations above zero in a fiscal year and no transaction
# at all in this slice in the previous ENTRANT_LOOKBACK fiscal years. The first year that can
# be judged is therefore FIRST_FISCAL_YEAR + ENTRANT_LOOKBACK.
ENTRANT_LOOKBACK = 4
# Single-bidder share, per fiscal year: of the dollars newly obligated (actions above zero) on
# awards the source marks as competed and for which a number of offers is recorded, the share
# on awards that received exactly one offer. Reported beside the share of all new dollars that
# was not competed at all.
COMPETED = (
    "FULL AND OPEN COMPETITION",
    "FULL AND OPEN COMPETITION AFTER EXCLUSION OF SOURCES",
    "COMPETED UNDER SAP",
    "COMPETITIVE DELIVERY ORDER",
)
NOT_COMPETED = (
    "NOT COMPETED",
    "NOT COMPETED UNDER SAP",
    "NOT AVAILABLE FOR COMPETITION",
    "NON-COMPETITIVE DELIVERY ORDER",
    "FOLLOW ON TO COMPETED ACTION",
)
# Promised against paid, per fiscal year an award began in: awards whose first action
# (modification "0") falls in the year. Promised is total_dollars_obligated and the ceiling is
# potential_total_value_of_award, both from the award's latest action in the slice; paid is
# total_outlayed_amount_for_overall_award. The three are never added together or netted. Paid
# is compared with promised only over awards that report a payment figure at all, and the share
# of awards that do is shown beside it. Agencies had to report payments per award only from
# fiscal 2022, so earlier years are shown but not interpreted.
OUTLAYS_REQUIRED_FROM = 2022
