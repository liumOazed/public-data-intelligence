"""Turn the stored procurement sums into the tables the Federal Procurement pages read."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pdi.common import findings
from pdi.common.export import write_table
from pdi.macro import config as macro
from pdi.macro import flags
from pdi.procurement import config, tables

M = config.MODULE
SOURCES = [config.SOURCE]
TABLES = ("monthly", "by_group", "agencies", "concentration", "top_suppliers", "competition",
          "promised_paid", "totals_check")  # fmt: skip
LIMITS = [
    "Contracts for computer hardware only, chosen by Product and Service Code. The buyer picks "
    "one code for a whole contract, so a contract that mixes hardware with software or services "
    "is counted in full or not at all.",
    "The code list was rewritten on 1 October 2020. Figures before and after that month are "
    "not on the same footing, and the new codes cover hardware together with software sold "
    "under a perpetual licence.",
    "Department of Defense records are published 90 days late, so the newest three months "
    "miss most defence buying. They are marked provisional.",
    "Obligations are money promised, not money paid. An action can be negative when money is "
    "taken back, so a month's total is a net figure.",
    "Agencies correct old records; every figure can change on the next retrieval.",
]
SUPPLIER_LIMITS = [
    "A supplier is the parent company the source names, else the company on the contract; "
    "groups with the same name after cleaning are merged. Two companies that share a cleaned "
    "name would be merged wrongly. A company whose recorded parent changed over the years "
    "(a takeover, or a correction) appears under each parent and is not merged, and the parent "
    "can be an investment firm that owns the company on the contract.",
    "Most hardware is bought through resellers. The supplier here is the company that holds "
    "the contract, not the maker of the equipment.",
    "Only the largest suppliers in each fiscal year are named. The rest are counted without a "
    "name.",
]
ARITHMETIC = findings.observation(
    "Sums of the obligations agencies reported. No model is fitted and no cause is claimed."
)
SUPPLIER_FINDING = findings.observation(
    "Shares of net obligations by supplier after name cleaning. Arithmetic on reported "
    "figures; the rules were fixed before the figures were computed."
)
COMPETITION_FINDING = findings.observation(
    "Shares of newly obligated dollars by what the contracting officer recorded about "
    "competition. Arithmetic on reported fields, not a judgement of any award."
)
PAID_FINDING = findings.observation(
    "Three reported figures side by side for the awards that began in each fiscal year. "
    "They are never added together."
)
RECONCILE_FINDING = findings.observation(
    "Transactions matched one by one between the two answers on award, modification, date and "
    "amount. Counting, not a model."
)
CHECK_FINDING = findings.observation(
    "The same filters asked two ways of the same source: the transaction download summed by "
    "month, and the monthly figure from USAspending's own search."
)


UNUSUAL_RULE = (
    "the change in net obligations from the month before, in the latest month that is not "
    "provisional, is 2.5 spreads or more from what that calendar month brought in up to ten "
    "earlier years (the Macro Monitor's rule)."
)
UNUSUAL_FINDING = findings.observation(
    "Arithmetic on summed transactions: the latest settled month's change against the same "
    "calendar month in earlier years. No model; it says the month was unusual, not why."
)


def unusual_items(by_month: pd.DataFrame) -> list[dict]:
    settled = by_month[~by_month["provisional"]]
    if settled.empty:
        return []
    s = settled.set_index("period")["obligations"].sort_index().astype(float)
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
    df = flags.surprises(100 * np.log(s.where(s > 0)).diff(), "nsa")
    period, last = df.index[-1], df.iloc[-1]
    if pd.isna(last["z"]) or abs(last["z"]) < macro.FLAG_Z:
        return []
    direction = "rose" if last["move"] > 0 else "fell"
    sentence = (
        f"Federal obligations for computer hardware {direction} "
        f"{abs(100 * np.expm1(last['move'] / 100)):.1f}% from the month before in "
        f"{period:%B %Y}, to ${s.iloc[-1] / 1e6:,.0f} million, {abs(last['z']):.1f} spreads "
        f"from what {period:%B} usually brings."
    )
    return [findings.item(
        module=M, subject="computer hardware", period=period, sentence=sentence,
        rule="same calendar month", measure=last["z"], threshold=macro.FLAG_Z,
        finding=UNUSUAL_FINDING, page="procurement/index.qmd", table="procurement/monthly.json",
    )]  # fmt: skip


def _empty(reason: str, retrieved: dict) -> None:
    for table in TABLES:
        write_table(M, table, pd.DataFrame(), description=reason, retrieved_at=retrieved,
                    sources=SOURCES, extra={"status": reason})  # fmt: skip
    findings.write_items(M, [], rule=UNUSUAL_RULE, sources=SOURCES, retrieved_at=retrieved,
                         extra={"status": reason})  # fmt: skip


def provisional_from(retrieved_at: str | None) -> pd.Timestamp:
    """First month whose end is fewer than PROVISIONAL_DAYS before retrieval."""
    when = datetime.fromisoformat(retrieved_at) if retrieved_at else datetime.now()
    cut = pd.Timestamp(when.date() - timedelta(days=config.PROVISIONAL_DAYS))
    return cut.to_period("M").to_timestamp()


def fiscal_year_of(period: pd.Series) -> pd.Series:
    return tables.fiscal_year(period)


def year_is_provisional(fy: int, first_provisional: pd.Timestamp) -> bool:
    return pd.Timestamp(fy, 9, 1) >= first_provisional


def main() -> None:
    retrieved = json.loads(config.RETRIEVED.read_text()) if config.RETRIEVED.exists() else {}
    if not config.MONTHLY.exists():
        _empty("Federal contracts have not been collected yet.", retrieved)
        print("nothing to export: federal contracts have not been collected yet")
        return
    monthly = pd.read_parquet(config.MONTHLY)
    sy = pd.read_parquet(config.SUPPLIER_YEAR)
    comp = pd.read_parquet(config.COMPETITION)
    cohorts = pd.read_parquet(config.COHORTS)
    check = pd.read_parquet(config.CHECK)
    names = json.loads(config.SUPPLIERS.read_text())
    cleaning = json.loads(config.CLEANING.read_text())
    diagnosis = json.loads(config.DIAGNOSIS.read_text()) if config.DIAGNOSIS.exists() else []

    first_prov = provisional_from(retrieved.get(config.SOURCE))
    last_month = check["period"].max()
    monthly = monthly[monthly["period"] <= last_month]
    monthly["fiscal_year"] = fiscal_year_of(monthly["period"])
    monthly["group"] = monthly["psc"].map(lambda c: config.PSC[c][0])
    # Yearly tables hold only fiscal years that had ended when the data were retrieved. The
    # year in progress appears in the monthly table alone.
    when = retrieved.get(config.SOURCE)
    as_of = pd.Timestamp(datetime.fromisoformat(when).date()) if when else last_month
    years = [int(y) for y in sorted(monthly["fiscal_year"].unique())
             if pd.Timestamp(int(y), 9, 30) < as_of]  # fmt: skip
    prov_years = [y for y in years if year_is_provisional(y, first_prov)]
    sy, comp = sy[sy["fiscal_year"].isin(years)], comp[comp["fiscal_year"].isin(years)]
    cohorts = cohorts[cohorts["fiscal_year"].isin(years)]
    extra = {
        "latest_month": last_month.strftime("%Y-%m"),
        "first_provisional_month": first_prov.strftime("%Y-%m"),
        "provisional_fiscal_years": prov_years,
        "code_change": config.CODE_CHANGE,
        "first_fiscal_year": config.FIRST_FISCAL_YEAR,
    }

    by_month = monthly.groupby("period", as_index=False).agg(
        obligations=("obligations", "sum"), actions=("actions", "sum")
    )
    by_month["obligations_12m"] = by_month["obligations"].rolling(12).sum()
    by_month["provisional"] = by_month["period"] >= first_prov
    write_table(M, "monthly", by_month, sources=SOURCES, retrieved_at=retrieved,
                limitations=LIMITS, extra={**extra, "finding": ARITHMETIC},
                description="Net obligations on federal contracts for computer hardware, by "
                "month of the action, with the sum over twelve months.")  # fmt: skip
    findings.write_items(M, unusual_items(by_month), rule=UNUSUAL_RULE, sources=SOURCES,
                         retrieved_at=retrieved, limitations=LIMITS, extra=extra)  # fmt: skip

    yearly = monthly[monthly["fiscal_year"].isin(years)]
    groups = yearly.groupby(["fiscal_year", "group"], as_index=False)["obligations"].sum()
    groups["provisional"] = groups["fiscal_year"].isin(prov_years)
    write_table(M, "by_group", groups, sources=SOURCES, retrieved_at=retrieved,
                limitations=LIMITS, extra={**extra, "finding": ARITHMETIC},
                description="Net obligations by fiscal year and kind of hardware.")  # fmt: skip

    agencies = yearly.groupby(["fiscal_year", "agency"], as_index=False)["obligations"].sum()
    agencies["share_pct"] = (
        100
        * agencies["obligations"]
        / agencies.groupby("fiscal_year")["obligations"].transform("sum")
    )
    agencies["provisional"] = agencies["fiscal_year"].isin(prov_years)
    agencies = agencies.sort_values(["fiscal_year", "obligations"], ascending=[False, False],
                                    ignore_index=True)  # fmt: skip
    write_table(M, "agencies", agencies, sources=SOURCES, retrieved_at=retrieved,
                limitations=LIMITS, extra={**extra, "finding": ARITHMETIC},
                description="Net obligations by fiscal year and awarding agency.")  # fmt: skip

    conc = tables.concentration(sy).merge(tables.entrants(sy), on="fiscal_year", how="left")
    conc["provisional"] = conc["fiscal_year"].isin(prov_years)
    supplier_extra = {**extra, "finding": SUPPLIER_FINDING, "cleaning": cleaning,
                      "entrant_lookback_years": config.ENTRANT_LOOKBACK,
                      "named_per_year": config.NAMED_SUPPLIERS_PER_YEAR}  # fmt: skip
    write_table(M, "concentration", conc, sources=SOURCES, retrieved_at=retrieved,
                limitations=LIMITS + SUPPLIER_LIMITS, extra=supplier_extra,
                description="Per fiscal year: suppliers with net obligations above zero, the "
                "share of the largest 4 and 10, the Herfindahl-Hirschman index, and suppliers "
                f"with no action in the {config.ENTRANT_LOOKBACK} years before.")  # fmt: skip

    paid = sy[sy["obligations"] > 0].copy()
    paid["share_pct"] = (
        100 * paid["obligations"] / paid.groupby("fiscal_year")["obligations"].transform("sum")
    )
    paid["rank"] = paid.groupby("fiscal_year")["obligations"].rank(ascending=False).astype(int)
    top = paid[paid["rank"] <= config.NAMED_SUPPLIERS_PER_YEAR].copy()
    top["name"] = top["supplier"].map(names).fillna("")
    top = top[top["name"] != ""]
    top = top[["fiscal_year", "rank", "name", "obligations", "share_pct"]].sort_values(
        ["fiscal_year", "rank"], ascending=[False, True], ignore_index=True
    )
    write_table(M, "top_suppliers", top, sources=SOURCES, retrieved_at=retrieved,
                limitations=LIMITS + SUPPLIER_LIMITS, extra=supplier_extra,
                description=f"The {config.NAMED_SUPPLIERS_PER_YEAR} largest suppliers by net "
                "obligations in each fiscal year, after name cleaning.")  # fmt: skip

    single = tables.single_bidder(comp)
    single["provisional"] = single["fiscal_year"].isin(prov_years)
    write_table(M, "competition", single, sources=SOURCES, retrieved_at=retrieved,
                limitations=[
                    *LIMITS,
                    "The number of offers is what the contracting officer typed. For an order "
                    "placed under an umbrella agreement it can describe the competition for the "
                    "agreement, not for the order.",
                    "Only dollars on awards marked as competed and with a number of offers "
                    "recorded are in the single-bidder share; the last column says how much of "
                    "the year's new dollars that covers.",
                ],
                extra={**extra, "finding": COMPETITION_FINDING},
                description="Per fiscal year, of newly obligated dollars: the share on "
                "competed awards that drew one offer, and the share not competed.")  # fmt: skip

    pp = cohorts.copy()
    pp["awards_reporting_paid_pct"] = 100 * pp["awards_reporting_paid"] / pp["awards"]
    pp["paid_share_of_promised_pct"] = 100 * pp["paid"] / pp["promised_where_paid_reported"]
    pp["promised_share_of_ceiling_pct"] = 100 * pp["promised"] / pp["ceiling"]
    pp["payments_required"] = pp["fiscal_year"] >= config.OUTLAYS_REQUIRED_FROM
    pp["provisional"] = pp["fiscal_year"].isin(prov_years)
    write_table(M, "promised_paid", pp, sources=SOURCES, retrieved_at=retrieved,
                limitations=[
                    *LIMITS,
                    "Paid is the award's total as the agency last reported it, and it is missing "
                    "for many awards. Agencies had to report it per award only from fiscal "
                    f"{config.OUTLAYS_REQUIRED_FROM}. A missing figure is not a payment of zero.",
                    "A low paid share for a recent year is expected: equipment is paid for on "
                    "delivery, and contracts run for years.",
                    "Promised and ceiling are read from each award's latest action in this "
                    "slice. An award that later changed product code is read at its last action "
                    "under a hardware code.",
                ],
                extra={**extra, "finding": PAID_FINDING,
                       "payments_required_from": config.OUTLAYS_REQUIRED_FROM},
                description="For the awards that began in each fiscal year: money promised, the "
                "ceiling if every option is used, and money paid, with how many awards report "
                "a paid figure.")  # fmt: skip

    out = check.rename(columns={"obligations": "ours"})
    out["gap_pct"] = 100 * out["gap_share"]
    out = out[["period", "ours", "published", "gap_pct", "within_limit"]]
    met = bool(check["within_limit"].all())
    write_table(M, "totals_check", out, sources=SOURCES, retrieved_at=retrieved,
                description="Monthly obligations summed from the downloaded transactions "
                "against the monthly figure USAspending's own search gives for the same "
                "filters.",
                extra={**extra, "finding": CHECK_FINDING, "criterion_met": met,
                       "tolerance_pct": 100 * config.TOTALS_TOLERANCE,
                       "months": int(len(check)),
                       "months_within_limit": int(check["within_limit"].sum()),
                       "worst_month_gap_pct": float(100 * check["gap_share"].max()),
                       "ours_total": float(check["obligations"].sum()),
                       "published_total": float(check["published"].sum()),
                       "reconciliation": diagnosis,
                       "reconcile_tolerance_pct": 100 * config.RECONCILE_TOLERANCE,
                       "all_missed_months_reconciled": bool(
                           diagnosis and all(d.get("reconciled") for d in diagnosis)
                           and len(diagnosis) == int((~check["within_limit"]).sum())),
                       "reconcile_finding": RECONCILE_FINDING})  # fmt: skip
    within = int(check["within_limit"].sum())
    print(f"exported procurement tables to {last_month:%Y-%m}; {within} of {len(check)} months "
          "within the limit")  # fmt: skip


if __name__ == "__main__":
    main()
