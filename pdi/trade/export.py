"""Turn stored US imports into the tables the Trade pages read."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pdi.common import findings
from pdi.common.export import write_table
from pdi.macro import config as macro
from pdi.macro import flags
from pdi.trade import collect, config, lead, shares

M = config.MODULE
SOURCES = [config.SOURCE]
LEAD_SOURCES = [config.SOURCE, "imf"]
TABLES = ("partners", "share_history", "chapter_shifts", "china_shift", "totals_check",
          "lead_test")  # fmt: skip
LIMITS = [
    "US-reported imports only: general imports at customs value, by country of origin. Goods "
    "made in one country and finished in another are counted under the last one.",
    "A rising share is not proof that production moved. Goods can be routed through a third "
    "country, and a share also rises when other suppliers' sales fall.",
    "Shares are of the value in dollars, so a price change moves a share without any change "
    "in volume.",
    "Chapters are the 97 two-digit groups of the Harmonized System; a shift inside a chapter "
    "(from one kind of machine to another) is invisible here.",
    "Recent months are revised by Census for several months after first publication.",
]
SHARE_FINDING = findings.observation(
    "Arithmetic on published import values: each partner's share of US imports over twelve "
    "months, and how that share has changed. No model is fitted and no cause is claimed."
)
CHECK_FINDING = findings.observation(
    "The chapters are summed and compared with the all-commodity total Census publishes for "
    "the same partner and month."
)


UNUSUAL_RULE = (
    "the change in US imports from a partner from the month before, in its latest month, is 2.5 "
    "spreads or more from what that calendar month brought in up to ten earlier years (the "
    "Macro Monitor's rule)."
)
UNUSUAL_FINDING = findings.observation(
    "Arithmetic on the all-commodity import totals Census publishes: the latest monthly change "
    "against the same calendar month in earlier years. No model; it says the month was unusual, "
    "not why."
)


def unusual_items(totals: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """Score each partner's monthly change in imports; list the latest month when flagged."""
    names = partner_names()
    items, scored = [], []
    for partner, g in totals.groupby("partner"):
        s = g.set_index("period")["value"].sort_index().astype(float)
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
        s = s.where(s > 0)
        df = flags.surprises(100 * np.log(s).diff(), "nsa").dropna(subset=["move"])
        if df.empty:
            continue
        last = df.iloc[-1]
        period = df.index[-1]
        scored.append({"partner": partner, "period": period, "z": last["z"]})
        if pd.isna(last["z"]) or abs(last["z"]) < macro.FLAG_Z:
            continue
        direction = "rose" if last["move"] > 0 else "fell"
        sentence = (
            f"US imports from {names.get(partner, partner)} {direction} "
            f"{abs(100 * np.expm1(last['move'] / 100)):.1f}% from the month before in "
            f"{period:%B %Y}, {abs(last['z']):.1f} spreads from what {period:%B} usually brings."
        )
        items.append(findings.item(
            module=M, subject=partner, period=period, sentence=sentence,
            rule="same calendar month", measure=last["z"], threshold=macro.FLAG_Z,
            finding=UNUSUAL_FINDING, page="trade/index.qmd", table="trade/findings.json",
        ))  # fmt: skip
    return items, pd.DataFrame(scored)


def _empty(reason: str, retrieved: dict) -> None:
    for table in TABLES:
        write_table(M, table, pd.DataFrame(), description=reason, retrieved_at=retrieved,
                    sources=LEAD_SOURCES if table == "lead_test" else SOURCES,
                    extra={"status": reason})  # fmt: skip
    findings.write_items(M, [], rule=UNUSUAL_RULE, sources=SOURCES, retrieved_at=retrieved,
                         extra={"status": reason})  # fmt: skip


def partner_names() -> dict[str, str]:
    return {code: name for code, (_c, name) in config.PARTNERS.items()}


def lead_finding(result: pd.DataFrame) -> dict:
    passed = result.loc[result["passes"].fillna(False), "country"].tolist()
    names = partner_names()
    if passed:
        who = ", ".join(names[c] for c in passed)
        return findings.finding(
            findings.PREDICTION,
            f"For {who}, adding US imports beat the persistence baseline on months the model "
            "never saw, by more than twice the standard error of the gain.",
        )
    return findings.observation(
        "No signal found: for no economy did US imports beat the persistence baseline by more "
        "than twice the standard error of the gain. The correlations are shown as observations."
    )


def main() -> None:
    retrieved = json.loads(config.RETRIEVED.read_text()) if config.RETRIEVED.exists() else {}
    if not config.IMPORTS.exists():
        _empty("US imports have not been collected yet.", retrieved)
        print("nothing to export: US imports have not been collected yet")
        return
    imports = pd.read_parquet(config.IMPORTS)
    totals = pd.read_parquet(config.TOTALS)
    chapter_names = json.loads(config.CHAPTERS.read_text()) if config.CHAPTERS.exists() else {}
    names = partner_names()
    latest = imports["period"].max()

    items, scored = unusual_items(totals)
    findings.write_items(M, items, rule=UNUSUAL_RULE, sources=SOURCES, retrieved_at=retrieved,
                         limitations=LIMITS,
                         extra={"partners_scored": int(len(scored))})  # fmt: skip

    rolled = shares.with_shares(shares.twelve_month(imports))
    shift = shares.shifts(rolled)
    extra = {
        "finding": SHARE_FINDING,
        "latest_month": latest.strftime("%Y-%m"),
        "base_year": config.BASE_YEAR,
        "min_chapter_usd": config.MIN_CHAPTER_USD,
    }

    partners = shift[shift["chapter"] == "ALL"].assign(name=lambda d: d["partner"].map(names))
    partners = partners[
        ["partner", "name", "period", "usd_12m", "share_pct", "share_base", "share_year_ago",
         "shift_since_base", "shift_on_year"]
    ].sort_values("shift_since_base", ascending=False, ignore_index=True)  # fmt: skip
    write_table(M, "partners", partners, sources=SOURCES, retrieved_at=retrieved,
                limitations=LIMITS, extra=extra,
                description="Each partner's share of all US goods imports over the latest twelve "
                f"months, against {config.BASE_YEAR} and against a year earlier.")  # fmt: skip

    history = rolled[rolled["chapter"] == "ALL"].assign(name=lambda d: d["partner"].map(names))
    write_table(M, "share_history", history[["partner", "name", "period", "usd_12m", "share_pct"]],
                sources=SOURCES, retrieved_at=retrieved, limitations=LIMITS, extra=extra,
                description="Each partner's share of all US goods imports, twelve months to "
                "each month.")  # fmt: skip

    goods = shift[~shift["chapter"].isin((*config.SPECIAL_CHAPTERS, "ALL"))].assign(
        name=lambda d: d["partner"].map(names),
        chapter_name=lambda d: d["chapter"].map(chapter_names).fillna(""),
    )
    goods = goods[
        ["partner", "name", "chapter", "chapter_name", "usd_12m", "world_12m", "share_pct",
         "share_base", "shift_since_base", "shift_on_year"]
    ].sort_values("shift_since_base", ascending=False, ignore_index=True)  # fmt: skip
    write_table(M, "chapter_shifts", goods, sources=SOURCES, retrieved_at=retrieved,
                limitations=LIMITS, extra=extra,
                description="Share of US imports by partner and product chapter, latest twelve "
                f"months, with the change since {config.BASE_YEAR}, in points.")  # fmt: skip

    china = shares.china_and_sea(shift, chapter_names)
    write_table(M, "china_shift", china, sources=SOURCES, retrieved_at=retrieved,
                limitations=LIMITS, extra=extra,
                description="For each large product chapter: the change in China's share of US "
                f"imports since {config.BASE_YEAR}, beside the change for each Southeast Asian "
                "economy, in percentage points.")  # fmt: skip

    check = collect.check_totals(imports, totals)
    check["year"] = check["period"].dt.year
    yearly = check.groupby(["partner", "year"], as_index=False).agg(
        months=("period", "size"), chapters_sum=("value_sum", "sum"),
        published_total=("value_total", "sum"), worst_month_gap_pct=("gap_share", "max"),
    )  # fmt: skip
    yearly["gap_pct"] = (
        100 * (yearly["chapters_sum"] - yearly["published_total"]).abs() / yearly["published_total"]
    )
    yearly["worst_month_gap_pct"] *= 100
    yearly["name"] = yearly["partner"].map(names)
    worst = float(check["gap_share"].max())
    write_table(M, "totals_check", yearly, sources=SOURCES, retrieved_at=retrieved,
                description="Sum of the product chapters against the all-commodity total Census "
                "publishes, by partner and year.",
                extra={"finding": CHECK_FINDING, "tolerance_pct": 100 * config.TOTALS_TOLERANCE,
                       "worst_month_gap_pct": 100 * worst,
                       "criterion_met": bool(worst <= config.TOTALS_TOLERANCE),
                       "partner_months": int(len(check))})  # fmt: skip

    obs = pd.read_parquet(macro.OBSERVATIONS) if macro.OBSERVATIONS.exists() else None
    result = lead.run(totals, obs)
    corr = pd.json_normalize(result.get("correlation", pd.Series(dtype=object)).map(
        lambda c: c if isinstance(c, dict) else {})).add_prefix("corr_shift_")  # fmt: skip
    flat = pd.concat([result.drop(columns=["correlation"], errors="ignore"), corr], axis=1)
    flat.insert(1, "name", flat["country"].map(names))
    macro_retrieved = {}
    if (macro.STORE_DIR / "retrieved.json").exists():
        macro_retrieved = json.loads((macro.STORE_DIR / "retrieved.json").read_text())
    ran = bool(flat.get("mae_model", pd.Series(dtype=float)).notna().any())
    write_table(M, "lead_test", flat, sources=LEAD_SOURCES,
                retrieved_at={**retrieved, "imf": macro_retrieved.get("imf")},
                limitations=[
                    "Scored on today's revised figures for both series, not on the values first "
                    "published, so the test is kinder than real time.",
                    "The economy's own exports are its exports to every country; the US is only "
                    "one buyer, so a weak link is expected for economies that sell little to it.",
                    "The baseline assumes the economy's own figure is exactly one month behind. "
                    "When it is further behind, US imports are worth more than this test shows.",
                ],
                description="Lead-time test: does the growth of US imports from an economy improve "
                "a same-month estimate of that economy's own total export growth over a "
                "persistence baseline, on months after the fitting period?",
                extra={"finding": lead_finding(flat) if ran else None, "test_ran": ran,
                       "train_end": config.LEAD_TRAIN_END,
                       "se_multiple": config.LEAD_SE_MULTIPLE,
                       "status": None if ran else "The economies' own export figures have not "
                       "been collected yet, so the test has not run."})  # fmt: skip
    config.LEAD_TEST.write_text(flat.to_json(orient="records", indent=1) + "\n")
    print(f"exported trade tables to {latest:%Y-%m}; totals within {worst:.4%}; lead test "
          f"{'ran' if ran else 'not run'}")  # fmt: skip


if __name__ == "__main__":
    main()
