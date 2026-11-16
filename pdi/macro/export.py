"""Export step for the Macro Monitor: everything the site shows, as files under
site/data/v1/macro/.

Computes the flags (seconds), reads AutoGluon's saved result if there is one, and writes one
JSON file per table. Missing inputs produce empty tables with an explanation in the metadata
rather than a failed build.
"""

from __future__ import annotations

import json

import pandas as pd

from pdi.common import findings
from pdi.common.export import write_table
from pdi.macro import config, flags, store

M = config.MODULE
SOURCES = ["fred", "bis", "opendosm", "imf", "indeed_postings"]
LIMITS = [
    "Figures are the latest published values and are revised by their publishers. A flag can "
    "appear or disappear when a figure is revised.",
    "A flag means a move was large against the series' own last ten years. It does not say why "
    "the move happened or what comes next, and it is a research lead, not advice.",
    "With a threshold of 2.5 spreads, roughly one or two readings in a hundred are flagged by "
    "chance alone, and more for series with fat tails. The backtest shows how often.",
    "Southeast Asian series come from the BIS and the IMF, which compile them from national "
    "sources on one definition per measure. The BIS has no consumer prices, policy rate or real "
    "effective exchange rate for Vietnam (its consumer prices come from the IMF) and no policy "
    "rate for Singapore. Trade and reserves are in US dollars, so a move can be a currency move.",
    "This product uses the FRED® API but is not endorsed or certified by the Federal Reserve "
    "Bank of St. Louis.",
    "Source: BIS statistics; FRED (Federal Reserve Bank of St. Louis, from US federal "
    "agencies); Department of Statistics Malaysia (OpenDOSM); International Monetary Fund, "
    "International Trade in Goods, International Liquidity and Consumer Price Index datasets "
    "(https://data.imf.org); Indeed Hiring Lab (CC BY 4.0). "
    "Surprises, flags and forecasts are this site's calculations, not the publishers'.",
]
FORECAST_LIMITS = [
    "The target is each series' monthly move one month ahead. Errors are divided by the "
    "series' own mean absolute month-to-month change in the training period.",
    "Models are fitted once on data before the validation year and are not refitted on later "
    "months when forecasting them.",
    "Scored on latest published values, not on the values first published, so the test is "
    "kinder than real time. First-published values are being stored from the first run on.",
]


def _empty(reason: str, retrieved: dict) -> None:
    note = {"status": reason}
    for table in ("flags_latest", "flags_backtest", "series", "history", "neighbours",
                  "leaderboard", "holdout", "folds", "forecasts"):  # fmt: skip
        write_table(M, table, pd.DataFrame(), sources=SOURCES, description=reason,
                    retrieved_at=retrieved, extra=note)  # fmt: skip
    findings.write_items(M, [], rule=UNUSUAL_RULE, sources=SOURCES, retrieved_at=retrieved,
                         extra=note)  # fmt: skip


UNUSUAL_RULE = (
    "a series' latest month is flagged: its move is 2.5 spreads or more from its own ten-year "
    "history, or from its Southeast Asian neighbours."
)


def unusual_items(current: pd.DataFrame) -> list[dict]:
    """The shared-format items: every flag on a series' own latest month."""
    out = []
    for _, r in current.iterrows():
        rule = "own history" if r["kind"] == "own history" else "neighbours"
        out.append(findings.item(
            module=M, subject=r["series_id"], period=r["period"], sentence=r["sentence"],
            rule=rule, measure=r["z"], threshold=config.FLAG_Z,
            finding={**flags.FLAG_FINDING, "label": r["label"]},
            page="macro/index.qmd", table="macro/flags_latest.json",
        ))  # fmt: skip
    return out


def series_table(obs: pd.DataFrame, scores: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    cat = config.catalogue()
    latest = scores.dropna(subset=["move"]).sort_values("period").groupby("series_id").tail(1)
    value = obs.sort_values("period").groupby("series_id").tail(1).set_index("series_id")["value"]
    n_obs = obs.groupby("series_id").size()
    cur = set()
    if not flagged.empty:
        own = flagged[flagged["kind"] == "own history"]
        cur = set(zip(own["series_id"], own["period"], strict=True))
    rows = []
    for _, r in latest.iterrows():
        s = cat[r["series_id"]]
        rows.append(
            {
                "series_id": s.id,
                "country": config.COUNTRIES[s.country],
                "indicator": s.indicator,
                "name": s.name,
                "units": s.units,
                "source": s.source,
                "source_code": s.source_code,
                "latest_month": r["period"],
                "value": value[s.id],
                "move": r["move"],
                "move_unit": "percent" if s.transform == "log" else "change in value",
                "expected_move": r["expected"],
                "z": r["z"],
                "flagged": (s.id, r["period"]) in cur,
                "seasonal_comparison": s.seasonal == "nsa",
                "months_of_history": int(n_obs[s.id]),
            }
        )
    return pd.DataFrame(rows).sort_values("z", key=lambda z: -z.abs(), ignore_index=True)


def backtest_summary(bt: pd.DataFrame, scores: pd.DataFrame, start: pd.Timestamp) -> dict:
    scored = scores[(scores["period"] >= start) & scores["z"].notna()]
    out = {
        "window_start": f"{start:%Y-%m}",
        "readings_scored": int(len(scored)),
        "flags": int(len(bt)),
    }
    for kind in ("own history", "neighbours"):
        k = bt[bt["kind"] == kind] if len(bt) else bt
        out[kind] = {
            "flags": int(len(k)),
            **{o: int((k["outcome"] == o).sum()) for o in ("held", "reversed", "too early")},
        }
    own = out["own history"]["flags"]
    out["own_flag_rate"] = own / len(scored) if len(scored) else None
    return out


def export_flags(obs: pd.DataFrame, retrieved: dict) -> None:
    scores, gaps, flagged = flags.build(obs)
    config.STORE_DIR.mkdir(parents=True, exist_ok=True)
    flagged.to_parquet(config.FLAGS, index=False)
    common = dict(sources=SOURCES, retrieved_at=retrieved, limitations=LIMITS)
    rule = {
        "flag_z": config.FLAG_Z,
        "history_months": config.HISTORY_MONTHS,
        "follow_months": config.FOLLOW_MONTHS,
        "reversal_share": config.REVERSAL_SHARE,
        "finding": flags.FLAG_FINDING,
    }

    # A flag is current when it is on the series' own latest month.
    last = scores.dropna(subset=["move"]).groupby("series_id")["period"].max()
    cols = ["period", "country", "indicator", "kind", "z", "outcome", "sentence", "series_id",
            "label"]  # fmt: skip
    if flagged.empty:
        flagged = pd.DataFrame(columns=cols + ["move", "expected"])
    current = flagged[flagged["period"] == flagged["series_id"].map(last)]
    current = current.assign(country=current["country"].map(config.COUNTRIES))
    current = current.sort_values("z", key=lambda z: -z.abs())
    write_table(
        M, "flags_latest", current[cols],
        description="Flags on each series' latest published month: moves of 2.5 spreads or "
        "more against the series' own ten-year history, or against its neighbours.",
        extra={**rule, "n_series": int(scores["series_id"].nunique())}, **common,
    )  # fmt: skip
    findings.write_items(M, unusual_items(current), rule=UNUSUAL_RULE, **common)

    end = scores.dropna(subset=["move"])["period"].max()
    start = end - pd.DateOffset(months=config.BACKTEST_MONTHS - 1)
    bt = flagged[flagged["period"] >= start]
    bt = bt.assign(country=bt["country"].map(config.COUNTRIES))
    write_table(
        M, "flags_backtest", bt[cols],
        description=f"Every flag the rule raised in the last {config.BACKTEST_MONTHS} months, "
        "with whether the move held over the next three months or was given back.",
        extra={**rule, "summary": backtest_summary(bt, scores, start)}, **common,
    )  # fmt: skip

    write_table(
        M, "series", series_table(obs, scores, flagged),
        description="One row per series: the latest value, its monthly move, the usual move "
        "and how many spreads apart they are.",
        extra=rule, **common,
    )  # fmt: skip

    hist = scores.merge(obs[["series_id", "period", "value"]], on=["series_id", "period"])
    hist = hist[hist["period"] >= end - pd.DateOffset(years=10)]
    write_table(
        M, "history", hist[["series_id", "period", "value", "move", "expected", "z"]].round(4),
        description="Ten years of monthly values, moves and surprises for every series.",
        extra=rule, **common,
    )  # fmt: skip

    cat = config.catalogue()
    recent = gaps[gaps["period"] >= end - pd.DateOffset(months=12)].dropna(subset=["own_z"])
    recent = recent.assign(
        country=recent["country"].map(config.COUNTRIES),
        name=recent["series_id"].map(lambda s: cat[s].name),
    )
    write_table(
        M, "neighbours",
        recent[["period", "country", "indicator", "name", "own_z", "neighbours_median_z",
                "n_neighbours", "gap", "z"]].round(3),
        sources=["bis", "imf"], retrieved_at=retrieved, limitations=LIMITS,
        description="For measures the BIS or IMF defines the same way in each country: each "
        "economy's "
        "surprise beside the median surprise of the other Southeast Asian economies.",
        extra=rule,
    )  # fmt: skip


def export_forecasts(retrieved: dict) -> None:
    common = dict(sources=SOURCES, retrieved_at=retrieved, limitations=FORECAST_LIMITS)
    if not config.AUTOML_SUMMARY.exists():
        note = {"status": "AutoGluon has not been trained on these series yet. The weekly "
                "training job writes this table."}  # fmt: skip
        for table in ("leaderboard", "holdout", "folds", "forecasts"):
            write_table(M, table, pd.DataFrame(), description=note["status"], extra=note, **common)
        return
    s = json.loads(config.AUTOML_SUMMARY.read_text())
    finding = findings.from_test(
        s["passed"],
        passed=f"The picked model ({s['picked']}) forecast next month's moves on "
        f"{s['holdout_n_forecasts']:,} holdout readings it never saw better than the naive "
        "baseline, by more than the fold-to-fold spread.",
        failed="No model beat the naive baseline on the holdout months by more than the "
        "fold-to-fold spread, so nothing on these pages is a forecast. No signal found.",
        otherwise=findings.EXPLANATION,
    )
    verdict = {**{k: v for k, v in s.items() if k != "folds"}, "finding": finding}
    verdict["statement"] = (
        f"{s['picked']} beats the naive baseline out of sample."
        if s["passed"]
        else "No signal found: the naive baseline is published."
    )
    extra = {"verdict": verdict}
    board = pd.read_parquet(config.AUTOML_LEADERBOARD)
    board.insert(0, "rank", range(1, len(board) + 1))
    write_table(M, "leaderboard", board, extra=extra, **common,
                description="Every model tried, ranked on one-month-ahead forecasts for the "
                "validation year. The naive baselines are always rows.")  # fmt: skip
    hold = pd.read_parquet(config.AUTOML_HOLDOUT)
    hold["published"] = hold["model"] == s["published"]
    write_table(M, "holdout", hold, extra=extra, **common,
                description="The pick and the naive baselines on the holdout months, scored "
                "once after the pick.")  # fmt: skip
    write_table(M, "folds", pd.DataFrame(s["folds"]), extra=extra, **common,
                description="Improvement over the baseline in each quarter of the validation "
                "year. Its standard deviation is the noise the holdout has to clear.")  # fmt: skip
    fc = pd.DataFrame()
    if s["passed"] and config.AUTOML_FORECASTS.exists():
        cat = config.catalogue()
        fc = pd.read_parquet(config.AUTOML_FORECASTS)
        fc = fc[fc["item_id"].isin(cat)]
        fc = fc.assign(
            country=fc["item_id"].map(lambda i: config.COUNTRIES[cat[i].country]),
            name=fc["item_id"].map(lambda i: cat[i].name),
            label=finding["label"],
        ).rename(columns={"item_id": "series_id"})
    status = {} if len(fc) else {"status": "No forecasts are published, because no model "
                                 "beat the naive baseline out of sample."}  # fmt: skip
    write_table(M, "forecasts", fc, extra={**extra, **status}, **common,
                description="Next month's forecast move for each series. Published only when "
                "the model passed the out-of-sample test.")  # fmt: skip


def main() -> None:
    if not config.OBSERVATIONS.exists():
        _empty("The macro series have not been collected yet.", {})
        print("macro: nothing collected yet; wrote empty tables")
        return
    obs, retrieved = store.load()
    export_flags(obs, retrieved)
    export_forecasts(retrieved)
    print(f"macro: exported tables for {obs['series_id'].nunique()} series")


if __name__ == "__main__":
    main()
