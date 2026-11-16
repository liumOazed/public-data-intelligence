"""Export step for Tech Value: everything the site shows, as files under site/data/v1/tech-value/.

Runs the plain model (seconds), reads AutoGluon's saved summary if there is one, and joins
prices to benchmarks. Missing inputs produce empty tables with an explanation in the metadata
rather than a failed build, so the site can say "not collected yet" truthfully.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd

from pdi.common import findings
from pdi.common.export import write_table
from pdi.tech_value import config, fair_price, model, value

M = config.MODULE
SPEC_LIMITS = [
    "SPEC results are submitted by hardware vendors and are dominated by servers.",
    "Each result is one tested configuration: memory, compiler and operating system differ "
    "between results for the same processor.",
    "Scores are SPECrate 2017 Integer base. No other suite or metric is mixed in.",
    "Predicted scores and residuals are this site's estimates derived from SPEC results. "
    "They are not SPEC metrics and were not produced by SPEC-compliant runs.",
    "SPEC, SPEC CPU and SPECrate are registered trademarks of the Standard Performance "
    "Evaluation Corporation. The licensee behind each result is named in its SPEC disclosure, "
    "linked from every row.",
]
PRICE_LIMITS = [
    "Prices are listing prices at the retailers named, not transaction prices. Coverage is "
    "limited to retailers that allow automated collection, so this is not the whole market.",
    "The price shown is the cheapest new listing available online on the collection date.",
    "Amazon prices count only listings sold by Amazon.com or the part's own brand, and "
    "Best Buy prices only listings sold by Best Buy itself. "
    "Marketplace sellers are left out because their listings cannot be confirmed as new.",
    "A price backed by a single listing is marked low confidence: it is often one "
    "marketplace seller's asking price for a discontinued part.",
    "Score per dollar measures rendering throughput per dollar. It is a research lead, "
    "not buying advice.",
]
BLENDER_LIMITS = [
    "Blender Open Data is crowd-submitted: cooling, memory and drivers are not controlled.",
    "Scores come from a single Blender version and are never combined across versions.",
]
FAIR_LIMITS = [
    "A fair price here means the price predicted from Blender rendering score alone. A part "
    "can cost more for reasons the benchmark does not measure, such as gaming performance, "
    "memory size or power draw.",
    "Graphics cards are scored on different Blender backends (OptiX, HIP, oneAPI), which "
    "favours NVIDIA cards, so AMD and Intel cards look more expensive for their score.",
    "The fit uses a few dozen parts. Gaps under the stated typical error are noise.",
]


def _retrieved(df: pd.DataFrame) -> str | None:
    if df.empty or "retrieved_at" not in df.columns:
        return None
    return str(df["retrieved_at"].max())


def export_model(spec: pd.DataFrame) -> None:
    got = {"spec_cpu2017": _retrieved(spec)}
    common = dict(sources=["spec_cpu2017"], retrieved_at=got, limitations=SPEC_LIMITS)
    automl = None
    if config.AUTOML_SUMMARY.exists():
        automl = json.loads(config.AUTOML_SUMMARY.read_text())

    if spec.empty:
        note = {"status": "SPEC results have not been collected yet."}
        for table in ("leaderboard", "holdout", "folds", "coefficients", "residuals"):
            write_table(M, table, pd.DataFrame(), description=note["status"], extra=note, **common)
        return

    df = model.prepare(spec)
    ev = model.evaluate(df, automl)
    spec_finding = findings.from_test(
        ev.verdict["beats_baseline"],
        passed="The published model predicts scores for the newest processor generation, "
        "which it never saw, better than the cores-times-clock baseline by more than the "
        "fold-to-fold spread.",
        failed="No model beat the cores-times-clock baseline on the newest generation, so "
        "nothing here is a forecast.",
        otherwise=findings.EXPLANATION,
    )
    ev.verdict["finding"] = spec_finding
    split = {
        "target": "log of SPECrate 2017 Integer base score",
        "valid_start": config.VALID_START,
        "holdout_start": config.HOLDOUT_START,
        "counts": ev.counts,
        "automl": (
            {k: v for k, v in automl.items() if k not in ("leaderboard", "holdout")}
            if automl
            else None
        ),
    }
    write_table(
        M,
        "leaderboard",
        ev.leaderboard,
        description="Validation-period error of every model tried. The naive baseline is "
        "always a row. Lower is better.",
        extra=split,
        **common,
    )
    write_table(
        M,
        "holdout",
        ev.holdout,
        description="Error on the newest generation, scored once after the pick was made on "
        "validation.",
        extra={**split, "verdict": ev.verdict},
        **common,
    )
    write_table(
        M,
        "folds",
        ev.folds,
        description="Expanding-window folds before the holdout. The spread of the improvement "
        "column is the noise level the holdout result has to clear.",
        extra={"fold_spread": ev.fold_spread},
        **common,
    )
    plain = model.Plain.fit(df)
    write_table(
        M,
        "coefficients",
        plain.coefficients(),
        description="Plain regression fitted on all results. Features are in logs, so a "
        "coefficient on log_cores is an elasticity.",
        extra={
            "r_squared": float(plain.result.rsquared),
            "n_results": int(len(df)),
            "finding": findings.explanation(
                "Coefficients describe how score moves with each specification across the "
                "results the regression was fitted on. They are a description, not a forecast."
            ),
        },
        **common,
    )
    ranking = model.residual_ranking(model.out_of_fold_residuals(df))
    write_table(
        M,
        "residuals",
        ranking,
        description="Per processor: median out-of-fold residual of the plain regression, with "
        "the tested configuration of a typical result.",
        extra={
            "min_results": config.MIN_RESULTS_FOR_RANK,
            "verdict": ev.verdict,
            "finding": findings.explanation(
                "A residual is the part of a processor's score its specifications do not "
                "account for, using the readable plain regression. It points at what does not "
                "fit; it does not forecast anything."
            ),
        },
        **common,
    )


UNUSUAL_RULE = (
    "a model's lowest listed price on the latest collection day is 5% or more above or below "
    "its price seven days earlier, or its earliest price within those seven days."
)
PRICE_MOVE_SHARE = 0.05
PRICE_WINDOW_DAYS = 7
PRICE_FINDING = findings.observation("Listed prices as collected, compared across days.")


def unusual_items(history: pd.DataFrame) -> list[dict]:
    if history.empty:
        return []
    h = history.assign(date=pd.to_datetime(history["date"]))
    latest = pd.Timestamp(h["date"].max())
    start = latest - timedelta(days=PRICE_WINDOW_DAYS)
    items = []
    for (kind, part), g in h.groupby(["kind", "model_key"]):
        now = g[g["date"] == latest]
        before = g[(g["date"] >= start) & (g["date"] < latest)].sort_values("date")
        if now.empty or before.empty:
            continue
        p1, p0 = float(now["price_usd"].iloc[0]), float(before["price_usd"].iloc[0])
        d0 = before["date"].iloc[0]
        change = p1 / p0 - 1
        if abs(change) < PRICE_MOVE_SHARE:
            continue
        direction = "rose" if change > 0 else "fell"
        sentence = (
            f"{part} ({kind.upper()}): the lowest listed price {direction} "
            f"{abs(100 * change):.1f}% to ${p1:,.2f} on {latest:%-d %b %Y}, "
            f"from ${p0:,.2f} on {d0:%-d %b}."
        )
        items.append(findings.item(
            module=M, subject=part, period=latest, sentence=sentence, rule="price move",
            measure=change, threshold=PRICE_MOVE_SHARE, finding=PRICE_FINDING,
            page="tech-value/index.qmd", table="tech-value/price_history.json",
        ))  # fmt: skip
    return items


def export_value(blender: pd.DataFrame, prices: pd.DataFrame) -> None:
    bench, version = value.benchmark_by_model(blender)
    table = value.value_table(prices, bench)
    price_sources = sorted(prices["source"].dropna().unique()) if not prices.empty else []
    got = {"blender_opendata": _retrieved(blender)}
    for key in price_sources:
        got[key] = _retrieved(prices[prices["source"] == key])
    status = []
    if prices.empty:
        status.append("No retail prices have been collected yet.")
    if blender.empty:
        status.append("Blender Open Data has not been collected yet.")
    extra = {"blender_version": version, "status": " ".join(status) or None}
    common = dict(sources=[*price_sources, "blender_opendata"], retrieved_at=got, extra=extra)
    observed = findings.observation(
        "Measured benchmark score divided by the lowest listed price. No model is involved."
    )
    write_table(
        M,
        "value",
        table,
        description="Latest US retail price per model joined to its Blender benchmark score.",
        limitations=PRICE_LIMITS + BLENDER_LIMITS,
        **{**common, "extra": {**extra, "finding": observed}},
    )
    fair = fair_price.evaluate(table)
    fair_extra = {
        **extra,
        "verdicts": fair.verdicts,
        "gap_pct_threshold": config.FAIR_PRICE_GAP_PCT,
    }
    write_table(
        M,
        "fair_price",
        fair.parts,
        description="Per part: the price its Blender score would predict, and how far the "
        "lowest listing is from it. Predictions for parts with two or more listings come "
        "from a fit that left that part out.",
        limitations=PRICE_LIMITS + BLENDER_LIMITS + FAIR_LIMITS,
        **{**common, "extra": fair_extra},
    )
    write_table(
        M,
        "fair_price_leaderboard",
        fair.leaderboard,
        description="Leave-one-out error of each fair price model, per kind of part, with "
        "the naive baseline.",
        limitations=FAIR_LIMITS,
        **{**common, "extra": fair_extra},
    )
    history = value.price_history(prices)
    findings.write_items(
        M, unusual_items(history), rule=UNUSUAL_RULE, sources=common["sources"],
        retrieved_at=got, limitations=PRICE_LIMITS, extra=extra,
    )  # fmt: skip
    write_table(
        M,
        "price_history",
        history,
        description="Daily lowest new, online-available listing price per model.",
        limitations=PRICE_LIMITS,
        **{
            **common,
            "extra": {
                **extra,
                "finding": findings.observation("Listed prices as collected, day by day."),
            },
        },
    )
    write_table(
        M,
        "coverage",
        value.coverage(prices, table),
        description="How many retail listings were recognised as a model, and how many of "
        "those models have a benchmark score.",
        limitations=PRICE_LIMITS,
        **common,
    )
    write_table(
        M,
        "benchmarks",
        bench,
        description="Blender benchmark median per recognised model, whether or not it has a price.",
        sources=["blender_opendata"],
        retrieved_at=got,
        limitations=BLENDER_LIMITS,
        extra=extra,
    )


def main() -> None:
    spec = pd.read_parquet(config.SPEC_TABLE) if config.SPEC_TABLE.exists() else pd.DataFrame()
    blender = (
        pd.read_parquet(config.BLENDER_TABLE) if config.BLENDER_TABLE.exists() else pd.DataFrame()
    )
    export_model(spec)
    export_value(blender, value.load_prices())
    print(f"exported {M} tables")


if __name__ == "__main__":
    main()
