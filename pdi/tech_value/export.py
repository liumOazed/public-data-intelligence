"""Export step for Tech Value: everything the site shows, as files under site/data/v1/tech-value/.

Runs the plain model (seconds), reads AutoGluon's saved summary if there is one, and joins
prices to benchmarks. Missing inputs produce empty tables with an explanation in the metadata
rather than a failed build, so the site can say "not collected yet" truthfully.
"""

from __future__ import annotations

import json

import pandas as pd

from pdi.common.export import write_table
from pdi.tech_value import config, model, value

M = config.MODULE
SPEC_LIMITS = [
    "SPEC results are submitted by hardware vendors and are dominated by servers.",
    "Each result is one tested configuration: memory, compiler and operating system differ "
    "between results for the same processor.",
    "Scores are SPECrate 2017 Integer base. No other suite or metric is mixed in.",
]
PRICE_LIMITS = [
    "Prices are listing prices at one US retailer (Best Buy), not transaction prices.",
    "The price shown is the cheapest new listing available online on the collection date.",
    "Score per dollar measures rendering throughput per dollar. It is a research lead, "
    "not buying advice.",
]
BLENDER_LIMITS = [
    "Blender Open Data is crowd-submitted: cooling, memory and drivers are not controlled.",
    "Scores come from a single Blender version and are never combined across versions.",
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
        extra={"r_squared": float(plain.result.rsquared), "n_results": int(len(df))},
        **common,
    )
    ranking = model.residual_ranking(model.out_of_fold_residuals(df))
    write_table(
        M,
        "residuals",
        ranking,
        description="Per processor: median out-of-fold residual of the plain regression, with "
        "the tested configuration of a typical result.",
        extra={"min_results": config.MIN_RESULTS_FOR_RANK, "verdict": ev.verdict},
        **common,
    )


def export_value(blender: pd.DataFrame, prices: pd.DataFrame) -> None:
    bench, version = value.benchmark_by_model(blender)
    table = value.value_table(prices, bench)
    got = {"blender_opendata": _retrieved(blender), "bestbuy_products": _retrieved(prices)}
    status = []
    if prices.empty:
        status.append("No retail prices have been collected yet.")
    if blender.empty:
        status.append("Blender Open Data has not been collected yet.")
    extra = {"blender_version": version, "status": " ".join(status) or None}
    common = dict(sources=["bestbuy_products", "blender_opendata"], retrieved_at=got, extra=extra)
    write_table(
        M,
        "value",
        table,
        description="Latest US retail price per model joined to its Blender benchmark score.",
        limitations=PRICE_LIMITS + BLENDER_LIMITS,
        **common,
    )
    write_table(
        M,
        "price_history",
        value.price_history(prices),
        description="Daily lowest new, online-available listing price per model.",
        limitations=PRICE_LIMITS,
        **common,
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
