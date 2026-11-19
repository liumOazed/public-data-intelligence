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
from pdi.tech_value import catalogue, config, fair_price, gwpg, model, value

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
GWPG_LIMITS = [
    "SPEC GWPG results (SPECviewperf 15/2020/13/12.1, SPECworkstation 4.0/3.1/3.0, and "
    "SPECapc Maya/Solidworks/Creo/3ds Max) are submitted by workstation OEMs (Dell, HP, "
    "Lenovo, Fujitsu, MIFCOM) and GPU vendors.",
    "Because SPEC GWPG spans multiple benchmark generations and resolutions (1080p vs 4K), "
    "each suite-resolution cohort is calibrated onto the SPECviewperf 2020 v3 FHD/QHD scale "
    "against physical hardware throughput before leave-one-GPU-out residuals are computed.",
    "Predicted scores and residuals are this site's estimates derived from SPEC GWPG disclosures. "
    "They are not official SPEC metrics.",
    "SPEC, SPECviewperf, SPECworkstation, and SPECapc are trademarks or registered trademarks "
    "of the Standard Performance Evaluation Corporation. Every row links to its official SPEC "
    "disclosure.",
]
PRICE_LIMITS = [
    "Prices are listing prices at the retailers named, not transaction prices. Coverage is "
    "limited to retailers that allow automated collection, so this is not the whole market.",
    "The price shown is the cheapest new listing available online on the collection date.",
    "Amazon prices count only listings sold by Amazon.com, Micro Center, or the part's "
    "own brand or board partner, and Best Buy prices only listings sold by Best Buy or "
    "Newegg Official. Other marketplace sellers are left out because their listings cannot "
    "be confirmed as new.",
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
        for table in (
            "leaderboard",
            "holdout",
            "folds",
            "coefficients",
            "residuals",
            "residual_validation",
        ):
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
    validation = model.residual_validation(ranking)
    write_table(
        M,
        "residual_validation",
        validation,
        description="Held-out error of the residual models for processors: one median residual "
        "per ranked processor, compared with the cores-times-clock baseline, overall and by "
        "vendor, family and first-available year.",
        extra={
            "interval": "80% range from the 10th and 90th percentiles of every other "
            "processor's held-out residual",
            "finding": findings.explanation(
                "Typical error of predictions made without the chip being predicted. It says "
                "how far to trust a single chip's residual."
            ),
        },
        **common,
    )
    write_table(
        M,
        "residuals",
        ranking,
        description="Per processor: median leave-one-chip-out residual of three models, "
        "predicted scores per socket with 80% ranges, evidence counts, and the tested "
        "configuration of a typical result.",
        extra={
            "min_results": config.MIN_RESULTS_FOR_RANK,
            "spec_source": "Specifications are read from each SPEC disclosure.",
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
    history = value.price_history(prices)
    fair = fair_price.evaluate(table, history=history)
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
    findings.write_items(
        M,
        unusual_items(history),
        rule=UNUSUAL_RULE,
        write=write_table,
        sources=common["sources"],
        retrieved_at=got,
        limitations=PRICE_LIMITS,
        extra=extra,
    )
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


def export_gpu_model(gwpg_df: pd.DataFrame) -> None:
    got = {"spec_gwpg": _retrieved(gwpg_df)}
    common = dict(sources=["spec_gwpg"], retrieved_at=got, limitations=GWPG_LIMITS)
    if gwpg_df.empty:
        note = {"status": "SPEC GWPG GPU results have not been collected yet."}
        write_table(
            M, "gpu_residuals", pd.DataFrame(), description=note["status"], extra=note, **common
        )
        return

    oof = gwpg.out_of_fold_gpu_residuals(gwpg_df)
    gwpg_df = gwpg.with_specs(gwpg_df)
    ranking = gwpg.gpu_residual_ranking(oof)
    write_table(
        M,
        "gpu_residual_validation",
        gwpg.gpu_residual_validation(ranking),
        description="Held-out error of the residual models for GPUs: one median residual per "
        "ranked GPU, compared with the shaders-times-clock baseline, overall and by vendor, form "
        "factor, architecture and release year.",
        extra={
            "interval": "80% range from the 10th and 90th percentiles of every other GPU's "
            "held-out residual",
            "finding": findings.explanation(
                "Typical error of predictions made without the GPU being predicted. With few "
                "GPUs per group, these errors are rough."
            ),
        },
        **common,
    )
    write_table(
        M,
        "gpu_residuals",
        ranking,
        description="Per GPU: median leave-one-GPU-out residual of three models across SPEC GWPG "
        "disclosures (SPECviewperf, SPECworkstation, SPECapc), predicted calibrated scores with "
        "80% ranges, evidence counts and spread across submitted configurations.",
        extra={
            "min_results": config.GWPG_MIN_RESULTS_FOR_RANK,
            "n_disclosures": int(len(gwpg_df)),
            "n_suites": int(gwpg_df["suite"].nunique()),
            "spec_source": "Shader count, clock, VRAM and bandwidth come from a specification "
            "table maintained in this repository (pdi/tech_value/gwpg.py), not from SPEC.",
            "finding": findings.explanation(
                "A GPU residual is the part of a graphics card's calibrated SPEC GWPG score its "
                "shader count, boost clock, VRAM capacity and memory bandwidth do not account "
                "for. It flags cards to look at more closely; it does not say why."
            ),
        },
        **common,
    )


def main() -> None:
    spec = pd.read_parquet(config.SPEC_TABLE) if config.SPEC_TABLE.exists() else pd.DataFrame()
    gwpg_df = pd.read_parquet(config.GWPG_TABLE) if config.GWPG_TABLE.exists() else pd.DataFrame()
    blender = (
        pd.read_parquet(config.BLENDER_TABLE) if config.BLENDER_TABLE.exists() else pd.DataFrame()
    )
    export_model(spec)
    export_gpu_model(gwpg_df)
    export_value(blender, value.load_prices())
    export_catalogue(spec, gwpg_df, blender)
    print(f"exported {M} tables")


def export_catalogue(spec: pd.DataFrame, gwpg_df: pd.DataFrame, blender: pd.DataFrame) -> None:
    """One row per CPU or GPU seen anywhere, with its coverage on each Tech Value page."""
    from pdi import site

    def table(name: str) -> pd.DataFrame:
        return site.load(M, name)[0]

    dates = {
        "spec_cpu2017": _retrieved(spec),
        "spec_gwpg": _retrieved(gwpg_df),
        "blender_opendata": _retrieved(blender),
    }
    cat = catalogue.build(
        value=table("value"),
        fair=table("fair_price"),
        benchmarks=table("benchmarks"),
        cpu_rank=table("residuals"),
        gpu_rank=table("gpu_residuals"),
        spec=spec,
        gwpg=gwpg_df,
        dates=dates,
    )
    counts = cat.groupby(["kind", "coverage_status"]).size() if not cat.empty else pd.Series()
    write_table(
        M,
        "catalogue",
        cat,
        description="Every CPU and GPU seen in a Blender benchmark, a retail price, a SPEC CPU "
        "2017 result or a SPEC GWPG disclosure, with the pages that show it and why it is or "
        "is not ranked on the over- and under-performer page.",
        sources=["spec_cpu2017", "spec_gwpg", "blender_opendata"],
        retrieved_at=dates,
        extra={
            "status_counts": {f"{k}|{s}": int(n) for (k, s), n in counts.items()},
            "finding": findings.observation(
                "Coverage as collected: which products have which measurements."
            ),
        },
    )


if __name__ == "__main__":
    main()
