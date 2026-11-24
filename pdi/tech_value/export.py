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
from pdi.tech_value import (
    blender_model,
    catalogue,
    config,
    cpu_specs,
    efficiency,
    efficiency_ml,
    efficiency_signals,
    electricity,
    fair_price,
    gwpg,
    mlperf,
    model,
    specpower,
    value,
)

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
    "Because SPEC GWPG spans multiple benchmark generations, OS families and resolutions, each "
    "suite-OS-resolution cohort is calibrated onto the SPECviewperf 2020 v3 Windows FHD/QHD scale "
    "against physical hardware throughput, refitted inside every leave-one-GPU-out fold. Only "
    "cohorts that pass the held-out calibration check (gpu_calibration) feed the ranking.",
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
BLENDER_MODEL_LIMITS = [
    *BLENDER_LIMITS,
    "The Blender residual model is separate from the SPEC models: it predicts a Blender "
    "rendering score from specifications, and its verdicts say nothing about SPEC workloads.",
    "Rendering rewards cores, clocks and power limits; cache-heavy parts such as AMD's X3D "
    "chips can look like underperformers here and still lead in games.",
    "CPU and GPU specifications come from tables maintained in this repository "
    "(pdi/tech_value/cpu_specs.py, pdi/tech_value/gwpg.py), entered from manufacturers' "
    "reference specifications. Factory-overclocked cards are measured but share one spec row.",
    "Predicted scores and residuals are this site's estimates. Blender Open Data is CC0; the "
    "Blender Foundation is credited and does not endorse these estimates.",
]
EFFICIENCY_LIMITS = [
    "Every power figure is rated, not measured: CPU TDP (Intel's processor base power) and GPU "
    "reference board power, entered by hand from manufacturers' specifications "
    "(pdi/tech_value/cpu_specs.py, pdi/tech_value/gpu_specs.py). Real draw under a Blender "
    "render can be 30% or more away from the label, and Intel desktop chips boost well above "
    "their base power.",
    "Performance is the Blender Open Data median score (samples per minute) in one Blender "
    "version. CPU and GPU scores use different backends and are never compared with each other.",
    "Laptop chips and Apple silicon are left out: their power limit is set by the laptop maker "
    "or not published.",
    "The cost calculator charges electricity only for the hours under load, because no idle "
    "power figure exists yet. The default rate is the BLS US city average price per kWh.",
]
EFFICIENCY_ML_LIMITS = [
    "The power these models learn from is rated, not measured, so 'efficiency' means points per "
    "rated watt and an estimated power figure is an estimated rating. The same code learns from "
    "measured desktop power once there is some.",
    "Process nodes are the foundries' marketing names, used as a rough density class "
    "(pdi/tech_value/process_nodes.py). Intel 7 counts as 7 nm and TSMC 4N as 5 nm.",
    "Every prediction comes from models refitted without the chip and without any chip with "
    "identical specifications. The 80% range is the spread of every other chip's held-out "
    "error, so it is checked, not assumed.",
]
SIGNAL_LIMITS = [
    "Signals describe how fast the past moved. None is a forecast.",
    "Launch years with fewer than 3 benchmarked chips are left out of the yearly series.",
    "The price signal uses today's prices for chips of every launch year, not launch prices.",
    "Apple silicon is not on this page, so no gap involving Apple is measured.",
]
MLPERF_LEGEND = (
    "The MLPerf name and logo are registered and unregistered trademarks of MLCommons "
    "Association in the United States and other countries. All rights reserved. Unauthorized "
    "use strictly prohibited. See www.mlcommons.org for more information."
)
MLPERF_LIMITS = [
    "MLPerf® Inference Closed division, datacenter suite, results with an MLPerf Power "
    "measurement only. Results verified by MLCommons Association.",
    "Power is the measured power of the whole system under test (accelerators, host CPUs, "
    "memory, fans), not of one chip.",
    "Work per joule (throughput divided by measured system power) is derived by this site. It "
    "is not an official MLPerf metric. Results are compared only within the same benchmark and "
    "scenario, across rounds MLCommons lists as compatible.",
    "Few submitters measure power, so most accelerators have no result here.",
    MLPERF_LEGEND,
]
SPECPOWER_LIMITS = [
    "SPECpower_ssj2008 measures the AC power of a whole server at each target load from 100% "
    "down to active idle, with a calibrated power analyser. It is a measurement, not a label.",
    "Only results whose disclosure names the processor model are kept, and they are grouped "
    "by that processor. Memory, power supplies and node count differ between results.",
    "ssj_ops per watt is SPEC's own overall metric. Per-processor medians are this site's "
    "summaries and are not SPEC metrics.",
    "SPEC and SPECpower are registered trademarks of the Standard Performance Evaluation "
    "Corporation. Every row links to its SPEC disclosure.",
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
    ranked = export_efficiency(bench, table, version, got)
    sp_trend = export_measured_power()
    # The efficiency signals cite SPECpower too, so they need its retrieval date as well as
    # Blender's; without it the heartbeat reads the source as never retrieved.
    sp = specpower.load()
    sp_got = {"spec_power": sp["retrieved_at"].max() if len(sp) else None}
    export_efficiency_ml(bench, ranked, history, sp_trend, version, {**got, **sp_got})


def export_efficiency(
    bench: pd.DataFrame, table: pd.DataFrame, version: str | None, got: dict
) -> pd.DataFrame:
    ranked, missing = efficiency.build(bench, table)
    elec = electricity.load()
    rate = electricity.latest_rate(elec)
    sources = ["blender_opendata", "fred"]
    retrieved = {**got, "fred": rate["retrieved_at"] if rate else None}
    status = None if len(ranked) else "Blender Open Data has not been collected yet."
    extra = {
        "blender_version": version,
        "status": status,
        "summary": efficiency.summary(ranked, rate),
        "finding": findings.observation(
            "Measured Blender score divided by the manufacturer's rated power. No model is "
            "involved, and the power figure is a label, not a measurement."
        ),
    }
    common = dict(sources=sources, retrieved_at=retrieved, limitations=EFFICIENCY_LIMITS)
    write_table(
        M,
        "efficiency",
        ranked,
        description="Blender score per rated watt, energy and EDP for a fixed render job, per "
        "model with both a score and a rated power figure.",
        extra=extra,
        **common,
    )
    write_table(
        M,
        "efficiency_trend",
        efficiency.year_trend(ranked),
        description="Median and best Blender score per rated watt by launch year.",
        extra={"blender_version": version, "status": status},
        **common,
    )
    write_table(
        M,
        "efficiency_missing",
        missing,
        description="Benchmarked models left off the efficiency page, with the reason.",
        extra={"blender_version": version, "status": status},
        **common,
    )
    return ranked


def export_efficiency_ml(
    bench: pd.DataFrame,
    ranked: pd.DataFrame,
    history: pd.DataFrame,
    sp_trend: pd.DataFrame,
    version: str | None,
    got: dict,
) -> None:
    built = efficiency_ml.build(bench, ranked)
    val = built["validation"]
    status = None if len(built["frame"]) else "Blender Open Data has not been collected yet."
    beats = {
        (k, t): efficiency_ml.beats_baseline(val, k, t)
        for k in ("cpu", "gpu")
        for t in ("rated power", "points per rated watt")
    }

    def label(target: str) -> dict:
        ok = all(beats[(k, target)] for k in ("cpu", "gpu") if (val["kind"] == k).any())
        return findings.from_test(
            ok and len(val) > 0,
            passed="Held-out predictions from models refitted without each chip beat the "
            "launch-year-only baseline for every kind.",
            failed="The models did not beat the launch-year-only baseline on held-out chips "
            "for every kind.",
            otherwise=findings.EXPLANATION,
        )

    common = dict(sources=["blender_opendata"], retrieved_at=got, limitations=EFFICIENCY_ML_LIMITS)
    base = {"blender_version": version, "status": status}
    write_table(
        M,
        "efficiency_power_estimates",
        built["power"],
        description="Power figure each chip's specifications predict, from models refitted "
        "without it, with an 80% range. 'Estimated' rows have no figure of their own; "
        "'Held-out check' rows show how close the estimate comes where the figure is known.",
        extra={**base, "finding": label("rated power"), "target": "rated power"},
        **common,
    )
    write_table(
        M,
        "efficiency_residuals",
        built["residuals"],
        description="Per chip: points per rated watt its specifications, process node, "
        "architecture and rated power predict, from models refitted without it, and the "
        "verdict when the measured value sits outside the 80% range.",
        extra={**base, "finding": label("points per rated watt")},
        **common,
    )
    write_table(
        M,
        "efficiency_model_validation",
        val,
        description="Typical held-out error of each model, against a launch-year-only baseline.",
        extra=base,
        **common,
    )
    write_table(
        M,
        "efficiency_tiers",
        built["tier_summary"],
        description="Efficiency tiers of the current line-up from a Gaussian mixture on log "
        "points per rated watt, with the number of tiers chosen by BIC.",
        extra={**base, "finding": findings.explanation(
            "Clusters found in the data; a description of the line-up, not a test.")},
        **common,
    )  # fmt: skip
    signals, series = efficiency_signals.build(built["frame"], sp_trend, history)
    write_table(
        M,
        "efficiency_signals",
        signals,
        description="Rates of change in efficiency, rated power, the vendor gap and the price "
        "of efficiency, recent pace against the pace before, and turning points. No forecasts.",
        sources=["blender_opendata", "spec_power"],
        retrieved_at=got,
        limitations=EFFICIENCY_ML_LIMITS + SIGNAL_LIMITS,
        extra=base,
    )
    write_table(
        M,
        "efficiency_signal_series",
        series,
        description="The yearly values each efficiency signal is computed from, and the fitted "
        "line for turning points.",
        sources=["blender_opendata", "spec_power"],
        retrieved_at=got,
        limitations=SIGNAL_LIMITS,
        extra=base,
    )


def export_measured_power() -> pd.DataFrame:
    ml = mlperf.load()
    got = {"mlperf_inference": ml["retrieved_at"].max() if len(ml) else None}
    write_table(
        M,
        "mlperf_power",
        ml.drop(columns=["retrieved_at"], errors="ignore"),
        description="MLPerf® Inference Closed-division results with measured system power, and "
        "the work per joule derived from them.",
        sources=["mlperf_inference"],
        retrieved_at=got,
        limitations=MLPERF_LIMITS,
        extra={
            "status": None if len(ml) else "MLPerf Inference results have not been collected yet.",
            "legend": MLPERF_LEGEND,
        },
    )
    sp = specpower.load()
    got = {"spec_power": sp["retrieved_at"].max() if len(sp) else None}
    trend = specpower.year_trend(sp)
    status = None if len(sp) else "SPECpower results have not been collected yet."
    common = dict(sources=["spec_power"], retrieved_at=got, limitations=SPECPOWER_LIMITS)
    write_table(
        M,
        "specpower",
        sp.drop(columns=["retrieved_at"], errors="ignore"),
        description="SPECpower_ssj2008 results that name their processor: measured server power "
        "at full load and active idle, and ssj_ops per watt at each target load.",
        extra={"status": status},
        **common,
    )
    write_table(
        M,
        "specpower_processors",
        specpower.by_processor(sp),
        description="Per processor: median measured ssj_ops per watt, idle power as a share of "
        "full-load power, and the median efficiency curve across target loads.",
        extra={
            "status": status,
            "doubling_years": specpower.doubling_years(trend),
            "n_results": int(len(sp)),
        },
        **common,
    )
    write_table(
        M,
        "specpower_trend",
        trend,
        description="Best and median measured ssj_ops per watt, and median idle share, by year "
        "of publication.",
        extra={"status": status},
        **common,
    )
    return trend


def export_gpu_model(gwpg_df: pd.DataFrame) -> None:
    got = {"spec_gwpg": _retrieved(gwpg_df)}
    common = dict(sources=["spec_gwpg"], retrieved_at=got, limitations=GWPG_LIMITS)
    if gwpg_df.empty:
        note = {"status": "SPEC GWPG GPU results have not been collected yet."}
        write_table(
            M, "gpu_residuals", pd.DataFrame(), description=note["status"], extra=note, **common
        )
        return

    calibration = gwpg.calibration_validation(gwpg_df)
    used = gwpg.validated_cohorts(calibration)
    if not config.GWPG_VALIDATED_COHORTS_ONLY:
        used = set(calibration["cohort"])
    calibration["used_in_ranking"] = calibration["cohort"].isin(used)
    write_table(
        M,
        "gpu_calibration",
        calibration,
        description="Held-out check of the SPEC GWPG suite calibration: for each suite, OS and "
        "resolution, how far a card's calibrated score there sits from the same card's "
        "calibrated score in the other suites, with the calibration refitted without that card.",
        extra={
            "max_gap_pct": gwpg.CALIBRATION_MAX_GAP_PCT,
            "min_shared_cards": gwpg.CALIBRATION_MIN_OVERLAP,
            "reference": gwpg.REF_COHORT,
            "validated_only": config.GWPG_VALIDATED_COHORTS_ONLY,
            "finding": findings.explanation(
                "A suite is combined into the GPU score only when cards measured in it and "
                f"elsewhere agree within {gwpg.CALIBRATION_MAX_GAP_PCT:.0f}% (median) after "
                "calibration."
            ),
        },
        **common,
    )
    write_table(
        M,
        "gpu_workloads",
        gwpg.workload_table(gwpg_df),
        description="Raw SPEC GWPG composite scores per card within one suite and version, OS "
        "family and resolution. No calibration: each group uses its suite's own metric.",
        extra={
            "finding": findings.observation(
                "Scores as published, grouped so only like-for-like results sit together."
            )
        },
        **common,
    )
    gwpg_df = gwpg.with_specs(gwpg_df)
    ranked_input = gwpg.in_cohorts(gwpg_df, used)
    oof = gwpg.out_of_fold_gpu_residuals(ranked_input)
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
            "n_disclosures_ranked": int(len(ranked_input)),
            "validated_suites_only": config.GWPG_VALIDATED_COHORTS_ONLY,
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


def export_blender_model(blender: pd.DataFrame) -> dict:
    """Blender rendering residuals for CPUs and GPUs, with both validation checks.

    Returns the build result so the catalogue uses only the kinds that passed.
    """
    got = {"blender_opendata": _retrieved(blender)}
    common = dict(sources=["blender_opendata"], retrieved_at=got, limitations=BLENDER_MODEL_LIMITS)
    if blender.empty:
        note = {"status": "Blender Open Data has not been collected yet."}
        for name in ("blender_residuals", "blender_residual_validation", "blender_version_check"):
            write_table(M, name, pd.DataFrame(), description=note["status"], extra=note, **common)
        return {"rank": {}, "passed": {}, "version": None}
    built = blender_model.build(blender)
    passed = built["passed"]
    ranks = [r for r in built["rank"].values() if not r.empty]
    rank = pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame()
    if not rank.empty:
        rank["used_for_assessment"] = rank["kind"].map(
            {k: bool(v["passed"]) for k, v in passed.items()}
        )
    vals = [v for v in built["validation"].values() if not v.empty]
    checks = [c for c in built["check"].values() if not c.empty]
    extra = {
        "blender_version": built["version"],
        "passed": passed,
        "max_gap_pct": gwpg.CALIBRATION_MAX_GAP_PCT,
        "min_shared": gwpg.CALIBRATION_MIN_OVERLAP,
        "min_products": blender_model.MIN_PRODUCTS,
        "metric": blender_model.METRIC,
    }
    write_table(
        M,
        "blender_version_check",
        pd.concat(checks, ignore_index=True) if checks else pd.DataFrame(),
        description="The Blender residual model refitted on other Blender versions: for "
        "products ranked in both, the median gap between their residuals. A source is used for "
        f"assessment only when the gap is at most {gwpg.CALIBRATION_MAX_GAP_PCT:.0f}% on "
        f"every version checked, over at least {gwpg.CALIBRATION_MIN_OVERLAP} shared products.",
        extra={
            **extra,
            "finding": findings.observation("Residuals compared across benchmark versions."),
        },
        **common,
    )
    write_table(
        M,
        "blender_residual_validation",
        pd.concat(vals, ignore_index=True) if vals else pd.DataFrame(),
        description="Held-out error of the Blender residual models, one median residual per "
        "product, against the cores-times-clock (CPU) or shaders-times-clock (GPU) baseline, "
        "overall and by vendor, family, generation and (GPU) backend.",
        extra={
            **extra,
            "finding": findings.explanation(
                "Typical error of predictions made without the product being predicted."
            ),
        },
        **common,
    )
    write_table(
        M,
        "blender_residuals",
        rank,
        description="Per CPU and GPU: measured Blender score against the score predicted from "
        "specifications by fits that left the product (and any product with identical "
        "specifications) out, with 80% ranges and a verdict across three models.",
        extra={
            **extra,
            "finding": findings.explanation(
                "A Blender residual is the part of a product's rendering score that its "
                "specifications do not account for. It flags products to look at more closely; "
                "it does not say why."
            ),
        },
        **common,
    )
    return built


def export_sources(
    spec: pd.DataFrame,
    gwpg_df: pd.DataFrame,
    blender: pd.DataFrame,
    ws_cpu: pd.DataFrame | None = None,
) -> None:
    """Every benchmark source the Tech Value pages use or have considered, with what is recorded."""

    def n(df: pd.DataFrame, col: str) -> int:
        return int(df[col].nunique()) if not df.empty and col in df.columns else 0

    rows = [
        {
            "source": "SPEC CPU 2017",
            "publisher": "Standard Performance Evaluation Corporation (vendor submissions, "
            "reviewed by SPEC)",
            "benchmark": "SPECrate 2017 Integer",
            "metric": "SPECrate2017_int_base and _peak; this site uses base per socket",
            "status": "Collected" if not spec.empty else "Not collected yet",
            "results": int(len(spec)),
            "products": n(spec, "processor"),
            "last_collected": _retrieved(spec),
            "recorded": "OEM, system, chips, cores, threads per core, clock, cache, memory, "
            "compiler, OS, test and publication dates, disclosure link",
            "used_for": "Over & under performers (CPU)",
        },
        {
            "source": "SPEC GWPG",
            "publisher": "SPEC Graphics and Workstation Performance Group (OEM and vendor "
            "submissions)",
            "benchmark": "SPECviewperf 12.1/13/2020/15 (Windows and Linux), SPECworkstation "
            "3.0/3.1/4.0, SPECapc (Solidworks 2022/2024/2025, Siemens NX 2024, Maya 2024, Creo 9, "
            "3ds Max 2020)",
            "metric": "Each suite's GPU measurement (viewset geometric mean, SPECapc GPU "
            "composite or viewport workloads, SPECworkstation graphics subsystem); compared raw "
            "within suite, version, OS and resolution, combined only where the calibration "
            "passes its check",
            "status": "Collected" if not gwpg_df.empty else "Not collected yet",
            "results": int(len(gwpg_df)),
            "products": n(gwpg_df, "processor"),
            "last_collected": _retrieved(gwpg_df),
            "recorded": "OEM, system, CPU, GPU as written, OS, resolution, anti-aliasing, test "
            "date, metric, number of viewsets, composite, best and worst viewset, disclosure link",
            "used_for": "Over & under performers (GPU), results by workload",
        },
        {
            "source": "SPECworkstation CPU subsystem",
            "publisher": "SPEC Graphics and Workstation Performance Group (OEM submissions)",
            "benchmark": "SPECworkstation 3.0/3.1/4.0, CPU subsystem score",
            "metric": "The CPU subsystem score of one SPECworkstation version; never mixed with "
            "SPEC CPU 2017 and not used by the CPU models",
            "status": "Collected"
            if ws_cpu is not None and not ws_cpu.empty
            else "Not collected yet",
            "results": 0 if ws_cpu is None else int(len(ws_cpu)),
            "products": 0 if ws_cpu is None else n(ws_cpu, "processor"),
            "last_collected": None if ws_cpu is None else _retrieved(ws_cpu),
            "recorded": "Submitter, system, CPU and GPU as written, suite version, disclosure link",
            "used_for": "Catalogue measurements for desktop and workstation CPUs SPEC CPU 2017 "
            "does not cover",
        },
        {
            "source": "Blender Open Data",
            "publisher": "Blender Foundation (community-submitted runs)",
            "benchmark": "Blender Benchmark",
            "metric": "Median samples per minute over runs, per device and Blender version",
            "status": "Collected" if not blender.empty else "Not collected yet",
            "results": int(blender["n_runs"].sum()) if not blender.empty else 0,
            "products": n(blender, "device_name"),
            "last_collected": _retrieved(blender),
            "recorded": "Device, CPU or GPU, Blender version, number of runs, median and "
            "interquartile range",
            "used_for": "Value ranking, Fair price, comparisons, and its own over & under "
            "performer models (CPU and GPU), kept apart from the SPEC ones",
        },
        {
            "source": "SPEC CPU 2017 floating point and speed",
            "publisher": "SPEC",
            "benchmark": "SPECrate 2017 Floating Point, SPECspeed 2017",
            "metric": "Would be a separate workload view, never combined with Integer",
            "status": "Not collected yet",
            "results": 0,
            "products": 0,
            "last_collected": None,
            "recorded": "",
            "used_for": "Planned",
        },
        {
            "source": "OpenBenchmarking.org",
            "publisher": "Phoronix Test Suite users (unreviewed)",
            "benchmark": "Per-test results, e.g. pts/blender, pts/build-linux-kernel",
            "metric": "Each test's own unit; compared only within one test, version and setting",
            "status": "Blocked: every page answers automated clients with a Cloudflare "
            "challenge (checked 2026-11-20 from a home PC)",
            "results": 0,
            "products": 0,
            "last_collected": None,
            "recorded": "",
            "used_for": "Not collected: would add consumer CPUs and GPUs SPEC does not cover, "
            "if the operator allows automated access",
        },
        {
            "source": "Geekbench Browser",
            "publisher": "Primate Labs (user uploads)",
            "benchmark": "Geekbench 5 and 6 CPU and GPU compute",
            "metric": "Geekbench score",
            "status": "Not used: the Geekbench licence gives Primate Labs exclusive rights to "
            "uploaded results, and robots.txt disallows its search pages (checked 2026-11-20)",
            "results": 0,
            "products": 0,
            "last_collected": None,
            "recorded": "",
            "used_for": "Not collected",
        },
        {
            "source": "PassMark (cpubenchmark.net, videocardbenchmark.net)",
            "publisher": "PassMark Software (user uploads)",
            "benchmark": "PerformanceTest CPU Mark and G3D Mark",
            "metric": "PassMark score",
            "status": "Not used: PassMark says its data may not be copied without a licence "
            "(checked 2026-11-20)",
            "results": 0,
            "products": 0,
            "last_collected": None,
            "recorded": "",
            "used_for": "Not collected",
        },
        {
            "source": "TechPowerUp GPU and CPU databases",
            "publisher": "TechPowerUp (editorial)",
            "benchmark": "Relative performance (TechPowerUp's own summary of its reviews)",
            "metric": "Percent of a reference card",
            "status": "Not used: no export or reuse terms, and its robots.txt could not be read "
            "by this project's fetcher (checked 2026-11-20)",
            "results": 0,
            "products": 0,
            "last_collected": None,
            "recorded": "",
            "used_for": "Not collected",
        },
        {
            "source": "Cinebench scores in published reviews",
            "publisher": "Review sites",
            "benchmark": "Cinebench R23 and 2024",
            "metric": "Cinebench points",
            "status": "Not used: scattered across copyrighted reviews with no structured source "
            "to collect automatically",
            "results": 0,
            "products": 0,
            "last_collected": None,
            "recorded": "",
            "used_for": "Not collected",
        },
        {
            "source": "Independently published SPEC runs",
            "publisher": "Reviewers and papers (not reviewed by SPEC)",
            "benchmark": "SPEC CPU 2017 and SPECviewperf run outside SPEC's process",
            "metric": "Would be labelled as unofficial and kept apart from SPEC-published results",
            "status": "Not collected yet",
            "results": 0,
            "products": 0,
            "last_collected": None,
            "recorded": "",
            "used_for": "Planned",
        },
    ]
    write_table(
        M,
        "benchmark_sources",
        pd.DataFrame(rows),
        description="Benchmark sources behind the Tech Value pages, collected or planned, with "
        "the metric each reports and the configuration details recorded for every result.",
        sources=["spec_cpu2017", "spec_gwpg", "blender_opendata"],
        retrieved_at={
            "spec_cpu2017": _retrieved(spec),
            "spec_gwpg": _retrieved(gwpg_df),
            "blender_opendata": _retrieved(blender),
        },
        extra={"finding": findings.observation("What each benchmark source contributes.")},
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
    built = export_blender_model(blender)
    ws_cpu = pd.read_parquet(config.WS_CPU_TABLE) if config.WS_CPU_TABLE.exists() else None
    export_catalogue(spec, gwpg_df, blender, built)
    export_sources(spec, gwpg_df, blender, ws_cpu)
    print(f"exported {M} tables")


def export_catalogue(
    spec: pd.DataFrame,
    gwpg_df: pd.DataFrame,
    blender: pd.DataFrame,
    built: dict | None = None,
) -> None:
    """One row per CPU or GPU seen anywhere, with its coverage on each Tech Value page."""
    from pdi import site

    def table(name: str) -> pd.DataFrame:
        return site.load(M, name)[0]

    dates = {
        "spec_cpu2017": _retrieved(spec),
        "spec_gwpg": _retrieved(gwpg_df),
        "blender_opendata": _retrieved(blender),
    }
    calib = table("gpu_calibration")
    if not gwpg_df.empty and not calib.empty and "used_in_ranking" in calib.columns:
        used = set(calib.loc[calib["used_in_ranking"].astype(bool), "cohort"])
        key = gwpg_df["suite_base"] + " | " + gwpg_df["os_family"] + " | " + gwpg_df["res_tier"]
        gwpg_df = gwpg_df.assign(in_ranked_suites=key.isin(used))
    cat = catalogue.build(
        value=table("value"),
        fair=table("fair_price"),
        benchmarks=table("benchmarks"),
        cpu_rank=table("residuals"),
        gpu_rank=table("gpu_residuals"),
        spec=spec,
        gwpg=gwpg_df,
        dates=dates,
        ws_cpu=pd.read_parquet(config.WS_CPU_TABLE) if config.WS_CPU_TABLE.exists() else None,
        blender_rank=_passed_blender_rows(built),
        blender_spec_keys=_blender_spec_keys(built),
    )
    counts = cat.groupby(["kind", "coverage_status"]).size() if not cat.empty else pd.Series()
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    history, first_seen = _read_coverage_store()
    history, first_seen = catalogue.update_history(cat, history, first_seen, today)
    _write_coverage_store(history, first_seen)
    if not cat.empty:
        cat["first_tracked"] = [
            first_seen.get(f"{k}|{m}") if t else None
            for k, m, t in zip(cat["kind"], cat["model_key"], cat["tracked"], strict=True)
        ]
    summary = catalogue.tracked_summary(cat)
    write_table(
        M,
        "coverage_history",
        pd.DataFrame(history),
        description="Tracked CPUs and GPUs (those on Value ranking or Fair price) per update, with "
        "how many the over- and under-performer models assess, how many have a SPEC measurement "
        "that is not assessed, and how many have none.",
        sources=["spec_cpu2017", "spec_gwpg", "blender_opendata"],
        retrieved_at=dates,
        extra={"finding": findings.observation("Coverage of tracked products over time.")},
    )
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
            "tracked": summary,
            "finding": findings.observation(
                "Coverage as collected: which products have which measurements."
            ),
        },
    )


def _passed_blender_rows(built: dict | None) -> pd.DataFrame | None:
    """Blender residual rows of the kinds whose model passed both checks."""
    if not built:
        return None
    parts = [
        r
        for k, r in built.get("rank", {}).items()
        if not r.empty and built["passed"].get(k, {}).get("passed")
    ]
    return pd.concat(parts, ignore_index=True) if parts else None


def _blender_spec_keys(built: dict | None) -> dict[str, set[str]]:
    """Model keys with specifications, for the kinds whose Blender model passed."""
    if not built:
        return {}
    keys = {
        "cpu": set(cpu_specs.CPU_SPECS),
        "gpu": set(blender_model._gpu_spec_by_key()),
    }
    return {k: v for k, v in keys.items() if built["passed"].get(k, {}).get("passed")}


def _read_coverage_store() -> tuple[list[dict], dict[str, str]]:
    if not config.COVERAGE_STORE.exists():
        return [], {}
    d = json.loads(config.COVERAGE_STORE.read_text(encoding="utf-8"))
    return list(d.get("history", [])), dict(d.get("first_seen", {}))


def _write_coverage_store(history: list[dict], first_seen: dict[str, str]) -> None:
    config.COVERAGE_STORE.parent.mkdir(parents=True, exist_ok=True)
    body = {"history": history, "first_seen": dict(sorted(first_seen.items()))}
    config.COVERAGE_STORE.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
