"""MLPerf® Inference results with measured system power, from MLCommons' results repositories.

Each inference round publishes ``summary_results.json`` in its GitHub repository
(mlcommons/inference_results_v<round>, Apache 2.0). Rows that went through MLPerf Power carry
``has_power`` and the measured *whole-system* power in ``Power_Result``. Only Closed-division,
verified rows with power in watts are kept, because MLCommons' messaging guidelines allow
perf-per-watt only from that official measured power and only between compatible results
(same benchmark and scenario).

    python -m pdi.tech_value.mlperf
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from pdi.common.fetch import CollectError, fetch, read_meta
from pdi.tech_value import config

URL = "https://raw.githubusercontent.com/mlcommons/inference_results_v{round}/main/summary_results.json"
REPO = "https://github.com/mlcommons/inference_results_v{round}/tree/main/{location}"
# Rounds since power rows have carried a per-row "has_power" flag. A round without a
# summary file yet is skipped; a new round is added here when MLCommons publishes it.
ROUNDS = ("5.0", "5.1", "6.0")
COLUMNS = [
    "round", "result_id", "submitter", "system", "accelerator", "n_accelerators", "processor",
    "suite", "availability", "model", "scenario", "performance", "performance_units", "watts",
    "perf_per_watt", "perf_per_watt_units", "source_url", "retrieved_at",
]  # fmt: skip
# Throughput units over watts give work per joule.
PER_JOULE = {"Tokens/s": "tokens per joule", "Samples/s": "samples per joule",
             "Queries/s": "queries per joule"}  # fmt: skip


def parse(records: list[dict], round_: str, retrieved_at: str | None = None) -> pd.DataFrame:
    rows = []
    for r in records:
        if not r.get("has_power") or r.get("Category") != "closed":
            continue
        if r.get("Power_Units") != "Watts" or r.get("Performance_Units") not in PER_JOULE:
            continue  # energy per query (millijoules) belongs to latency scenarios
        if r.get("inferred") or not r.get("compliance", 1):
            continue
        watts = float(r["Power_Result"])
        perf = float(r["Performance_Result"])
        if watts <= 0 or perf <= 0:
            continue
        rows.append(
            {
                "round": round_,
                "result_id": r["ID"],
                "submitter": r["Submitter"],
                "system": r["System"],
                "accelerator": r.get("Accelerator") or "",
                "n_accelerators": int(r.get("a#") or 0),
                "processor": r.get("Processor") or "",
                "suite": r.get("Suite"),
                "availability": r.get("Availability"),
                "model": r["Model"],
                "scenario": r["Scenario"],
                "performance": perf,
                "performance_units": r["Performance_Units"],
                "watts": watts,
                "perf_per_watt": perf / watts,
                "perf_per_watt_units": PER_JOULE[r["Performance_Units"]],
                "source_url": REPO.format(round=round_, location=r.get("Location", "")),
                "retrieved_at": retrieved_at,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def drop_accuracy_twins(df: pd.DataFrame) -> pd.DataFrame:
    """Drop a "-99.9" row whose "-99" twin is the same run (same system, scenario and result).

    MLPerf lists one run under both accuracy targets when it meets the stricter one, so the
    stricter copy repeats the same numbers.
    """
    if df.empty:
        return df
    base = df["model"].str.replace(r"-99\.9$", "-99", regex=True)
    key = pd.DataFrame({"m": base, "s": df["system"], "c": df["scenario"],
                        "p": df["performance"].round(3), "r": df["round"]})  # fmt: skip
    twin = df["model"].str.endswith("-99.9") & key.duplicated(keep=False)
    return df[~twin].reset_index(drop=True)


def comparable_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Benchmark and scenario pairs measured on two or more systems: the only fair comparisons."""
    if df.empty:
        return df
    n = df.groupby(["model", "scenario"])["system"].transform("nunique")
    return df[n >= 2].sort_values(["model", "scenario", "perf_per_watt"], ascending=[1, 1, 0])


def load() -> pd.DataFrame:
    path = config.MLPERF_TABLE
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)


def collect(today: date | None = None) -> pd.DataFrame:
    today = today or date.today()
    frames = []
    for round_ in ROUNDS:
        dest = config.RAW_DIR / "mlperf" / f"{today.isoformat()}_v{round_}.json"
        try:
            path = fetch(URL.format(round=round_), dest, retries=2)
        except CollectError as exc:
            if "404" in str(exc):
                print(f"MLPerf Inference v{round_}: no summary file yet")
                continue
            raise
        records = json.loads(path.read_text())
        part = parse(records, round_, read_meta(path)["retrieved_at"])
        print(f"MLPerf Inference v{round_}: {len(part)} measured-power results")
        if len(part):
            frames.append(part)
    if not frames:
        raise CollectError("No MLPerf Inference round has measured-power results")
    df = drop_accuracy_twins(pd.concat(frames, ignore_index=True))
    config.MLPERF_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.MLPERF_TABLE, index=False)
    return df


def main() -> None:
    df = collect()
    print(
        f"MLPerf Inference Power: {len(df)} measured results from {df['system'].nunique()} systems"
    )


if __name__ == "__main__":
    main()
